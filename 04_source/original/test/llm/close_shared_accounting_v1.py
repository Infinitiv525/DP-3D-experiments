"""Explicit operational extension: close missing shared-job accounting at full cap.

Does not change frozen scientific sources/identity, fabricate Slurm records or
refund any reservation. 'settled' means budget closed; measurement_kind clearly
distinguishes a conservative charge from a measured scheduler runtime.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from allocation_budget import PhysicalBudget
from main_protocol import read, digest, identity, require_authorization, shared_plan, training_config
from main_execution import budget_config, load_task, verify_runtime_freeze
from main_workflow import atomic, shared_rows

CONSENT = ('User message "Maš povolenie" in this dissertation chat, in reply to '
           'conservative accounting closure without refund and subsequent search-stage launch, '
           'within the existing70GPU-hour cap. No scientific protocol or test-access change.')


def close_at_full_cap(ledger, labels, absence, evidence):
    """Validate the entire transaction before one commit; tests inject evidence."""
    if not absence.get('query_succeeded') or absence.get('user') != ledger.spec['identity']['user']:
        raise ValueError('Successful authenticated scheduler absence query required')
    if len(labels) != len(set(labels)): raise ValueError('Duplicate close labels')
    before = ledger.report()['charged_or_reserved_gpu_seconds']; changes = []
    for label in labels:
        row = next(r for r in ledger.state['allocations'] if r['label'] == label)
        if row['status'] == 'settled': continue
        if not row['job_id'] or row['job_id'] in absence['active_job_ids']: raise RuntimeError('Job absent/finished not established')
        proof = evidence[label]
        if (proof.get('job_id') != row['job_id'] or not proof.get('complete_verified') or
            proof.get('historical_record_unavailable') is not True or proof.get('cluster') != absence['cluster']):
            raise ValueError('Matching completed application and missing-history proof required')
        if row['status'] != 'submitted_on_hold' or row['charged_gpu_seconds'] != row['reservation_gpu_seconds']:
            raise RuntimeError('Only untouched attached reservations can be closed at full cap')
        if row['reservation_gpu_seconds'] != row['gpus'] * (row['wall_limit_seconds'] + 1): raise ValueError('Reservation differs')
        changes.append((row, proof))
    for row, proof in changes:
        row.update(status='settled', measurement_kind='conservative_full_reservation_no_scheduler_runtime',
                   actual_allocated_gpu_seconds=None, refund_gpu_seconds=0,
                   conservative_closure_proof=proof, scheduler_absence_proof=absence)
    if changes: ledger.save()
    if ledger.report()['charged_or_reserved_gpu_seconds'] != before: raise RuntimeError('Unexpected accounting refund/change')
    return [row['label'] for row, proof in changes]


def shared_evidence(root, output, manifest, provenance, ledger):
    jobs = shared_plan(manifest); planned = read(output / 'shared_stage_plan.json'); submitted = read(output / 'shared_submitted.json')
    if len(planned) != 9 or len(submitted['job_ids']) != 9: raise ValueError('Original9job submission map required')
    mapping = {p['label']: job for p, job in zip(planned, submitted['job_ids'], strict=True)}
    records = {j['id']: load_task(output / 'shared' / j['id']) for j in jobs}
    for repeat in range(3): shared_rows(manifest, [records[j['id']] for j in jobs if j['repeat'] == repeat], repeat)
    split = read(root / manifest['dataset_relative'] / 'split.json'); evidence = {}
    for job in jobs:
        label = 'shared:' + job['id']; row = next(r for r in ledger.state['allocations'] if r['label'] == label)
        task = records[job['id']]; folder = output / 'shared' / job['id']; training = folder / 'training'
        summary = read(training / 'summary.json')
        if (row['job_id'] != mapping[job['id']] or task['owner']['mode'] != 'slurm' or task['owner']['job'] != row['job_id']
            or task['owner']['user'] != ledger.spec['identity']['user'] or task['spec']['provenance'] != provenance
            or summary['status'] != 'FLOWBENCH_MAIN_TRIAL_COMPLETE' or summary['pilot_only'] or summary['test_evaluated']
            or summary['config'] != training_config(manifest, job)
            or summary['training_indices'] != sorted(split['indices']['train'])
            or summary['validation_indices'] != sorted(split['indices']['validation'])
            or [h['epoch'] for h in summary['history']] != list(range(1, 21))
            or digest(training / 'summary.json') != task['result']['summary_sha256']
            or digest(training / 'best_model.pt') != summary['checkpoint_sha256']
            or summary['checkpoint_sha256'] != task['result']['checkpoint_sha256']):
            raise ValueError('Completed main evidence differs: ' + label)
        evidence[label] = dict(job_id=row['job_id'], cluster=task['owner']['cluster'], complete_verified=True,
            task_sha256=digest(folder / 'task.json'), summary_sha256=digest(training / 'summary.json'),
            checkpoint_sha256=summary['checkpoint_sha256'], phase='main_shared',
            application_process_seconds=task['seconds'], application_process_time_is_not_scheduler_runtime=True)
    return evidence


def run(a):
    manifest, freeze, approval = require_authorization(a.manifest, a.freeze, a.authorization)
    verify_runtime_freeze(a.root, manifest, freeze)
    a.audit.mkdir(parents=True, exist_ok=False)
    provenance = dict(freeze_sha256=digest(a.freeze), authorization_sha256=digest(a.authorization))
    query = ['squeue', '-h', '-r', '--states=all', '-u', 'lutoke823', '-o', '%i %T']
    snapshot = subprocess.run(query, check=True, capture_output=True, text=True, timeout=20)
    active = []
    for line in snapshot.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2: raise ValueError('Malformed scheduler snapshot')
        active.append(fields[0])
    absence = dict(query_succeeded=True, user='lutoke823', authenticated_uid=os.getuid(),
                   cluster=os.environ['SLURM_CLUSTER_NAME'], command=query, raw_output=snapshot.stdout,
                   active_job_ids=active, checked_at=datetime.now(timezone.utc).isoformat())
    with PhysicalBudget(a.output / 'physical_budget', **budget_config(a.manifest, a.freeze)) as ledger:
        atomic(a.audit / 'allocations_before.json', read(a.output / 'physical_budget/allocations.json'))
        evidence = shared_evidence(a.root, a.output, manifest, provenance, ledger)
        for row in ledger.state['allocations']:
            if row['status'] == 'settled': continue
            if not row['label'].startswith('shared:'): raise ValueError('Extension restricted to original shared jobs')
            result = subprocess.run(['scontrol', 'show', 'job', '-o', row['job_id']], capture_output=True, text=True, timeout=20)
            if result.returncode == 0 or 'Invalid job id specified' not in result.stderr:
                raise RuntimeError('Missing historical record not established; use measured reconciliation if available')
            evidence[row['label']].update(historical_record_unavailable=True, scheduler_query_stderr=result.stderr,
                                          scheduler_query_returncode=result.returncode)
        closed = close_at_full_cap(ledger, ['shared:' + j['id'] for j in shared_plan(manifest)], absence, evidence)
        report = dict(status='SHARED_ACCOUNTING_CLOSED_AT_FULL_RESERVED_CAP', closed_labels=closed,
            original_source_freeze_sha256=digest(a.freeze), source_sha256=digest(Path(__file__)), user_consent_reference=CONSENT,
            charged_or_reserved_gpu_seconds=ledger.report()['charged_or_reserved_gpu_seconds'], refund_gpu_seconds=0,
            measured_scheduler_rows=sum(r['status'] == 'settled' and r.get('measurement_kind') != 'conservative_full_reservation_no_scheduler_runtime' for r in ledger.state['allocations']),
            conservative_rows=sum(r.get('measurement_kind') == 'conservative_full_reservation_no_scheduler_runtime' for r in ledger.state['allocations']),
            note='Closed budget, not reconstructed Slurm measurements. Unknown actual runtime remains unknown; no refund.',
            test_target_values_analyzed=False)
        atomic(a.audit / 'allocations_after.json', read(a.output / 'physical_budget/allocations.json'))
        atomic(a.audit / 'summary.json', report)
    print(report['status'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'manifest', 'freeze', 'authorization', 'output', 'audit'): p.add_argument('--' + key, type=Path, required=True)
    run(p.parse_args())

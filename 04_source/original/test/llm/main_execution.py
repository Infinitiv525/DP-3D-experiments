"""Real process backend and one authorized Slurm worker. Never evaluates test."""
import argparse
import hashlib
import os
from pathlib import Path
import time

from main_protocol import (read, digest, identity, require_authorization, shared_plan,
                           search_plan, final_plan, PARAMETERS)
from main_workflow import atomic, fixed_task, run_search, final_catalog
from allocation_budget import PhysicalBudget
from configurator_contract_v3 import SYSTEM_SHA256, parse_reply, canonical, messages
from configurator_transport_v1 import POLICY
from search_controls import run_bounded_process


def load_task(directory):
    envelope = read(Path(directory) / 'task.json')
    if identity(envelope['payload']) != envelope['sha256']: raise ValueError('Task checksum differs')
    return envelope['payload']


def budget_config(manifest_path, freeze_path):
    return dict(gpu_seconds_limit=252000, identity=dict(user='lutoke823', account='perun2601449',
                manifest_file_sha256=digest(manifest_path), freeze_file_sha256=digest(freeze_path), phase='main'))


def runtime_paths(root, manifest):
    required = ['llm/' + name for name in ('main_protocol.py', 'main_workflow.py', 'main_execution.py', 'main_submit.py',
                'persistent_search.py', 'allocation_budget.py', 'settle_allocation_uid.py', 'search_controls.py',
                'sequential_feedback.py', 'configurator_contract_v3.py', 'configurator_transport_v1.py',
                'propose_configuration_transport_v1.py', 'propose_configuration_v3.py', 'smoke_local_configurator.py')]
    required += ['training/' + name for name in ('train_flowbench_main.py', 'train_flowbench_protocol.py',
                 'train_flowbench_smoke.py', 'flowbench_physics.py', 'flowbench_protocol_metrics.py', 'fno_model.py')]
    required += ['scripts/117_main_worker.sbatch']
    paths = {str((root / 'test' / r).resolve()) for r in required}
    paths.update(str((root / manifest[k]).resolve()) for k in ('llm_config_relative', 'normalization_source_relative'))
    paths.update(str((root / manifest['dataset_relative'] / name).resolve()) for name in ('split.json', 'pressure_gauge_audit.json'))
    paths.update(str((root / manifest['model_preparation_relative'] / (label + '.json')).resolve()) for label in manifest['methods'][2:])
    return paths


def verify_runtime_freeze(root, manifest, freeze):
    paths = runtime_paths(root, manifest)
    if not paths.issubset(freeze['source_files']): raise ValueError('Incomplete main runtime/input source freeze')


def stage_jobs(manifest, stage, output, provenance=None):
    output = Path(output)
    if stage == 'shared':
        return [dict(label=j['id'], stage=stage, job=j, seconds=900) for j in shared_plan(manifest)]
    if stage == 'search':
        jobs = []
        for spec in search_plan(manifest):
            records = [load_task(output / 'shared' / key) for key in spec['shared_ids']]
            from main_workflow import shared_rows
            shared_rows(manifest, records, spec['repeat'])
            if provenance is not None and any(r['spec']['provenance'] != provenance for r in records):
                raise ValueError('Shared source freeze/approval differs')
            remaining = 7200 - sum(r['seconds'] for r in records)
            # Allocated wall includes proposals, trainer imports/cleanup and idle.
            extra = manifest['proposal_seconds_per_llm_search_cap'] if spec['method'] not in ('random', 'tpe') else 0
            seconds = int((remaining + extra + 179) // 60) * 60
            jobs.append(dict(label=f"{spec['method']}_r{spec['repeat']}", stage=stage, job=spec, seconds=seconds))
        return jobs
    if stage != 'final': raise ValueError('Unknown stage')
    searches = [read(output / 'search' / f"{s['method']}_r{s['repeat']}" / 'summary.json') for s in search_plan(manifest)]
    if provenance is not None and any(s.get('provenance') != provenance for s in searches):
        raise ValueError('Search source freeze/approval differs')
    plan = final_plan(manifest, searches)
    return [dict(label=j['id'], stage=stage, job=j, seconds=900) for j in plan['physical_trainings']]


class ProcessBackend:
    def __init__(self, a, manifest): self.a = a; self.manifest = manifest

    def train(self, job, config, folder, cap):
        a = self.a; dest = folder / 'training'; path = folder / 'config.json'; atomic(path, config)
        outcome = run_bounded_process([str(a.root / 'envs/training-torch-2.12-cu130/bin/python'), '-B',
            str(a.root / 'test/training/train_flowbench_main.py'), '--root', str(a.root),
            '--manifest', str(a.manifest), '--freeze', str(a.freeze), '--authorization', str(a.authorization),
            '--config', str(path), '--data', str(a.root / self.manifest['dataset_relative']),
            '--normalization', str(a.root / self.manifest['normalization_source_relative']),
            '--output', str(dest), '--max-wall-seconds', str(max(1, cap - 10))], cap, folder / 'training.log')
        result = read(dest / 'summary.json') if (dest / 'summary.json').exists() else {}
        success = outcome['exit_code'] == 0 and result.get('status') == 'FLOWBENCH_MAIN_TRIAL_COMPLETE'
        report = dict(status='timeout' if outcome['timed_out'] else 'failed', seconds=outcome['wall_seconds'], outcome=outcome)
        if success:
            if (result['test_evaluated'] or result['config'] != config or result['split_sha256'] != self.manifest['split_sha256']
                or result['normalization_sha256'] != digest(a.root / self.manifest['normalization_source_relative'])
                or result['freeze_file_sha256'] != digest(a.freeze) or result['authorization_file_sha256'] != digest(a.authorization)
                or (self.manifest.get('required_gpu_name') and result.get('gpu') != self.manifest['required_gpu_name'])
                or result['checkpoint_sha256'] != digest(dest / 'best_model.pt')):
                raise ValueError('Training provenance differs; keep full reservation')
            report.update(status='complete', measurement=dict(result['best_validation'], divergence=result['predicted_divergence_rms_mean']),
                          summary_sha256=digest(dest / 'summary.json'), checkpoint_sha256=result['checkpoint_sha256'],
                          checkpoint_path=str(dest / 'best_model.pt'))
        atomic(folder / 'outcome.json', report); return report

    def propose(self, method, feedback, folder, cap):
        a = self.a; cfg_path = a.root / self.manifest['llm_config_relative']; cfg = read(cfg_path)
        index = next(i for i, m in enumerate(cfg['models']) if m['label'] == method)
        dest = folder / 'proposal'; path = folder / 'feedback.json'
        outcome = run_bounded_process([str(a.root / 'envs/llm-feasibility-20261021/bin/python'), '-B',
            str(a.root / 'test/llm/propose_configuration_transport_v1.py'), '--config', str(cfg_path),
            '--preparation', str(a.root / self.manifest['model_preparation_relative']), '--feedback', str(path),
            '--index', str(index), '--output', str(dest)], cap, folder / 'proposal.log')
        result = read(dest / 'summary.json') if (dest / 'summary.json').exists() else {}
        report = dict(status='decision_timeout' if outcome['timed_out'] else 'decision_failed',
                      seconds=outcome['wall_seconds'], outcome=outcome, feedback_sha256=digest(path))
        if outcome['exit_code'] == 0 and result.get('status') == 'LOCAL_CONFIGURATOR_V3_COMPLETE':
            if (result['system_prompt_sha256'] != SYSTEM_SHA256 or result['feedback_sha256'] != digest(path)
                or result['phase'] != 'main_search' or result['test_evaluated'] or result['model'] != cfg['models'][index]
                or result['transport_policy'] != POLICY or result['generation'] != cfg['generation']
                or result['contract_version'] != 3
                or result['transport_source_sha256'] != digest(a.root / 'test/llm/configurator_transport_v1.py')
                or result['initial_messages_sha256'] != hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest()):
                raise ValueError('Proposal provenance differs; keep reservation')
            # Recheck structured public explanation/evidence, not hidden reasoning.
            parsed, errors = parse_reply(canonical(result['configuration']),
                                        [r['id'] for r in feedback['history'] if r['status'] == 'complete'])
            if errors: raise ValueError('Successful proposal violates contract')
            report.update(status='complete', parameters={k: parsed[k] for k in PARAMETERS},
                          configuration_with_reason=parsed, summary_sha256=digest(dest / 'summary.json'))
        elif result.get('status') == 'LOCAL_CONFIGURATOR_V3_INVALID': report['status'] = 'invalid_reply'
        atomic(folder / 'decision_outcome.json', report); return report


def worker(a):
    manifest, freeze, approval = require_authorization(a.manifest, a.freeze, a.authorization)
    verify_runtime_freeze(a.root, manifest, freeze)
    provenance = dict(freeze_sha256=digest(a.freeze), authorization_sha256=digest(a.authorization))
    jobs = stage_jobs(manifest, a.stage, a.output, provenance)
    job = next(j for j in jobs if j['label'] == a.label)
    # A direct invocation cannot escape the physically reserved Slurm allocation.
    deadline = time.monotonic() + 15
    while True:
        try:
            with PhysicalBudget(a.output / 'physical_budget', **budget_config(a.manifest, a.freeze)) as ledger:
                found = [r for r in ledger.state['allocations'] if r['label'] == a.stage + ':' + a.label]
                if (len(found) != 1 or found[0]['job_id'] != os.environ.get('SLURM_JOB_ID')
                    or found[0]['wall_limit_seconds'] != job['seconds'] or found[0]['gpus'] != 1
                    or found[0]['status'] != 'submitted_on_hold'): raise ValueError('Own live physical reservation required')
            break
        except BlockingIOError:
            if time.monotonic() >= deadline: raise
            time.sleep(.1)
    backend = ProcessBackend(a, manifest); folder = a.output / a.stage / a.label
    if a.stage == 'search':
        records = [load_task(a.output / 'shared' / key) for key in job['job']['shared_ids']]
        return run_search(manifest, job['job']['method'], job['job']['repeat'], records, folder, backend, provenance)
    return fixed_task(manifest, job['job'], folder, backend, provenance)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'manifest', 'freeze', 'authorization', 'output'): p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--stage', choices=['shared', 'search', 'final'], required=True); p.add_argument('--label', required=True)
    worker(p.parse_args())

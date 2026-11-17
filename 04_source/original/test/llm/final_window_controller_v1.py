"""Versioned operational extension around unchanged authorized main workers.

Scientific source freeze, selected parameters, seeds and test gate stay intact.
CPU afterany chains stop on incomplete training/accounting. Never opens test.
"""
import argparse
import fcntl
import math
from collections import defaultdict
from pathlib import Path
import re
import subprocess

from main_protocol import read, digest, identity, require_authorization, search_plan, final_plan
from main_execution import budget_config, stage_jobs, load_task, verify_runtime_freeze
from main_workflow import atomic
from allocation_budget import PhysicalBudget
from final_window_engine_v1 import advance, release_window


def command(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def job_id(value):
    if not re.fullmatch(r'[0-9]+', value): raise ValueError('Numeric non-array job ID required')
    return value


def locked_plan(a, manifest, provenance):
    searches = [read(a.output / 'search' / f"{s['method']}_r{s['repeat']}" / 'summary.json') for s in search_plan(manifest)]
    if any(s['provenance'] != provenance for s in searches): raise ValueError('Search freeze/approval differs')
    plan = final_plan(manifest, searches); jobs = stage_jobs(manifest, 'final', a.output, provenance)
    paths = [a.output / 'search' / f"{s['method']}_r{s['repeat']}" / 'summary.json' for s in search_plan(manifest)]
    return plan, jobs, {str(p): digest(p) for p in paths}


class SlurmBackend:
    def __init__(self, a, manifest, provenance): self.a = a; self.manifest = manifest; self.provenance = provenance

    def queue_count(self):
        rows = command(['squeue', '-h', '-r', '--states=all', '-u', 'lutoke823', '-o', '%i %T']).splitlines()
        for row in rows:
            if len(row.split()) != 2: raise ValueError('Unparseable queue')
        return len(rows)  # Conservatively includes terminal-looking teardown entries.

    def submit_gpu(self, job):
        a = self.a
        return job_id(command(['sbatch', '--hold', '--parsable', '--no-requeue', '--export=NIL',
            '--account=perun2601449', '--qos=perun2601449', '--partition=gpu_short', '--nodes=1', '--ntasks=1',
            '--gres=gpu:1', '--cpus-per-task=4', '--mem=64G', '--time=15', '--job-name=dp-main-final',
            '--output=' + str(a.output / '%x_%j.out'), '--error=' + str(a.output / '%x_%j.err'),
            str(a.root / 'test/scripts/117_main_worker.sbatch'), str(a.root), str(a.manifest), str(a.freeze),
            str(a.authorization), str(a.output), 'final', job['label']]))

    def submit_capture(self, job, gpu):
        return job_id(command(['sbatch', '--parsable', '--dependency=afterany:' + gpu,
            str(self.a.root / 'test/scripts/122_capture_main_allocation.sbatch'), 'final:' + job['label'], gpu]))

    def submit_followup(self, dependencies):
        return job_id(command(['sbatch', '--parsable', '--dependency=afterany:' + ':'.join(dependencies),
            str(self.a.root / 'test/scripts/123_final_window_controller.sbatch'), str(self.a.extension_review)]))

    def release(self, job): command(['scontrol', 'release', job])

    def review(self, job):
        a = self.a; folder = a.output / 'final' / job['label']; task = load_task(folder)
        summary = read(folder / 'training/summary.json'); config = read(folder / 'config.json')
        from main_protocol import training_config
        if (task['status'] != 'complete' or task['spec']['job'] != job['job'] or task['spec']['provenance'] != self.provenance
            or config != training_config(self.manifest, job['job']) or summary['config'] != config
            or summary['status'] != 'FLOWBENCH_MAIN_TRIAL_COMPLETE' or summary['pilot_only'] or summary['test_evaluated']
            or [h['epoch'] for h in summary['history']] != list(range(1, 21))
            or summary['best_epoch'] != min(summary['history'], key=lambda h: (h['validation']['S'], h['epoch']))['epoch']
            or digest(folder / 'training/summary.json') != task['result']['summary_sha256']
            or digest(folder / 'training/best_model.pt') != summary['checkpoint_sha256']
            or summary['checkpoint_sha256'] != task['result']['checkpoint_sha256']):
            raise ValueError('Incomplete/mismatched final model; stop entire balanced phase')
        split = read(a.root / self.manifest['dataset_relative'] / 'split.json')
        vals = read(folder / 'training/best_validation_rows.json'); groups = defaultdict(list)
        if (summary['training_indices'] != sorted(split['indices']['train'])
            or summary['validation_indices'] != sorted(split['indices']['validation'])
            or summary['gpu'] != self.manifest['required_gpu_name']
            or summary['normalization_sha256'] != digest(a.root / self.manifest['normalization_source_relative'])
            or len(vals) != 150 or {v['sample'] for v in vals} != set(split['indices']['validation'])):
            raise ValueError('Final data/normalization identity differs')
        for row in vals:
            if row['geometry'] != split['geometry_groups'][row['sample']]: raise ValueError('Geometry mapping differs')
            if any(not math.isfinite(row[k]) or row[k] < 0 for k in ('velocity_error', 'pressure_error')):
                raise ValueError('Nonfinite final validation')
            groups[row['geometry']].append(row)
        if len(groups) != 15 or any(len(v) != 10 for v in groups.values()): raise ValueError('Geometry15x10 required')
        mean = lambda values: math.fsum(values) / len(values)
        eu = mean([mean([r['velocity_error'] for r in g]) for g in groups.values()])
        ep = mean([mean([r['pressure_error'] for r in g]) for g in groups.values()])
        metrics = dict(S=.5 * (eu + ep), velocity_error=eu, pressure_error=ep)
        if any(abs(metrics[k] - summary['best_validation'][k]) > 1e-12 for k in metrics):
            raise ValueError('Independent final validation aggregation differs')
        atomic(a.output / 'final_windows_v1/model_reviews' / (job['label'] + '.json'),
               dict(status='FINAL_MODEL_TRAIN_VALIDATION_QC_PASSED', seed=config['seed'],
                    metrics=metrics, checkpoint_sha256=summary['checkpoint_sha256'], test_evaluated=False))
        return task


def run(a):
    manifest, freeze, approval = require_authorization(a.manifest, a.freeze, a.authorization)
    verify_runtime_freeze(a.root, manifest, freeze)
    extension = read(a.extension_review)
    if extension['status'] != 'FINAL_WINDOW_EXTENSION_CHECKS_PASSED' or extension['original_freeze_sha256'] != digest(a.freeze):
        raise ValueError('Matching tested operational extension required')
    for source, expected in extension['source_files'].items():
        if digest(source) != expected: raise ValueError('Window extension changed: ' + source)
    provenance = dict(freeze_sha256=digest(a.freeze), authorization_sha256=digest(a.authorization))
    plan, jobs, inputs = locked_plan(a, manifest, provenance)
    if inputs != extension['search_summary_files'] or len(jobs) != extension['budget']['final_unique_trainings']:
        raise ValueError('Selected inputs differ from the reviewed final plan')
    folder = a.output / 'final_windows_v1'; folder.mkdir(exist_ok=True)
    with (folder / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(extension_review_sha256=digest(a.extension_review), provenance=provenance,
                    final_plan_sha256=identity(plan), search_summary_files=inputs)
        path = folder / 'state.json'
        if path.exists():
            envelope = read(path); state = envelope['payload']
            if identity(state) != envelope['sha256'] or state['spec'] != spec: raise ValueError('Window identity/checksum differs')
        else:
            if any((a.output / 'final' / j['label']).exists() for j in jobs): raise RuntimeError('Existing final output; no silent restart')
            state = dict(spec=spec, phase='new', next_index=0, windows=[])
            atomic(a.output / 'final_plan.json', plan); atomic(a.output / 'final_stage_plan.json', jobs)
        def save(): atomic(path, dict(payload=state, sha256=identity(state)))
        save(); backend = SlurmBackend(a, manifest, provenance)
        with PhysicalBudget(a.output / 'physical_budget', **budget_config(a.manifest, a.freeze)) as ledger:
            if state['phase'] == 'new': atomic(folder / 'allocations_before_final.json', read(a.output / 'physical_budget/allocations.json'))
            ids = advance(state, jobs, ledger, backend, save)
            atomic(folder / 'budget_report.json', ledger.report())
        # Workers need the physical lock. Release only after that lock closes.
        require_authorization(a.manifest, a.freeze, a.authorization)
        release_window(state, ids, backend, save)
        if state['phase'] == 'ready_to_seal':
            from main_submit import main as original_submit
            args = argparse.Namespace(**vars(a)); args.action = 'seal'; args.stage = 'final'
            original_submit(args)
            state.update(phase='complete', final_catalog_sha256=digest(a.output / 'final_catalog.json')); save()
        atomic(folder / 'summary.json', dict(status=state['phase'], completed_or_dispatched=state['next_index'],
            total_trainings=len(jobs), reference_count=95, windows=len(state['windows']), test_evaluated=False,
            original_source_freeze_sha256=digest(a.freeze), operational_extension_sha256=digest(a.extension_review)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'manifest', 'freeze', 'authorization', 'output', 'extension-review'):
        p.add_argument('--' + key, type=Path, required=True)
    run(p.parse_args())

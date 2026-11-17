"""Separate heldout authorization. Standard library only until the gate passes."""
import argparse
import os
from pathlib import Path
from main_protocol import read, digest, identity, require_authorization, final_plan, search_plan
from allocation_budget import PhysicalBudget

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
MAIN = ROOT / 'results/llm_main/main_20261021_release124502'
RELEASE = ROOT / 'results/llm_search_prepare/main_release_124502'
MANIFEST = ROOT / 'test/config/flowbench_main_release_20261021.json'
DEST = MAIN / 'final_measurements_v3'
CATALOG_SHA = '0ea1956f69ff0d3e4fc238c92dbec67cd9a3f52ac7d2d076e47101c3691963c5'
SOURCES = ['llm/final_measurement_gate_v3.py', 'llm/final_measurement_submit_v3.py',
           'llm/test_final_measurement_v3.py', 'training/measure_final_models_v4.py',
           'scripts/127_final_measurement.sbatch', 'finalne_meranie_protokol_v3.txt',
           'training/audit_protocol_latency.py', 'training/validate_final_adapter_v4.py',
           'training/test_final_numerical_metrics_v4.py', 'llm/submit_final_validation_preflight_v3.py']
POLICY = dict(version=1, physical_models=70, logical_references=95, test_samples=150,
              test_geometries=15, reynolds_per_geometry=10, wall_seconds_per_stage=7200,
              modes=['model_only', 'gpu_pipeline', 'host_to_host'], rounds=3,
              warmup=100, measurements=1000, latency_validation_geometries=15,
              primary='gpu_pipeline synchronized wall p99 <= 5ms in every round',
              target_ms=5, percentile_method='linear', automatic_retries=False,
              validation_preflight_seconds=900)


class TestOnly:
    def __init__(self, array, indices):
        self.array = array; self.allowed = frozenset(indices)
    def __getitem__(self, index):
        if type(index) is not int or index not in self.allowed:
            raise PermissionError('Only the locked test IDs are allowed')
        return self.array[index]


def verify_files(files):
    if not files: raise ValueError('Empty source lock')
    for path, sha in files.items():
        if digest(path) != sha: raise ValueError('Changed locked file: ' + path)


def exclusive_claim(path, payload):
    # A failed/partial run leaves this durable claim: no automatic test replay.
    with Path(path).open('x') as f:
        from main_protocol import canonical
        f.write(canonical(payload) + '\n'); f.flush(); os.fsync(f.fileno())


def locked_models(manifest):
    if digest(MAIN / 'final_catalog.json') != CATALOG_SHA: raise ValueError('Catalog changed')
    catalog = read(MAIN / 'final_catalog.json')
    if catalog['status'] != 'MAIN_FINAL_TRAINING_COMPLETE_NOT_TESTED': raise ValueError('Unsealed final catalog')
    searches = [read(MAIN / 'search' / f"{s['method']}_r{s['repeat']}" / 'summary.json') for s in search_plan(manifest)]
    plan = final_plan(manifest, searches)
    if plan != read(MAIN / 'final_plan.json') or plan['references'] != catalog['references']:
        raise ValueError('Final selection/aliases changed')
    if len(plan['physical_trainings']) != 70 or len(catalog['references']) != 95:
        raise ValueError('Incomplete balanced final phase')
    models = []
    for job in plan['physical_trainings']:
        entry = catalog['trainings'][job['id']]; folder = MAIN / 'final' / job['id']
        task = load_task(folder); result = entry['result']
        if (identity(task) != entry['task_sha256'] or task['spec']['job'] != job
                or task['status'] != 'complete' or task['result'] != result):
            raise ValueError('Task/catalog identity differs')
        checkpoint = Path(result['checkpoint_path']); summary = checkpoint.with_name('summary.json')
        if checkpoint != folder / 'training/best_model.pt': raise ValueError('Checkpoint outside final job')
        if digest(checkpoint) != result['checkpoint_sha256'] or digest(summary) != result['summary_sha256']:
            raise ValueError('Final model/summary changed')
        s = read(summary); cfg = s['config']
        if (s['status'] != 'FLOWBENCH_MAIN_TRIAL_COMPLETE' or s['test_evaluated']
                or cfg['seed'] != job['seed'] or cfg['stage'] != 'final'
                or not cfg['fresh_initialization'] or len(s['history']) != 20
                or any(cfg['training'][k] != v for k, v in job['parameters'].items())
                or any(cfg['training'][k] != v for k, v in manifest['training'].items())
                or s['normalization_sha256'] != digest(ROOT / manifest['normalization_source_relative'])
                or s['split_sha256'] != manifest['split_sha256']):
            raise ValueError('Final recipe differs')
        models.append(dict(id=job['id'], seed=job['seed'], parameters=job['parameters'],
            checkpoint=str(checkpoint), checkpoint_sha256=result['checkpoint_sha256'],
            summary=str(summary), summary_sha256=result['summary_sha256']))
    return catalog, models


def original_gate():
    return require_authorization(MANIFEST, RELEASE / 'freeze.json', RELEASE / 'authorization.json')[0]


def prepare():
    manifest = original_gate(); catalog, models = locked_models(manifest)
    recovery = MAIN / 'final_measurements_v2/interrupted_test_review.json'
    review = read(recovery)
    if (review['status'] != 'INTERRUPTED_AFTER_FIRST_TEST_TARGET_NO_AUTOMATIC_REPLAY'
            or review['opened_test_ids'] != [10] or review['saved_test_scores']
            or review['completed_model_measurements'] != 0):
        raise ValueError('Prior interrupted attempt audit differs')
    with PhysicalBudget(MAIN / 'physical_budget', **budget_config(MANIFEST, RELEASE / 'freeze.json')) as ledger:
        if any(r['status'] != 'settled' for r in ledger.state['allocations']):
            raise RuntimeError('Earlier allocations must all be closed')
        budget = ledger.report()
        if budget['available_gpu_seconds'] < 2 * (POLICY['wall_seconds_per_stage'] + 1) + 901:
            raise RuntimeError('Both final measurement stages must fit original70h cap')
        snapshot = dict(charged_gpu_seconds=budget['charged_or_reserved_gpu_seconds'],
                        available_gpu_seconds=budget['available_gpu_seconds'])
    split_path = ROOT / manifest['dataset_relative'] / 'split.json'
    split = read(split_path)
    if digest(split_path) != manifest['split_sha256']: raise ValueError('Changed split')
    test_ids = sorted(split['indices']['test'])
    groups = {split['geometry_groups'][i] for i in test_ids}
    if len(test_ids) != 150 or len(groups) != 15: raise ValueError('Unexpected test split')
    if any(sum(split['geometry_groups'][i] == g for i in test_ids) != 10 for g in groups):
        raise ValueError('Unbalanced test geometries')
    selected = []; seen = set()
    for i in sorted(split['indices']['validation']):
        g = split['geometry_groups'][i]
        if g not in seen: selected.append(i); seen.add(g)
    if len(selected) != 15: raise ValueError('Unbalanced latency input set')
    sources = {str(ROOT / 'test' / s): digest(ROOT / 'test' / s) for s in SOURCES}
    payload = dict(policy=POLICY, sources=sources, catalog_sha256=CATALOG_SHA,
        original_manifest_sha256=digest(MANIFEST), original_freeze_sha256=digest(RELEASE / 'freeze.json'),
        original_authorization_sha256=digest(RELEASE / 'authorization.json'), models=models,
        references=catalog['references'], split_sha256=manifest['split_sha256'], test_ids=test_ids,
        latency_input_ids=selected, normalization_sha256=digest(ROOT / manifest['normalization_source_relative']),
        budget_before=snapshot, recovery_review_path=str(recovery), recovery_review_sha256=digest(recovery),
        prior_accuracy_claim_sha256=digest(MAIN / 'final_measurements_v2/accuracy.claim.json'),
        consent=dict(date='2026-10-22', user_text='ano', technical_replay_approved=True,
        scope='User explicitly approved documented replay after diagnostic-mask interruption; validation-only preflight, one complete heldout evaluation and separate latency. All70models and scientific settings unchanged.',
        test_execution_approved=True, latency_execution_approved=True))
    DEST.mkdir(parents=True, exist_ok=False)
    atomic(DEST / 'authorization.json', dict(payload=payload, sha256=identity(payload)))
    # Keep exact measured source copies as well as hashes.
    import shutil
    for path in sources:
        target = DEST / 'sources' / Path(path).relative_to(ROOT / 'test')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    print('FINAL_MEASUREMENT_AUTHORIZED', identity(payload), snapshot, flush=True)


def authorized():
    manifest = original_gate()
    envelope = read(DEST / 'authorization.json'); lock = envelope['payload']
    if identity(lock) != envelope['sha256'] or lock['policy'] != POLICY:
        raise ValueError('Measurement authorization checksum/policy differs')
    if (not lock['consent']['test_execution_approved'] or not lock['consent']['latency_execution_approved']
            or lock['consent']['user_text'] != 'ano' or lock['consent'].get('technical_replay_approved') is not True
            or lock['catalog_sha256'] != CATALOG_SHA
            or digest(MANIFEST) != lock['original_manifest_sha256']
            or digest(RELEASE / 'freeze.json') != lock['original_freeze_sha256']
            or digest(RELEASE / 'authorization.json') != lock['original_authorization_sha256']):
        raise PermissionError('Separate test approval required')
    required = {str(ROOT / 'test' / s) for s in SOURCES}
    if set(lock['sources']) != required: raise ValueError('Incomplete measurement source freeze')
    verify_files(lock['sources'])
    if (digest(lock['recovery_review_path']) != lock['recovery_review_sha256']
            or digest(MAIN / 'final_measurements_v2/accuracy.claim.json') != lock['prior_accuracy_claim_sha256']):
        raise ValueError('Prior interrupted run evidence changed')
    catalog, models = locked_models(manifest)
    if models != lock['models'] or catalog['references'] != lock['references']:
        raise ValueError('Locked models/aliases differ')
    data = ROOT / manifest['dataset_relative']; split = read(data / 'split.json')
    if (digest(data / 'split.json') != lock['split_sha256']
            or sorted(split['indices']['test']) != lock['test_ids']
            or digest(ROOT / manifest['normalization_source_relative']) != lock['normalization_sha256']):
        raise ValueError('Test/norm identity changed')
    return manifest, lock


def own_allocation(stage):
    with PhysicalBudget(MAIN / 'physical_budget', **budget_config(MANIFEST, RELEASE / 'freeze.json')) as ledger:
        rows = [r for r in ledger.state['allocations'] if r['label'] == 'measurement_v3:' + stage]
        if (len(rows) != 1 or rows[0]['job_id'] != os.environ.get('SLURM_JOB_ID')
                or rows[0]['status'] != 'submitted_on_hold' or rows[0]['gpus'] != 1
                or rows[0]['wall_limit_seconds'] != (900 if stage == 'validation_smoke' else POLICY['wall_seconds_per_stage'])
                or ledger.report()['overrun_gpu_seconds']):
            raise PermissionError('Own attached budgeted Slurm allocation required')




# No operational-training imports: the numerical venv intentionally lacks psutil.
# These standard-library helpers preserve the original ledger identity and task hash.
def atomic(path, value):
    import json
    import tempfile
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.measurement-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def budget_config(manifest_path, freeze_path):
    return dict(gpu_seconds_limit=252000, identity=dict(user='lutoke823', account='perun2601449',
        manifest_file_sha256=digest(manifest_path), freeze_file_sha256=digest(freeze_path), phase='main'))


def load_task(directory):
    envelope = read(Path(directory) / 'task.json')
    if identity(envelope['payload']) != envelope['sha256']: raise ValueError('Task checksum differs')
    return envelope['payload']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'verify'])
    if parser.parse_args().action == 'prepare': prepare()
    else: authorized(); print('MEASUREMENT_LOCK_VERIFIED_WITHOUT_TARGET_ACCESS')

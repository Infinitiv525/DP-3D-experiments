"""Declarative main-experiment planning and fail-closed execution gates.

Standard library only. Planning cannot launch jobs, load arrays or evaluate test.
Main training uses fresh initialization; only final physical replicas deduplicate.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import importlib.metadata
import platform
import sys

from search_controls import validate_parameters, RandomConfigurator

METHODS = ('random', 'tpe', 'qwen3_8b', 'ministral3_8b', 'phi4_mini', 'granite33_8b')
PARAMETERS = ('learning_rate', 'weight_decay', 'pressure_weight', 'divergence_weight')


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_manifest(manifest):
    if manifest['version'] != 1 or tuple(manifest['methods']) != METHODS: raise ValueError('Six predeclared methods required')
    if manifest['requires_separate_user_approval'] is not True: raise ValueError('Explicit approval gate mandatory')
    if manifest['training'] != dict(width=12, layers=2, modes=4, epochs=20, max_train_samples=700, max_validation_samples=150):
        raise ValueError('Fixed20epoch architecture/data required')
    if manifest['shared_attempts'] != 3 or manifest['attempts_per_search_including_shared'] != 12: raise ValueError('Fixed trial counts required')
    if manifest['search_repeat_seeds'] != [20261101, 20261102, 20261103] or manifest['final_training_seeds'] != list(range(410001, 410006)):
        raise ValueError('Predeclared disjoint search/final seeds required')
    if manifest['training_seed_base'] != 310000 or manifest['training_seed_repeat_stride'] != 100 or manifest['sampler_seed_offset'] != 1000000:
        raise ValueError('Seed policy differs')
    if manifest['global_physical_gpu_hours_cap'] != 70 or manifest['training_seconds_per_method_repeat'] != 7200:
        raise ValueError('Budget proposal differs')
    if manifest['transport_policy'] != 'single_complete_json_fence_only_v1' or manifest['prompt_contract'] != 3: raise ValueError('Common contract required')
    if not 0 < manifest['trial_process_seconds_cap'] < manifest['trial_allocation_seconds_cap'] <= 900:
        raise ValueError('Cleanup/startup margin required')
    validate_parameters(manifest['manual']); canonical(manifest)


def training_seed(manifest, repeat, attempt):
    if type(repeat) is not int or not 0 <= repeat < 3 or type(attempt) is not int or not 0 <= attempt < 12:
        raise ValueError('Invalid repeat/attempt')
    return manifest['training_seed_base'] + manifest['training_seed_repeat_stride'] * repeat + attempt


def shared_plan(manifest):
    validate_manifest(manifest); result = []
    for repeat, seed in enumerate(manifest['search_repeat_seeds']):
        rng = RandomConfigurator(seed)
        configurations = [manifest['manual'], rng.ask(), rng.ask()]
        for attempt, parameters in enumerate(configurations):
            result.append({'id': f'r{repeat}_s{attempt:02}', 'stage': 'shared', 'repeat': repeat, 'number': attempt,
                           'parameters': parameters, 'seed': training_seed(manifest, repeat, attempt), 'epochs': 20,
                           'fresh_initialization': True})
    return result


def search_plan(manifest):
    validate_manifest(manifest)
    return [{'method': method, 'repeat': repeat, 'search_seed': seed,
             'sampler_seed': seed + manifest['sampler_seed_offset'], 'logical_training_seconds': 7200,
             'max_attempts_including_shared': 12, 'shared_ids': [f'r{repeat}_s{j:02}' for j in range(3)]}
            for repeat, seed in enumerate(manifest['search_repeat_seeds']) for method in METHODS]


def canonical_parameters(parameters):
    validate_parameters(parameters)
    return {key: float(parameters[key]) for key in PARAMETERS}


def winner(manifest, search):
    validate_manifest(manifest)
    if search['phase'] != 'main_search' or search['epochs'] != 20 or search['test_evaluated']:
        raise ValueError('Never promote pilot/test results to main winners')
    if search['manifest_sha256'] != identity(manifest) or search['method'] not in METHODS or type(search['repeat']) is not int or not 0 <= search['repeat'] < 3:
        raise ValueError('Search provenance differs')
    trials = search['trials']
    if not 3 <= len(trials) <= 12 or [t['number'] for t in trials] != list(range(len(trials))): raise ValueError('Contiguous counted attempts including3shared required')
    spent = search.get('training_seconds_spent')
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0:
        raise ValueError('Actual search training cost required')
    reason = search.get('stop_reason')
    if search.get('status') != 'MAIN_SEARCH_COMPLETE': raise ValueError('Interrupted/unreviewed search cannot be finalized')
    if reason == 'attempt_cap':
        if len(trials) != 12: raise ValueError('Premature attempt-cap declaration')
    elif reason == 'training_time_cap':
        if spent < manifest['training_seconds_per_method_repeat'] - 5: raise ValueError('Training budget not exhausted')
    elif reason == 'decision_time_cap':
        seconds = search.get('decision_seconds_spent')
        if search['method'] in ('random', 'tpe') or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 1885:
            raise ValueError('LLM decision budget not exhausted')
    else: raise ValueError('Incomplete/global-aborted search cannot be silently finalized')
    expected_shared = [r for r in shared_plan(manifest) if r['repeat'] == search['repeat']]
    completed = []
    for trial in trials:
        if trial['seed'] != training_seed(manifest, search['repeat'], trial['number']): raise ValueError('Seed differs')
        if trial['number'] < 3 and (trial['parameters'] != expected_shared[trial['number']]['parameters'] or trial['status'] != 'complete'):
            raise ValueError('Main shared starts must complete and match fixed plan')
        if trial['status'] == 'complete':
            canonical_parameters(trial['parameters']); score = trial['S']
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or score < 0: raise ValueError('Invalid completed score')
            completed.append(trial)
        elif trial['status'] not in ('failed', 'timeout', 'oom', 'invalid_reply', 'decision_timeout', 'decision_failed', 'interrupted_reserved_cost') or trial['S'] is not None:
            raise ValueError('Failed attempts have no selectable score')
    return min(completed, key=lambda t: (t['S'], t['number']))


def final_plan(manifest, searches):
    validate_manifest(manifest)
    expected = {(m, r) for m in METHODS for r in range(3)}
    pairs = [(s['method'], s['repeat']) for s in searches]
    if len(pairs) != 18 or len(set(pairs)) != 18 or set(pairs) != expected: raise ValueError('All18searches required before final plan')
    selected = {(s['method'], s['repeat']): winner(manifest, s) for s in searches}
    references = []; unique = {}; manifest_hash = identity(manifest)
    # Interleave seed -> repeat -> method; manual once/seed, not3pseudo-repeats.
    for seed in manifest['final_training_seeds']:
        choices = [('manual', None, manifest['manual'], None)]
        choices += [(method, repeat, selected[(method, repeat)]['parameters'], selected[(method, repeat)]['number'])
                    for repeat in range(3) for method in METHODS]
        for method, repeat, parameters, source_attempt in choices:
            params = canonical_parameters(parameters)
            key = identity({'manifest_sha256': manifest_hash, 'parameters': params, 'seed': seed, 'epochs': 20})
            reference = {'method': method, 'repeat': repeat, 'seed': seed, 'source_attempt': source_attempt, 'training_id': key}
            references.append(reference)
            if key not in unique:
                unique[key] = {'id': key, 'stage': 'final', 'parameters': params, 'seed': seed, 'epochs': 20,
                               'fresh_initialization': True, 'initial_checkpoint': None, 'references': []}
            unique[key]['references'].append(reference)
    return {'phase': 'main_final_plan_not_test', 'manifest_sha256': manifest_hash, 'references': references,
            'physical_trainings': list(unique.values()), 'reference_count': len(references), 'physical_training_count': len(unique),
            'saved_duplicate_trainings': len(references) - len(unique),
            'all_final_allocations_must_be_reserved_before_start': True, 'test_evaluated': False}


def require_authorization(manifest_path, freeze_path, authorization_path):
    manifest = read(manifest_path); validate_manifest(manifest)
    if not manifest.get('execution_ready') or manifest.get('not_yet_released'):
        raise RuntimeError('Main workflow not yet released; candidate is not executable')
    freeze = read(freeze_path)
    if freeze.get('status') != 'MAIN_SOURCE_FREEZE_PASSED' or freeze.get('manifest_file_sha256') != digest(manifest_path):
        raise ValueError('Verified matching source freeze required')
    if authorization_path is None: raise PermissionError('No explicit approval file supplied')
    authorization = read(authorization_path)
    if (authorization.get('approved') is not True or authorization.get('manifest_file_sha256') != digest(manifest_path)
        or authorization.get('freeze_file_sha256') != digest(freeze_path) or authorization.get('gpu_hours_cap') != 70
        or not isinstance(authorization.get('user_consent_reference'), str) or not authorization['user_consent_reference'].strip()):
        raise PermissionError('Approval must explicitly identify this70GPU-h experiment and freeze')
    for absolute, expected in freeze['source_files'].items():
        if digest(absolute) != expected: raise ValueError('Frozen source changed: ' + absolute)
    if manifest.get('asset_integrity_policy'):
        receipt_path = freeze['asset_receipt_path']
        if str(receipt_path) not in freeze['source_files']: raise ValueError('Asset receipt must itself be frozen')
        receipt = read(receipt_path)
        if receipt['status'] != 'ALL_MODEL_AND_CACHE_FILES_REHASHED': raise ValueError('Full asset verification required')
        for row in receipt['files']:
            stat = Path(row['path']).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (row['bytes'], row['mtime_ns'], row['inode']):
                raise ValueError('Verified asset metadata changed; rehash/review required: ' + row['path'])
        environment = freeze['environments'][sys.prefix]
        if environment['python'] != platform.python_version(): raise ValueError('Frozen interpreter changed')
        for package, expected in environment['packages'].items():
            try: current = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: current = None
            if current != expected: raise ValueError('Frozen package changed: ' + package)
    return manifest, freeze, authorization


def training_config(manifest, job):
    validate_manifest(manifest); validate_parameters(job['parameters'])
    if job['epochs'] != 20 or not job.get('fresh_initialization') or job.get('initial_checkpoint'):
        raise ValueError('Only fresh20epoch main training permitted')
    if job['stage'] not in ('shared', 'search', 'final'): raise ValueError('Training cannot perform test')
    if type(job['seed']) is not int: raise ValueError('Integer seed required')
    allowed_seeds = (manifest['final_training_seeds'] if job['stage'] == 'final' else
        [training_seed(manifest, r, j) for r in range(3) for j in range(3 if job['stage'] == 'shared' else 12)])
    if job['seed'] not in allowed_seeds: raise ValueError('Seed outside predeclared stage policy')
    return {'revision': manifest['revision'], 'seed': job['seed'], 'pilot_only': False, 'phase': 'main_training',
            'stage': job['stage'], 'manifest_sha256': identity(manifest), 'fresh_initialization': True,
            'purpose': 'Approved main train/validation-only trial; no test evaluation.',
            'training': dict(manifest['training'], **job['parameters'])}

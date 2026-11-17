"""CPU release review: stream-hash assets, pin source/inputs, record human consent.

Never submits a GPU job or computes target statistics. A failed check produces
no executable freeze or approval. Old candidates/pilots remain unchanged.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

from main_protocol import read, digest, validate_manifest, shared_plan, search_plan, require_authorization
from main_execution import runtime_paths, verify_runtime_freeze
from main_workflow import atomic
from configurator_contract_v3 import SYSTEM_SHA256
from configurator_transport_v1 import POLICY
import verify_main_workflow

PACKAGES = ['numpy', 'torch', 'optuna', 'psutil', 'transformers', 'safetensors', 'tokenizers',
            'mistral-common', 'accelerate', 'huggingface-hub']
CONSENT = ('User message in this dissertation chat,2026-10-21: "okej nech sa paci", '
           'in direct reply to request to finalize protocol/source freeze and approve a maximum70GPU-hours. '
           'Scope: main train/validation stages; heldout test execution remains separately gated.')


def checked_asset(path, expected_hash, expected_bytes=None):
    before = path.stat(); h = hashlib.sha256()
    if expected_bytes is not None and before.st_size != expected_bytes: raise ValueError('Asset size differs: ' + str(path))
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('Asset changed during rehash')
    if h.hexdigest() != expected_hash: raise ValueError('Asset SHA256 differs: ' + str(path))
    return dict(path=str(path), bytes=after.st_size, mtime_ns=after.st_mtime_ns, inode=after.st_ino, sha256=h.hexdigest())


def environment(python):
    code = ('import importlib.metadata as m,json,platform,sys\nout={}\nfor p in ' + repr(PACKAGES) +
            ':\n try: out[p]=m.version(p)\n except m.PackageNotFoundError: out[p]=None\n' +
            'print(json.dumps(dict(prefix=sys.prefix,python=platform.python_version(),packages=out)))')
    import json
    return json.loads(subprocess.run([str(python), '-B', '-c', code], check=True, capture_output=True, text=True, timeout=60).stdout)


def run(a):
    started = time.perf_counter(); manifest = read(a.manifest); validate_manifest(manifest)
    if (not manifest['execution_ready'] or manifest['not_yet_released'] or manifest['test_execution_released']
        or manifest['required_gpu_name'] != 'NVIDIA H200' or 'Bonferroni' not in manifest['statistics']['multiplicity']
        or manifest['shared_release_policy'] != 'release_r0_s00_first_then_verify_before_releasing_remaining8'):
        raise ValueError('Fixed train/validation-only release policy required')
    a.output.mkdir(parents=True, exist_ok=False)
    if not verify_main_workflow.run(a.output / 'cpu_checks'): raise RuntimeError('CPU regression failed')
    sources = runtime_paths(a.root, manifest)
    sources.update(str(a.root / 'test/llm' / p) for p in ('main_statistics.py', 'test_main_statistics.py',
                   'verify_main_workflow.py', 'test_main_protocol.py', 'test_main_workflow.py', 'test_main_release.py', 'freeze_main_release.py'))
    sources.update([str(a.root / 'test/scripts/119_freeze_main_release.sbatch'), str(a.root / 'test/protokol.txt')])
    for relative, status in [('results/flowbench/data_audit_123590/summary.json', 'FLOWBENCH_DATA_AUDIT_PASSED'),
            ('results/llm_search_prepare/transport_review_123872/summary.json', 'FOUR_FAMILY_TRANSPORT_CONFIRMATION_QC_PASSED')]:
        path = a.root / relative; record = read(path)
        if record['status'] != status or not all(record['checks'].values()): raise ValueError('Required QC failed')
        sources.add(str(path))
    data = a.root / manifest['dataset_relative']; audit = read(data / 'pressure_gauge_audit.json')
    if (audit['status'] != 'FLOWBENCH_NATIVE_GAUGE64_CACHE_COMPLETE' or audit['source_revision'] != manifest['revision']
        or digest(data / 'split.json') != manifest['split_sha256']): raise ValueError('Audited dataset identity differs')
    normalization = read(a.root / manifest['normalization_source_relative'])
    if normalization['identity']['split_sha256'] != manifest['split_sha256'] or normalization['identity']['audit_sha256'] != digest(data / 'pressure_gauge_audit.json'):
        raise ValueError('Train-only normalization differs')
    assets = []
    for channel in ('X', 'Y'):
        print('STREAM_VERIFY_CACHE', channel, flush=True)
        assets.append(checked_asset(data / (channel + '_pilot.npy'), audit['channels'][channel]['sha256']))
    config = read(a.root / manifest['llm_config_relative'])
    if config['generation'] != dict(seed=20261021, max_new_tokens=512, max_time_seconds=90, max_input_tokens=8192, do_sample=False, max_repairs=1):
        raise ValueError('Common generation differs')
    for spec in config['models']:
        proof = read(a.root / manifest['model_preparation_relative'] / (spec['label'] + '.json'))
        if proof['spec'] != spec or proof['status'] != 'LOCAL_MODEL_VERIFIED': raise ValueError('Pinned model differs')
        model_root = Path(proof['path'])
        if not model_root.resolve().is_relative_to((a.root / 'public_models').resolve()): raise ValueError('Model outside experiment storage')
        print('STREAM_VERIFY_MODEL', spec['label'], flush=True)
        for item in proof['files']:
            path = model_root / item['file']
            if not path.resolve().is_relative_to(model_root.resolve()): raise ValueError('Asset outside pinned model')
            assets.append(checked_asset(path, item['sha256'], item['bytes']))
    receipt_path = a.output / 'asset_receipt.json'
    atomic(receipt_path, dict(status='ALL_MODEL_AND_CACHE_FILES_REHASHED', files=assets,
        bytes_hashed=sum(r['bytes'] for r in assets), test_target_bytes_hashed=True, test_target_values_analyzed=False))
    sources.add(str(receipt_path))
    envs = [environment(a.root / 'envs' / name / 'bin/python') for name in ('llm-feasibility-20261021', 'training-torch-2.12-cu130')]
    freeze = dict(status='MAIN_SOURCE_FREEZE_PASSED', manifest_file_sha256=digest(a.manifest),
        source_files={p: digest(p) for p in sorted(sources)}, asset_receipt_path=str(receipt_path),
        environments={e['prefix']: {k: v for k, v in e.items() if k != 'prefix'} for e in envs},
        system_prompt_sha256=SYSTEM_SHA256, transport_policy=POLICY, cpu_checks_sha256=digest(a.output / 'cpu_checks/summary.json'),
        scope='train/validation only; first shared run is real GPU integration canary; test not released')
    tested_sources = read(a.output / 'cpu_checks/summary.json')['source_files']
    for path, expected in tested_sources.items():
        if digest(path) != expected: raise RuntimeError('Source changed after CPU tests: ' + path)
    verify_runtime_freeze(a.root, manifest, freeze)
    freeze_path = a.output / 'freeze.json'; atomic(freeze_path, freeze)
    authorization_path = a.output / 'authorization.json'
    atomic(authorization_path, dict(approved=True, gpu_hours_cap=70, manifest_file_sha256=digest(a.manifest),
        freeze_file_sha256=digest(freeze_path), user_consent_reference=CONSENT, test_execution_approved=False))
    require_authorization(a.manifest, freeze_path, authorization_path)
    atomic(a.output / 'shared_plan.json', shared_plan(manifest)); atomic(a.output / 'search_plan.json', search_plan(manifest))
    shutil.copyfile(a.manifest, a.output / 'manifest.json')
    snapshots = a.output / 'sources'; snapshots.mkdir()
    for p in sources:
        path = Path(p)
        if path.is_relative_to(a.root / 'test'):
            dest = snapshots / path.relative_to(a.root / 'test'); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    atomic(a.output / 'summary.json', dict(status='MAIN_TRAIN_VALIDATION_RELEASE_PASSED',
        cpu_checks=read(a.output / 'cpu_checks/summary.json')['tests'], files_rehashed=len(assets), bytes_rehashed=sum(r['bytes'] for r in assets),
        freeze_sha256=digest(freeze_path), authorization_sha256=digest(authorization_path), manifest_sha256=digest(a.manifest),
        gpu_hours_cap=70, gpu_jobs_submitted=0, test_execution_released=False, test_target_values_analyzed=False,
        elapsed_seconds=time.perf_counter() - started, user_consent_reference=CONSENT))
    print('MAIN_TRAIN_VALIDATION_RELEASE_PASSED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'manifest', 'output'): p.add_argument('--' + key, type=Path, required=True)
    run(p.parse_args())

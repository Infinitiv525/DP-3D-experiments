"""Authorization-gated adapter; identical numerical backend, fresh main trial.

The legacy numerical trainer is immutable. Its original raw report is retained
as backend_summary.json; a new report records the approved main provenance.
No Torch import, array read or GPU initialization before approval checks.
"""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm'))
from main_protocol import require_authorization, identity, digest, read, save, training_config


def gate(a):
    manifest, freeze, approval = require_authorization(a.manifest, a.freeze, a.authorization)
    config = read(a.config)
    if config.get('phase') != 'main_training' or config.get('manifest_sha256') != identity(manifest) or config.get('pilot_only') is not False:
        raise ValueError('Matching fresh main-trial configuration required')
    if config.get('initial_checkpoint') or not config.get('fresh_initialization'): raise ValueError('No warm starting')
    if config['training'] != dict(manifest['training'], **{k: config['training'][k] for k in ('learning_rate', 'weight_decay', 'pressure_weight', 'divergence_weight')}):
        raise ValueError('Fixed main training settings required')
    # Rebuild through the pure validator rather than trusting caller-provided fields.
    expected = training_config(manifest, dict(stage=config['stage'], seed=config['seed'], epochs=20,
        parameters={k: config['training'][k] for k in ('learning_rate', 'weight_decay', 'pressure_weight', 'divergence_weight')}, fresh_initialization=True))
    if config != expected: raise ValueError('Unexpected config fields or provenance')
    if a.data.resolve() != (a.root / manifest['dataset_relative']).resolve(): raise ValueError('Dataset path differs')
    if not math.isfinite(a.max_wall_seconds) or not 0 < a.max_wall_seconds <= manifest['trial_process_seconds_cap'] - 10:
        raise ValueError('Finite positive time and interpreter/cleanup margin required')
    if not a.normalization.exists(): raise ValueError('Verified shared train-only normalization must exist before trials')
    normalization = read(a.normalization)
    if normalization['identity']['split_sha256'] != manifest['split_sha256'] or normalization['identity']['audit_sha256'] != digest(a.data / 'pressure_gauge_audit.json'):
        raise ValueError('Normalization provenance differs')
    return manifest


def run(a):
    manifest = gate(a)
    import train_flowbench_protocol as numerical_backend
    if manifest.get('required_gpu_name') and numerical_backend.torch.cuda.get_device_name() != manifest['required_gpu_name']:
        raise RuntimeError('Predeclared GPU hardware required; no heterogeneous training benchmark')
    success = numerical_backend.run(a)
    raw_path = a.output / 'summary.json'; raw = read(raw_path)
    raw_path.rename(a.output / 'backend_summary.json')
    report = dict(raw)
    report.update(status='FLOWBENCH_MAIN_TRIAL_COMPLETE' if success else 'FLOWBENCH_MAIN_TRIAL_FAILED',
        pilot_only=False, phase='main_training', stage=read(a.config)['stage'], manifest_file_sha256=digest(a.manifest),
        freeze_file_sha256=digest(a.freeze), authorization_file_sha256=digest(a.authorization),
        backend_summary_sha256=digest(a.output / 'backend_summary.json'),
        backend_status=raw['status'], test_evaluated=False,
        provenance_note='Fresh authorized20epoch main run using immutable numerical backend; legacy backend report preserved unchanged.')
    save(raw_path, report)
    return success


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True); p.add_argument('--authorization', type=Path)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--normalization', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-wall-seconds', type=float, default=800)
    args = p.parse_args()
    try: result = run(args)
    except (ValueError, PermissionError, RuntimeError) as error:
        print('MAIN_GATE_DENIED_OR_TRIAL_FAILED:', str(error), flush=True); raise SystemExit(2)
    raise SystemExit(0 if result else 1)

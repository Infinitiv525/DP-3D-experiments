"""Independent CPU QC of the first counted main shared run; no test targets.

Recompute validation aggregation from saved rows, inspect actual checkpoint,
and fail closed. Does not rank models or introduce a new accuracy threshold.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys

from main_protocol import read, digest, identity, require_authorization, shared_plan, training_config


def report_checks(manifest, task, config, summary, rows, split):
    job = shared_plan(manifest)[0]; expected = training_config(manifest, job)
    groups = defaultdict(list)
    for row in rows: groups[row['geometry']].append(row)
    mean = lambda values: math.fsum(values) / len(values)
    eu = mean([mean([r['velocity_error'] for r in group]) for group in groups.values()])
    ep = mean([mean([r['pressure_error'] for r in group]) for group in groups.values()])
    recalculated = dict(S=.5 * (eu + ep), velocity_error=eu, pressure_error=ep)
    history = summary['history']; best = min(history, key=lambda h: (h['validation']['S'], h['epoch']))
    checks = dict(
        fixed_first_shared_job=task['spec']['job'] == job and task['spec']['manifest_sha256'] == identity(manifest),
        fresh_complete_main=task['status'] == 'complete' and config == expected and summary['config'] == expected
            and summary['status'] == 'FLOWBENCH_MAIN_TRIAL_COMPLETE' and summary['pilot_only'] is False
            and summary['phase'] == 'main_training' and summary['stage'] == 'shared' and not summary['test_evaluated'],
        actual20epochs=[h['epoch'] for h in history] == list(range(1, 21)),
        best_epoch_earliest_minimum=summary['best_epoch'] == best['epoch'] and summary['best_validation'] == best['validation'],
        exact_train_validation_ids=summary['training_indices'] == sorted(split['indices']['train'])
            and summary['validation_indices'] == sorted(split['indices']['validation'])
            and len(summary['training_indices']) == 700 and len(summary['validation_indices']) == 150,
        all150saved_rows_validation_only=len(rows) == 150 and len({r['sample'] for r in rows}) == 150
            and {r['sample'] for r in rows} == set(split['indices']['validation']),
        geometry15x10=len(groups) == 15 and all(len(g) == 10 for g in groups.values())
            and all(r['geometry'] == split['geometry_groups'][r['sample']] for r in rows),
        finite_nondegenerate_rows=all(math.isfinite(r[k]) and r[k] >= 0 for r in rows for k in ('velocity_error', 'pressure_error'))
            and all(not r['velocity_below_epsilon'] and not r['pressure_below_epsilon'] for r in rows),
        independent_macro_matches=all(abs(recalculated[k] - summary['best_validation'][k]) < 1e-12 for k in recalculated),
        history_objective_and_counts=all(h['validation']['sample_count'] == 150 and h['validation']['geometry_count'] == 15
            and math.isfinite(h['train_composite_loss']) and abs(h['validation']['S'] - .5 * (h['validation']['velocity_error'] + h['validation']['pressure_error'])) < 1e-12 for h in history),
        predeclared_hardware=summary['gpu'] == manifest['required_gpu_name'],
        bounded_actual_process_cost=0 < task['seconds'] <= manifest['trial_process_seconds_cap'],
    )
    return checks, recalculated


def run(a):
    a.review.mkdir(parents=True, exist_ok=False)
    result = dict(status='MAIN_CANARY_QC_NOT_PASSED', test_target_values_analyzed=False,
                  reviewer_source_sha256=digest(Path(__file__)), checks={})
    try:
        manifest, freeze, approval = require_authorization(a.manifest, a.freeze, a.authorization)
        folder = a.output / 'shared/r0_s00'; envelope = read(folder / 'task.json'); task = envelope['payload']
        if identity(task) != envelope['sha256'] or task['status'] != 'complete': raise ValueError('First task incomplete or damaged')
        training = folder / 'training'; summary = read(training / 'summary.json'); config = read(folder / 'config.json')
        split = read(a.root / manifest['dataset_relative'] / 'split.json')
        checks, recalculated = report_checks(manifest, task, config, summary, read(training / 'best_validation_rows.json'), split)
        checks['source_approval_identity'] = (task['spec']['provenance'] == dict(freeze_sha256=digest(a.freeze), authorization_sha256=digest(a.authorization))
            and summary['manifest_file_sha256'] == digest(a.manifest) and summary['freeze_file_sha256'] == digest(a.freeze)
            and summary['authorization_file_sha256'] == digest(a.authorization))
        checks['raw_backend_preserved'] = digest(training / 'backend_summary.json') == summary['backend_summary_sha256']
        checks['raw_summary_and_config_hashes'] = (digest(training / 'summary.json') == task['result']['summary_sha256']
            and digest(folder / 'config.json') == summary['config_sha256'])
        checks['actual_checkpoint_hash'] = digest(training / 'best_model.pt') == summary['checkpoint_sha256'] == task['result']['checkpoint_sha256']
        normalization = read(a.root / manifest['normalization_source_relative'])
        checks['normalization_and_split'] = (digest(a.root / manifest['normalization_source_relative']) == summary['normalization_sha256']
            and summary['split_sha256'] == manifest['split_sha256'])
        # Inspect only our newly generated, hash-verified checkpoint on CPU.
        sys.path.insert(0, str(a.root / 'test/training'))
        import torch
        from train_flowbench_smoke import SteadyFNO
        checkpoint = torch.load(training / 'best_model.pt', map_location='cpu', weights_only=True)
        model = SteadyFNO(**{k: manifest['training'][k] for k in ('width', 'layers', 'modes')})
        checks['checkpoint_recipe_and_normalization'] = checkpoint['config'] == config and checkpoint['normalization'] == normalization['stats']
        checks['model_keys_shapes_and_finite_fp32'] = (set(checkpoint['model']) == set(model.state_dict())
            and all(v.shape == model.state_dict()[k].shape and torch.isfinite(v).all().item()
                    and v.dtype in (torch.float32, torch.complex64) for k, v in checkpoint['model'].items()))
        model.load_state_dict(checkpoint['model'], strict=True)
        result.update(status='MAIN_CANARY_QC_PASSED' if all(checks.values()) else 'MAIN_CANARY_QC_NOT_PASSED',
                      checks=checks, independently_aggregated_validation=recalculated,
                      process_seconds=task['seconds'], best_epoch=summary['best_epoch'], gpu=summary['gpu'],
                      checkpoint_sha256=summary['checkpoint_sha256'], freeze_sha256=digest(a.freeze))
    except Exception as error:
        result['error'] = repr(error)
    from main_workflow import atomic
    atomic(a.review / 'summary.json', result)
    return result['status'] == 'MAIN_CANARY_QC_PASSED'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'manifest', 'freeze', 'authorization', 'output', 'review'): p.add_argument('--' + key, type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)

"""Independent small-report QC; never loads flow arrays, weights or test targets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from allocation_budget import PhysicalBudget
from persistent_search import PersistentSearch


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def macro(rows):
    grouped = {}
    for row in rows: grouped.setdefault(row['geometry'], []).append(row)
    if len(rows) != 150 or len(grouped) != 15: raise ValueError('Full 150/15 validation expected')
    return sum(sum((r['velocity_error'] + r['pressure_error']) / 2 for r in group) / len(group)
               for group in grouped.values()) / len(grouped)


def run(a):
    base = a.root / 'results/llm_search_prepare'; a.output.mkdir(parents=True, exist_ok=False)
    latency_dir = base / ('latency_' + a.latency); controller_dir = base / ('persistent_controller_' + a.controller)
    latency = read(latency_dir / 'summary.json'); controller = read(controller_dir / 'summary.json')
    records = read(latency_dir / 'measurements.json'); config = latency['config']
    checks = {'latency_completed': latency['status'] == 'FLOWBENCH_CURRENT_CHECKPOINT_LATENCY_AUDIT_COMPLETE',
              'controller_completed': controller['status'] == 'PERSISTENT_CONTROLLER_REAL_PILOT_COMPLETE' and all(controller['checks'].values()),
              'no_test_evaluation': not latency['test_evaluated'] and not controller['test_evaluated'] and not latency['test_targets_loaded'],
              'all_timing_wrappers_equal': all(r['wrappers_bit_identical'] for r in latency['output_equivalence_checks'])}
    old = a.root / config['checkpoint_relative']
    checks['unchanged_model_sources'] = all(digest(old / 'sources' / name) == digest(a.root / 'test/training' / name)
        for name in ('train_flowbench_smoke.py', 'fno_model.py', 'flowbench_physics.py'))
    for mode in config['mode_order']:
        selected = [r for r in records if r['mode'] == mode]
        checks[mode + '_measurement_count'] = len(selected) == config['rounds'] * config['measured_per_mode_per_round']
        for key in ('wall_ms', 'cuda_event_ms'):
            values = np.array([r[key] for r in selected])
            checks[mode + '_' + key + '_finite'] = bool(np.isfinite(values).all() and (values >= 0).all())
            for percentile in (50, 95, 99):
                checks[f'{mode}_{key}_p{percentile}_recomputed'] = abs(float(np.percentile(values, percentile)) - latency['per_mode'][mode][key]['p' + str(percentile)]) < 1e-10
        for k in range(1, config['rounds'] + 1):
            part = [r for r in selected if r['round'] == k]
            values = [r['wall_ms'] for r in part]
            checks[f'{mode}_round{k}_p99'] = len(part) == config['measured_per_mode_per_round'] and abs(float(np.percentile(values, 99)) - latency['per_round'][mode][str(k)]['wall_ms']['p99']) < 1e-10
        checks[mode + '_five_distinct_val_geometries'] = set(r['sample'] for r in selected) == set(latency['selected_validation_ids']) and len(latency['selected_validation_ids']) == 5
    pilot = read(a.root / 'test/config/persistent_controller_pilot_20261021.json')
    data_split = read(a.root / read(a.root / pilot['pilot_config_relative'])['dataset_relative'] / 'split.json')
    feedback = read(a.root / pilot['source_relative'] / 'feedback.json')
    checks['equal_shared_history'] = all(r['search']['state']['spec']['history'] == controller['methods'][0]['search']['state']['spec']['history'] for r in controller['methods'])
    for row in controller['methods']:
        label = row['label']; search_dir = controller_dir / label / 'search'; spec = row['search']['state']['spec']
        with PersistentSearch(search_dir, identity=spec['identity'], method=spec['method'], seed=spec['seed'],
                history=spec['history'], training_limit=spec['training_limit'], attempt_limit=spec['attempt_limit'],
                shared_seconds=spec['shared_seconds'], decision_limit=spec['decision_limit']) as restored:
            checks[label + '_real_saved_study_reloads'] = restored.export() == row['search']
            items = restored.state['attempts']
            checks[label + '_single_adaptive_trial_counted'] = len(items) == 1 and restored.export()['attempts_spent'] == 4
            item = items[0]
            trial_dir = controller_dir / label / ('trial_' + str(item['number']))
            summary = read(trial_dir / 'summary.json'); rows = read(trial_dir / 'best_validation_rows.json')
            score = macro(rows)
            checks[label + '_macro_score_matches_ledger'] = abs(score - item['S']) < 1e-12 and abs(score - summary['best_validation']['S']) < 1e-12
            checks[label + '_same_train_validation_membership'] = (summary['training_indices'] == sorted(data_split['indices']['train'])
                and summary['validation_indices'] == sorted(data_split['indices']['validation'])
                and set(r['sample'] for r in rows) == set(data_split['indices']['validation']) and not summary['test_evaluated'])
            checks[label + '_same_normalization_seed_params'] = (summary['normalization_sha256'] == controller['identity']['normalization_sha256']
                and summary['config']['seed'] == pilot['training_seed'] and summary['config']['training']['epochs'] == 2
                and all(summary['config']['training'][k] == v for k, v in item['parameters'].items()))
            outcome = read(controller_dir / label / ('trial_' + str(item['number']) + '_outcome.json'))['outcome']
            checks[label + '_measured_training_seconds_charged'] = outcome['wall_seconds'] == item['training_seconds']
            if label == 'tpe':
                exported = restored.sampler.export()
                checks['tpe_restored_real_tell'] = len(exported['trials']) == 4 and exported['trials'][-1]['S'] == item['S'] and exported['trials'][-1]['state'] == 'COMPLETE'
            if label == 'granite33_8b':
                from smoke_local_configurator import parse_reply
                proposal_dir = controller_dir / label / ('proposal_' + str(item['number']))
                proposal = read(proposal_dir / 'summary.json'); metadata = item['metadata']
                parsed, errors, strict = parse_reply(proposal['attempts'][-1]['raw_output'], {r['id'] for r in feedback['history']})
                checks['llm_v2_reason_and_evidence_persisted'] = (not errors and strict and parsed == proposal['configuration']
                    and parsed == metadata['configuration_with_reason'] and proposal['prompt_contract_version'] == 2
                    and digest(proposal_dir / 'summary.json') == metadata['proposal_summary_sha256'])
                checks['llm_process_decision_seconds_charged'] = metadata['outcome']['wall_seconds'] == item['decision_seconds']
    allocation_cfg = read(a.root / pilot['allocation_config_relative'])
    with PhysicalBudget(a.root / pilot['allocation_ledger_relative'], **allocation_cfg) as ledger:
        allocation = ledger.report()
        checks['physical_jobs_settled_and_within_global_cap'] = (allocation['overrun_gpu_seconds'] == 0
            and len(allocation['state']['allocations']) == 2 and all(r['status'] == 'settled' for r in allocation['state']['allocations'])
            and {r['job_id'] for r in allocation['state']['allocations']} == {a.latency, a.controller})
    result = {'status': 'CONTROLLER_LATENCY_INTEGRATION_QC_PASSED' if all(checks.values()) else 'CONTROLLER_LATENCY_INTEGRATION_QC_NOT_PASSED',
              'checks': checks, 'latency_job': a.latency, 'controller_job': a.controller,
              'latency_per_mode': latency['per_mode'], 'allocation': allocation, 'test_evaluated': False,
              'warning': 'Only technical integration, one LLM family, two epochs, one proposal per method. Not comparative efficacy evidence or main-search readiness certification.'}
    save(a.output / 'summary.json', result)
    lines = ['PREPOJENIE EXPERIMENTU A LATENCIA', result['status'],
             'Overený reálny tréning cez persistent random/TPE a lokálny Granite v2 suloženým dôvodom.',
             'Žiadne testové chyby ani výsledok oprínose LLM.']
    for mode, values in latency['per_mode'].items(): lines.append(f"{mode}: wall p99={values['wall_ms']['p99']:.4f}ms")
    lines.append(f"Fyzická GPU spotreba sledovaných jobov vrátane1s/job rezervy: {allocation['charged_or_reserved_gpu_seconds']/3600:.6f}GPU-h.")
    lines += [f'{key}: {value}' for key, value in checks.items()]
    (a.output / 'vysledok.txt').write_text('\n'.join(lines) + '\n')
    print(result['status'], 'checks', len(checks), flush=True)
    return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--latency', required=True); p.add_argument('--controller', required=True)
    args = p.parse_args()
    if not args.latency.isdigit() or not args.controller.isdigit(): raise ValueError('Numeric job IDs required')
    raise SystemExit(0 if run(args) else 1)

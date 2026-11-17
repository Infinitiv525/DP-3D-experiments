"""Two genuine sequential attempts per method, still a short technical pilot."""
import argparse
import os
from pathlib import Path
import time

from allocation_budget import PhysicalBudget
from configurator_contract_v3 import SYSTEM_SHA256, VERSION, messages, canonical
from persistent_search import PersistentSearch
from preflight_support import read, digest, save, verify_freeze
from sequential_feedback import build_feedback
from search_controls import run_bounded_process
import hashlib


def budget_entry(root, config, method):
    deadline = time.monotonic() + 15
    while True:
        try:
            with PhysicalBudget(root / config['allocation_ledger_relative'], **read(root / config['allocation_config_relative'])) as ledger:
                found = [r for r in ledger.state['allocations'] if r['label'] == method and r['job_id'] == os.environ['SLURM_JOB_ID']]
                if len(found) != 1 or found[0]['wall_limit_seconds'] != (360 if method == 'controls' else 600): raise ValueError('Physical reservation required')
                return dict(found[0])
        except BlockingIOError:
            if time.monotonic() >= deadline: raise
            time.sleep(.1)


def measurement_history(search):
    return {'a' + str(i['number']).zfill(2): i['metadata']['training']['measurement']
            for i in search.state['attempts'] if i['phase'] == 'finished' and i['status'] == 'complete'}


def run(a):
    started = time.perf_counter(); freeze = verify_freeze(a.root, a.freeze)
    cfg = read(a.root / 'test/config/sequential_preflight_20261021.json')
    if cfg['phase'] != 'technical_pilot' or cfg['main_authorized'] or cfg['epochs'] != 2 or cfg['attempt_limit_including_shared'] != 5:
        raise ValueError('This entry point cannot run main experiments')
    if a.method not in cfg['methods'] + ['controls']: raise ValueError('Unknown method')
    allocation = budget_entry(a.root, cfg, a.method)
    base = read(a.root / cfg['pilot_config_relative']); source = a.root / cfg['source_relative']; data = a.root / base['dataset_relative']
    original = read(source / 'feedback.json')['history']; shared = sum(r['training_gpu_hours'] * 3600 for r in original)
    imported = [{'parameters': {k: r[k] for k in base['manual']}, 'S': r['S']} for r in original]
    if len(imported) != 3: raise ValueError('Three shared starts required')
    audit = read(a.root / cfg['data_audit_relative'] / 'summary.json')
    if audit['status'] != 'FLOWBENCH_DATA_AUDIT_PASSED' or digest(data / 'split.json') != base['split_sha256']:
        raise ValueError('Audited immutable dataset required')
    a.output.mkdir(parents=True, exist_ok=True)
    methods = ['random', 'tpe'] if a.method == 'controls' else [a.method]
    wall_cap = cfg['controls_controller_seconds'] if a.method == 'controls' else cfg['family_controller_seconds']
    report = {'status': 'RUNNING', 'phase': 'technical_pilot', 'test_evaluated': False,
              'freeze_sha256': digest(a.freeze), 'system_prompt_sha256': SYSTEM_SHA256, 'allocation': allocation, 'methods': []}
    save(a.output / 'summary.json', report)
    def available(limit):
        result = min(limit, wall_cap - (time.perf_counter() - started) - 10)
        if result <= 5: raise TimeoutError('Controller wall budget exhausted')
        return result
    for label in methods:
        folder = a.output / label; folder.mkdir(exist_ok=True); is_llm = label not in ('random', 'tpe')
        identity = {'freeze_sha256': digest(a.freeze), 'method': label, 'normalization_sha256': digest(source / 'normalization.json'),
                    'split_sha256': base['split_sha256'], 'phase': 'technical_pilot'}
        with PersistentSearch(folder / 'search', identity=identity, method='external' if is_llm else label,
                seed=cfg['sampler_seed'], history=imported, training_limit=shared + cfg['additional_training_seconds_per_method'],
                attempt_limit=cfg['attempt_limit_including_shared'], shared_seconds=shared,
                decision_limit=cfg['decision_seconds_per_llm'] if is_llm else 0) as search:
            search.recover_interrupted()
            halt_reason = 'attempt_cap'
            while search.pending or search.export()['attempts_spent'] < cfg['attempt_limit_including_shared']:
                if search.pending is None:
                    if search.export()['remaining_training_seconds'] <= 5: halt_reason = 'training_time_cap'; break
                    cap = available(min(cfg['decision_cap_seconds'], search.export()['remaining_decision_seconds'])) if is_llm else None
                    if is_llm and cap <= 5: halt_reason = 'decision_time_cap'; break
                    search.ask(decision_cap=cap)
                item = search.pending; number = item['number']; identifier = 'a' + str(number).zfill(2)
                feedback = build_feedback(original, search.export(), measurement_history(search), cfg['epochs'])
                feedback_path = folder / (identifier + '_feedback.json')
                save(feedback_path, feedback)
                if item['phase'] == 'selecting':
                    llm_config = read(a.root / cfg['llm_config_relative'])
                    index = next(i for i, spec in enumerate(llm_config['models']) if spec['label'] == label)
                    dest = folder / (identifier + '_proposal')
                    outcome = run_bounded_process([str(a.root / 'envs/llm-feasibility-20261021/bin/python'), '-B',
                        str(a.root / 'test/llm/propose_configuration_v3.py'), '--config', str(a.root / cfg['llm_config_relative']),
                        '--preparation', str(a.root / cfg['preparation_relative']), '--feedback', str(feedback_path),
                        '--index', str(index), '--output', str(dest)], available(item['decision_seconds']), folder / (identifier + '_proposal.log'))
                    result = read(dest / 'summary.json') if (dest / 'summary.json').exists() else {}
                    good = (outcome['exit_code'] == 0 and result.get('status') == 'LOCAL_CONFIGURATOR_V3_COMPLETE'
                            and result.get('system_prompt_sha256') == SYSTEM_SHA256 and result.get('contract_version') == VERSION
                            and result.get('feedback_sha256') == digest(feedback_path) and result['model']['label'] == label)
                    if good:
                        params = {k: result['configuration'][k] for k in base['manual']}
                        search.selected(params, outcome['wall_seconds'], {'proposal': {'configuration_with_reason': result['configuration'],
                            'summary_sha256': digest(dest / 'summary.json'), 'outcome': outcome, 'feedback_sha256': digest(feedback_path),
                            'system_prompt_sha256': SYSTEM_SHA256, 'initial_messages_sha256': result['initial_messages_sha256']}})
                    else:
                        search.selection_failed(outcome['wall_seconds'], 'decision_timeout' if outcome['timed_out'] else 'invalid_reply',
                                                {'proposal': {'outcome': outcome, 'status': result.get('status'), 'feedback_sha256': digest(feedback_path)}})
                if search.pending and search.pending['phase'] == 'ready':
                    limit = search.reserve_training(available(cfg['trial_cap_seconds']))
                    trial_config = {'revision': base['revision'], 'seed': cfg['training_seed_base'] + number,
                        'pilot_only': True, 'purpose': cfg['warning'], 'training': dict(base['training'], epochs=cfg['epochs'], **search.pending['parameters'])}
                    path = folder / (identifier + '_config.json'); save(path, trial_config); dest = folder / (identifier + '_training')
                    outcome = run_bounded_process([str(a.root / 'envs/training-torch-2.12-cu130/bin/python'), '-B',
                        str(a.root / 'test/training/train_flowbench_protocol.py'), '--config', str(path), '--data', str(data),
                        '--normalization', str(source / 'normalization.json'), '--output', str(dest), '--max-wall-seconds', str(max(1, limit - 10))],
                        limit, folder / (identifier + '_training.log'))
                    result = read(dest / 'summary.json') if (dest / 'summary.json').exists() else {}
                    good = outcome['exit_code'] == 0 and result.get('status') == 'FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE'
                    if good and (result['test_evaluated'] or result['normalization_sha256'] != identity['normalization_sha256']):
                        raise ValueError('Unexpected training provenance; retain reservation')
                    outcome_path = folder / (identifier + '_outcome.json'); save(outcome_path, outcome)
                    training = {'outcome_sha256': digest(outcome_path), 'config_sha256': digest(path), 'outcome': outcome}
                    if good:
                        training.update(summary_sha256=digest(dest / 'summary.json'),
                            measurement=dict(result['best_validation'], divergence=result['predicted_divergence_rms_mean']))
                    search.pending['metadata']['training'] = training
                    search.pending['metadata']['feedback_sha256'] = digest(feedback_path)
                    search.complete(outcome['wall_seconds'], result['best_validation']['S'] if good else None,
                                    'complete' if good else 'timeout' if outcome['timed_out'] else 'failed')
                save(folder / 'search_report.json', search.export())
                print('SEQUENTIAL_ATTEMPT_FINISHED', label, identifier, search.state['attempts'][-1]['status'], flush=True)
            exported = search.export(); finished = exported['state']['attempts']
            row = {'label': label, 'search': exported, 'halt_reason': halt_reason,
                   'completed_adaptive_trials': sum(i['status'] == 'complete' for i in finished)}
            if len(finished) == 2:
                feedback2 = read(folder / 'a04_feedback.json')
                row['second_feedback_includes_first_attempt'] = feedback2['history'][-1]['id'] == 'a03' and len(feedback2['history']) == 4
            else: row['second_feedback_includes_first_attempt'] = False
            report['methods'].append(row); save(a.output / 'summary.json', report)
    checks = {'two_completed_attempts_each': all(r['completed_adaptive_trials'] == 2 for r in report['methods']),
              'two_step_feedback_each': all(r['second_feedback_includes_first_attempt'] for r in report['methods']),
              'attempt_cap_reached': all(r['search']['attempts_spent'] == 5 for r in report['methods']),
              'no_training_overrun': all(r['search']['training_overrun_seconds'] == 0 for r in report['methods']),
              'no_decision_overrun': all(r['search']['decision_overrun_seconds'] == 0 for r in report['methods'])}
    report.update(status='SEQUENTIAL_TECHNICAL_PREFLIGHT_COMPLETE' if all(checks.values()) else 'SEQUENTIAL_TECHNICAL_PREFLIGHT_INCOMPLETE',
                  checks=checks, elapsed_seconds=time.perf_counter() - started, warning=cfg['warning'])
    save(a.output / 'summary.json', report); return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--method', required=True); p.add_argument('--output', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)

"""Real train/validation subprocesses behind persistent per-search budgets.

Dedicated technical-pilot controller, not an approval gate for main experiments.
The outer GPU allocation must already be reserved by allocation_budget.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from allocation_budget import PhysicalBudget
from persistent_search import PersistentSearch
from search_controls import run_bounded_process, validate_parameters


def read(path): return json.loads(path.read_text())
def save(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(a):
    started = time.perf_counter(); cfg = read(a.config)
    if not cfg['pilot_only'] or cfg['attempt_limit_including_shared'] != 4 or cfg['epochs'] != 2:
        raise ValueError('This controller only supports the bounded technical pilot')
    allocation_cfg = read(a.root / cfg['allocation_config_relative'])
    with PhysicalBudget(a.root / cfg['allocation_ledger_relative'], **allocation_cfg) as ledger:
        entries = [r for r in ledger.state['allocations'] if r['job_id'] == os.environ['SLURM_JOB_ID']]
        if len(entries) != 1 or entries[0]['label'] != 'controller_integration' or entries[0]['wall_limit_seconds'] != 600:
            raise ValueError('Matching pre-submission physical reservation required')
        allocation_entry = dict(entries[0])
    a.output.mkdir(parents=True, exist_ok=True)
    base = read(a.root / cfg['pilot_config_relative']); source = a.root / cfg['source_relative']
    audit = read(a.root / cfg['data_audit_relative'] / 'summary.json')
    if audit['status'] != 'FLOWBENCH_DATA_AUDIT_PASSED' or not all(audit['checks'].values()): raise ValueError('Data audit required')
    data = a.root / base['dataset_relative']
    if digest(data / 'split.json') != base['split_sha256'] or audit['split_sha256'] != base['split_sha256']:
        raise ValueError('Split provenance differs')
    original_feedback = read(source / 'feedback.json'); rows = original_feedback['history']
    keys = tuple(base['manual']); imported = [{'parameters': {k: r[k] for k in keys}, 'S': r['S']} for r in rows]
    if len(imported) != 3: raise ValueError('Exactly three shared starts required')
    shared = sum(r['training_gpu_hours'] * 3600 for r in rows)
    source_names = ['llm/pilot_persistent_controller.py', 'llm/persistent_search.py', 'llm/search_controls.py',
                    'llm/propose_local_configuration.py', 'llm/smoke_local_configurator.py',
                    'training/train_flowbench_protocol.py', 'training/train_flowbench_smoke.py',
                    'training/flowbench_protocol_metrics.py', 'training/flowbench_physics.py', 'training/fno_model.py']
    identity = {'config_sha256': digest(a.config), 'split_sha256': base['split_sha256'],
                'cache_audit_sha256': digest(data / 'pressure_gauge_audit.json'),
                'feedback_sha256': digest(source / 'feedback.json'), 'normalization_sha256': digest(source / 'normalization.json'),
                'llm_config_sha256': digest(a.root / base['llm_config_relative']),
                'data_audit_sha256': digest(a.root / cfg['data_audit_relative'] / 'summary.json'),
                'source_sha256': {name: digest(a.root / 'test' / name) for name in source_names}}
    report = {'status': 'RUNNING', 'technical_pilot': True, 'test_evaluated': False, 'identity': identity,
              'allocation_entry': allocation_entry, 'methods': [], 'warning': cfg['warning']}
    save(a.output / 'summary.json', report)

    def available(cap):
        seconds = min(cap, cfg['controller_process_seconds'] - (time.perf_counter() - started) - 10)
        if seconds <= 5: raise TimeoutError('Controller wall budget exhausted')
        return seconds

    for label in cfg['methods']:
        folder = a.output / label; folder.mkdir(exist_ok=True)
        is_llm = label not in ('random', 'tpe')
        with PersistentSearch(folder / 'search', identity=dict(identity, method=label),
                method='external' if is_llm else label, seed=cfg['sampler_seed'], history=imported,
                training_limit=shared + cfg['additional_training_seconds_per_method'],
                attempt_limit=cfg['attempt_limit_including_shared'], shared_seconds=shared,
                decision_limit=cfg['decision_seconds_per_llm'] if is_llm else 0) as search:
            search.recover_interrupted()  # refuses a still-active old owner, does not relaunch lost training
            if search.pending is None and search.export()['attempts_spent'] < cfg['attempt_limit_including_shared']:
                search.ask(decision_cap=available(cfg['decision_seconds_per_llm']) if is_llm else None)
            if search.pending and search.pending['phase'] == 'selecting':
                proposal_dir = folder / ('proposal_' + str(search.pending['number']))
                feedback = dict(original_feedback)
                feedback['remaining_training_gpu_hours'] = search.export()['remaining_training_seconds'] / 3600
                save(folder / 'feedback.json', feedback)
                limit = search.pending['decision_seconds']
                outcome = run_bounded_process([str(a.root / 'envs/llm-feasibility-20261021/bin/python'), '-B',
                    str(a.root / 'test/llm/propose_local_configuration.py'), '--config', str(a.root / base['llm_config_relative']),
                    '--preparation', str(a.root / base['preparation_relative']), '--feedback', str(folder / 'feedback.json'),
                    '--index', str(cfg['llm_index']), '--output', str(proposal_dir)],
                    available(limit), folder / 'proposal.log')
                result = read(proposal_dir / 'summary.json') if (proposal_dir / 'summary.json').exists() else {}
                valid = (outcome['exit_code'] == 0 and result.get('status') == 'LOCAL_FEEDBACK_PROPOSAL_COMPLETE'
                         and result.get('prompt_contract_version') == 2 and result['model']['label'] == label)
                if valid:
                    params = {k: result['configuration'][k] for k in keys}; validate_parameters(params)
                    search.selected(params, outcome['wall_seconds'], {'configuration_with_reason': result['configuration'],
                        'proposal_summary_sha256': digest(proposal_dir / 'summary.json'),
                        'feedback_sha256': digest(folder / 'feedback.json'), 'outcome': outcome,
                        'prompt_contract_version': result['prompt_contract_version'], 'model': result['model']})
                else:
                    search.selection_failed(outcome['wall_seconds'], 'decision_timeout' if outcome['timed_out'] else 'invalid_reply',
                                            {'outcome': outcome, 'proposal_status': result.get('status')})
            if search.pending and search.pending['phase'] == 'ready':
                item = dict(search.pending); number = item['number']
                limit = search.reserve_training(available(cfg['additional_training_seconds_per_method']))
                trial_cfg = {'revision': base['revision'], 'seed': cfg['training_seed'], 'pilot_only': True,
                             'purpose': cfg['purpose'], 'training': dict(base['training'], epochs=cfg['epochs'], **item['parameters'])}
                path = folder / ('trial_' + str(number) + '.json'); save(path, trial_cfg)
                dest = folder / ('trial_' + str(number))
                outcome = run_bounded_process([str(a.root / 'envs/training-torch-2.12-cu130/bin/python'), '-B',
                    str(a.root / 'test/training/train_flowbench_protocol.py'), '--config', str(path), '--data', str(data),
                    '--normalization', str(source / 'normalization.json'), '--output', str(dest),
                    '--max-wall-seconds', str(max(1, limit - 10))], limit, folder / ('trial_' + str(number) + '.log'))
                result = read(dest / 'summary.json') if (dest / 'summary.json').exists() else {}
                success = outcome['exit_code'] == 0 and result.get('status') == 'FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE'
                if success and (result['test_evaluated'] or result['normalization_sha256'] != identity['normalization_sha256']):
                    raise ValueError('Trainer provenance changed; leave conservative reservation')
                save(folder / ('trial_' + str(number) + '_outcome.json'), {'outcome': outcome, 'trainer_status': result.get('status')})
                search.complete(outcome['wall_seconds'], result['best_validation']['S'] if success else None,
                                'complete' if success else 'timeout' if outcome['timed_out'] else 'failed')
            exported = search.export()
            successful = [r for r in exported['state']['attempts'] if r['status'] == 'complete']
            next_blocked = False
            try: search.ask(decision_cap=1 if is_llm else None)
            except RuntimeError: next_blocked = True
            # The test must never consume a surprise second adaptive attempt.
            if not next_blocked: raise RuntimeError('Attempt cap guard failed')
            report['methods'].append({'label': label, 'search': exported, 'next_attempt_blocked': next_blocked,
                                       'completed_adaptive_trials': len(successful)})
            save(a.output / 'summary.json', report)
            print('PERSISTENT_METHOD_COMPLETE', label, len(successful), flush=True)
    checks = {'three_methods_completed': len(report['methods']) == 3 and all(r['completed_adaptive_trials'] == 1 for r in report['methods']),
              'all_attempt_caps_block': all(r['next_attempt_blocked'] for r in report['methods']),
              'no_training_budget_overrun': all(r['search']['training_overrun_seconds'] == 0 for r in report['methods']),
              'no_decision_budget_overrun': all(r['search']['decision_overrun_seconds'] == 0 for r in report['methods'])}
    report.update(checks=checks, status='PERSISTENT_CONTROLLER_REAL_PILOT_COMPLETE' if all(checks.values()) else 'PERSISTENT_CONTROLLER_REAL_PILOT_INCOMPLETE',
                  elapsed_seconds=time.perf_counter() - started)
    save(a.output / 'summary.json', report)
    return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); raise SystemExit(0 if run(p.parse_args()) else 1)

"""Two fresh adaptive Ministral trials with the uniformly tested transport rule.

Separate study and accounting; never converts historical failed trials to wins.
Same v3 SYSTEM, first feedback, decoding, training length and seeds as preflight.
"""
import argparse
import os
from pathlib import Path
import time
from configurator_contract_v3 import SYSTEM_SHA256
from configurator_transport_v1 import POLICY
from persistent_search import PersistentSearch
from preflight_support import read, save, digest, verify_freeze
from run_sequential_preflight import budget_entry, measurement_history
from sequential_feedback import build_feedback
from search_controls import run_bounded_process


def run(a):
    started = time.perf_counter(); freeze = verify_freeze(a.root, a.freeze)
    cfg = read(a.root / 'test/config/sequential_preflight_20261021.json'); base = read(a.root / cfg['pilot_config_relative'])
    replay = read(a.root / 'results/llm_search_prepare/transport_123870/summary.json')
    if replay['status'] != 'UNIFORM_TRANSPORT_REPLAY_PASSED': raise ValueError('Uniform four-family transport replay required')
    allocation = budget_entry(a.root, cfg, 'transport_ministral3_8b')
    source = a.root / cfg['source_relative']; original = read(source / 'feedback.json')['history']
    imported = [{'parameters': {k: r[k] for k in base['manual']}, 'S': r['S']} for r in original]
    shared = sum(r['training_gpu_hours'] * 3600 for r in original)
    a.output.mkdir(parents=True, exist_ok=False)
    identity = {'phase': 'technical_pilot', 'model': 'ministral3_8b', 'transport_policy': POLICY,
        'freeze_sha256': digest(a.freeze), 'replay_sha256': digest(a.root / 'results/llm_search_prepare/transport_123870/summary.json'),
        'normalization_sha256': digest(source / 'normalization.json'), 'system_prompt_sha256': SYSTEM_SHA256,
        'sources': {name: digest(a.root / 'test/llm' / name) for name in ('run_transport_confirmation.py',
            'propose_configuration_transport_v1.py', 'configurator_transport_v1.py')}}
    save(a.output / 'identity.json', identity)
    report = {'status': 'RUNNING', 'identity': identity, 'allocation': allocation, 'test_evaluated': False,
              'historical_ministral_failures_preserved': True}
    save(a.output / 'summary.json', report)
    def available(cap):
        seconds = min(cap, 530 - (time.perf_counter() - started) - 10)
        if seconds <= 5: raise TimeoutError('Transport confirmation cap exhausted')
        return seconds
    with PersistentSearch(a.output / 'search', identity=identity, method='external', seed=cfg['sampler_seed'],
            history=imported, training_limit=shared + 300, attempt_limit=5, shared_seconds=shared, decision_limit=420) as search:
        for number in (3, 4):
            identifier = 'a' + str(number).zfill(2)
            search.ask(decision_cap=available(min(210, search.export()['remaining_decision_seconds'])))
            feedback = build_feedback(original, search.export(), measurement_history(search), 2)
            path = a.output / (identifier + '_feedback.json'); save(path, feedback)
            dest = a.output / (identifier + '_proposal')
            outcome = run_bounded_process([str(a.root / 'envs/llm-feasibility-20261021/bin/python'), '-B',
                str(a.root / 'test/llm/propose_configuration_transport_v1.py'), '--config', str(a.root / cfg['llm_config_relative']),
                '--preparation', str(a.root / cfg['preparation_relative']), '--feedback', str(path), '--index', '1', '--output', str(dest)],
                available(search.pending['decision_seconds']), a.output / (identifier + '_proposal.log'))
            result = read(dest / 'summary.json') if (dest / 'summary.json').exists() else {}
            good = (outcome['exit_code'] == 0 and result.get('status') == 'LOCAL_CONFIGURATOR_V3_COMPLETE'
                and result.get('transport_policy') == POLICY and result.get('system_prompt_sha256') == SYSTEM_SHA256
                and result.get('feedback_sha256') == digest(path) and result['model']['label'] == 'ministral3_8b')
            if not good:
                search.selection_failed(outcome['wall_seconds'], 'decision_timeout' if outcome['timed_out'] else 'invalid_reply', {'outcome': outcome})
                print('TRANSPORT_ATTEMPT_FINISHED', identifier, 'invalid_reply', flush=True); continue
            params = {k: result['configuration'][k] for k in base['manual']}
            search.selected(params, outcome['wall_seconds'], {'proposal': {'summary_sha256': digest(dest / 'summary.json'),
                'configuration_with_reason': result['configuration'], 'outcome': outcome, 'transport_policy': POLICY}})
            limit = search.reserve_training(available(150)); training = a.output / (identifier + '_training')
            config = {'revision': base['revision'], 'seed': 310000 + number, 'pilot_only': True,
                      'purpose': 'Transport-format confirmation only, not main experiment or efficacy evidence.',
                      'training': dict(base['training'], **params)}
            config_path = a.output / (identifier + '_config.json'); save(config_path, config)
            outcome = run_bounded_process([str(a.root / 'envs/training-torch-2.12-cu130/bin/python'), '-B',
                str(a.root / 'test/training/train_flowbench_protocol.py'), '--config', str(config_path),
                '--data', str(a.root / base['dataset_relative']), '--normalization', str(source / 'normalization.json'),
                '--output', str(training), '--max-wall-seconds', str(max(1, limit - 10))], limit, a.output / (identifier + '_training.log'))
            trained = read(training / 'summary.json') if (training / 'summary.json').exists() else {}
            good = outcome['exit_code'] == 0 and trained.get('status') == 'FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE'
            if good and (trained['test_evaluated'] or trained['normalization_sha256'] != identity['normalization_sha256']): raise ValueError('Provenance changed')
            metadata = {'outcome': outcome}
            if good: metadata.update(summary_sha256=digest(training / 'summary.json'),
                                    measurement=dict(trained['best_validation'], divergence=trained['predicted_divergence_rms_mean']))
            search.pending['metadata']['training'] = metadata
            search.complete(outcome['wall_seconds'], trained['best_validation']['S'] if good else None,
                            'complete' if good else 'timeout' if outcome['timed_out'] else 'failed')
            save(a.output / 'search_report.json', search.export())
            print('TRANSPORT_ATTEMPT_FINISHED', identifier, search.state['attempts'][-1]['status'], flush=True)
        exported = search.export(); items = exported['state']['attempts']
        first_reference = a.root / 'results/llm_search_prepare/sequential_123758/qwen3_8b/a03_feedback.json'
        checks = {'two_new_trials_complete': len(items) == 2 and all(i['status'] == 'complete' for i in items),
                  'attempts_charged': exported['attempts_spent'] == 5,
                  'same_first_feedback_as_original_four_families': digest(a.output / 'a03_feedback.json') == digest(first_reference),
                  'second_feedback_includes_first': read(a.output / 'a04_feedback.json')['history'][-1]['id'] == 'a03',
                  'no_training_overrun': exported['training_overrun_seconds'] == 0, 'no_decision_overrun': exported['decision_overrun_seconds'] == 0}
        report.update(status='TRANSPORT_CONFIRMATION_COMPLETE' if all(checks.values()) else 'TRANSPORT_CONFIRMATION_INCOMPLETE',
                      checks=checks, search=exported, elapsed_seconds=time.perf_counter() - started)
        save(a.output / 'summary.json', report); return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); raise SystemExit(0 if run(p.parse_args()) else 1)

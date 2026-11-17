"""Verify the separate confirmation; preserve all original failures and costs."""
import argparse
from pathlib import Path
from configurator_contract_v3 import SYSTEM_SHA256, validate_feedback
from configurator_transport_v1 import POLICY, parse_reply
from persistent_search import PersistentSearch
from preflight_support import read, digest, save, verify_freeze
from review_sequential_preflight import macro
from settle_allocation_uid import settle


def run(a):
    verify_freeze(a.root, a.freeze); a.output.mkdir(parents=True, exist_ok=False)
    cfg = read(a.root / 'test/config/sequential_preflight_20261021.json')
    old = read(a.root / 'results/llm_search_prepare/sequential_review_123763/summary.json')
    replay = read(a.root / 'results/llm_search_prepare/transport_123870/summary.json')
    folder = a.root / 'results/llm_search_prepare' / ('transport_confirmation_' + a.job)
    confirmed = read(folder / 'summary.json'); spec = confirmed['search']['state']['spec']
    checks = {'new_two_step_confirmation_complete': confirmed['status'] == 'TRANSPORT_CONFIRMATION_COMPLETE' and all(confirmed['checks'].values()),
              'same_system_prompt': confirmed['identity']['system_prompt_sha256'] == SYSTEM_SHA256,
              'uniform_replay_all_four_passed': replay['status'] == 'UNIFORM_TRANSPORT_REPLAY_PASSED' and all(replay['checks'].values()),
              'old_ministral_failures_still_recorded': all(i['status'] == 'invalid_reply' for m in old['methods'] if m['label'] == 'ministral3_8b' for i in m['attempts']),
              'no_test_evaluation': not confirmed['test_evaluated'] and not old['test_evaluated']}
    for label in ('qwen3_8b', 'phi4_mini', 'granite33_8b', 'random', 'tpe'):
        checks[label + '_two_real_trials_complete'] = all(i['status'] == 'complete' for m in old['methods'] if m['label'] == label for i in m['attempts'])
    checks['old_raw_outputs_unchanged'] = all(digest(a.root / 'results/llm_search_prepare' /
        ('sequential_' + {'qwen3_8b':'123758', 'ministral3_8b':'123759', 'phi4_mini':'123760', 'granite33_8b':'123761'}[r['model']]) /
        r['model'] / (r['id'] + '_proposal') / 'summary.json') == r['source_summary_sha256'] for r in replay['rows'])
    with PersistentSearch(folder / 'search', identity=spec['identity'], method=spec['method'], seed=spec['seed'],
            history=spec['history'], training_limit=spec['training_limit'], attempt_limit=spec['attempt_limit'],
            shared_seconds=spec['shared_seconds'], decision_limit=spec['decision_limit']) as restored:
        checks['real_committed_study_reloads'] = restored.export() == confirmed['search']
        items = restored.state['attempts']
        for item in items:
            identifier = 'a' + str(item['number']).zfill(2); proposal_dir = folder / (identifier + '_proposal')
            proposal = read(proposal_dir / 'summary.json'); feedback = read(folder / (identifier + '_feedback.json'))
            valid_ids = validate_feedback(feedback); parsed, errors, info = parse_reply(proposal['attempts'][-1]['raw_output'], valid_ids)
            checks[identifier + '_same_public_payload_and_transport'] = (not errors and parsed == proposal['configuration']
                and parsed == item['metadata']['proposal']['configuration_with_reason'] and proposal['transport_policy'] == POLICY
                and proposal['transport_source_sha256'] == digest(a.root / 'test/llm/configurator_transport_v1.py')
                and proposal['system_prompt_sha256'] == SYSTEM_SHA256
                and proposal['attempts'][-1]['transport'] == info)
            checks[identifier + '_proposal_provenance_and_cost'] = (digest(proposal_dir / 'summary.json') == item['metadata']['proposal']['summary_sha256']
                and item['decision_seconds'] == item['metadata']['proposal']['outcome']['wall_seconds'])
            training_dir = folder / (identifier + '_training'); training = read(training_dir / 'summary.json')
            score = macro(read(training_dir / 'best_validation_rows.json'))
            checks[identifier + '_real_macro_score_and_params'] = (abs(score - item['S']) < 1e-12 and abs(score - training['best_validation']['S']) < 1e-12
                and training['config']['seed'] == 310000 + item['number'] and training['config']['training']['epochs'] == 2
                and all(training['config']['training'][k] == v for k, v in item['parameters'].items()))
            checks[identifier + '_normalization_and_training_cost'] = (training['normalization_sha256'] == spec['identity']['normalization_sha256']
                and digest(training_dir / 'summary.json') == item['metadata']['training']['summary_sha256']
                and item['training_seconds'] == item['metadata']['training']['outcome']['wall_seconds'] and not training['test_evaluated'])
        first = read(folder / 'a03_feedback.json'); second = read(folder / 'a04_feedback.json')
        checks['real_first_result_and_cost_in_second_feedback'] = (second['history'][-1]['S'] == items[0]['S']
            and second['history'][-1]['training_seconds'] == items[0]['training_seconds']
            and second['history'][-1]['decision_seconds'] == items[0]['decision_seconds']
            and abs(second['remaining_training_seconds'] - (300 - items[0]['training_seconds'])) < 1e-8)
        checks['same_initial_input_as_original_all_four'] = digest(folder / 'a03_feedback.json') == digest(a.root / 'results/llm_search_prepare/sequential_123758/qwen3_8b/a03_feedback.json')
        checks['cap_after_shared_and_two_attempts'] = restored.export()['attempts_spent'] == 5 and len(items) == 2
    allocation = settle(a.root / cfg['allocation_ledger_relative'], read(a.root / cfg['allocation_config_relative']))
    checks['all_six_allocations_settled_within_original_cap'] = (len(allocation['state']['allocations']) == 6
        and all(r['status'] == 'settled' for r in allocation['state']['allocations']) and allocation['overrun_gpu_seconds'] == 0)
    result = {'status': 'FOUR_FAMILY_TRANSPORT_CONFIRMATION_QC_PASSED' if all(checks.values()) else 'FOUR_FAMILY_TRANSPORT_CONFIRMATION_QC_NOT_PASSED',
              'original_preflight_summary_sha256': digest(a.root / 'results/llm_search_prepare/sequential_review_123763/summary.json'),
              'transport_replay_summary_sha256': digest(a.root / 'results/llm_search_prepare/transport_123870/summary.json'),
              'checks': checks, 'allocation': allocation, 'ministral_confirmation': confirmed, 'test_evaluated': False,
              'scope': 'Original three valid LLM families/control studies plus separate two-step Ministral confirmation. Uniform transport replay covers all four; not a fresh rerun of all four under the adapter.',
              'warning': 'Technical two-epoch pilots only. Original failed Ministral trials remain failed/charged. No cross-family efficacy ranking, main search, final test or retrospective replacement.'}
    save(a.output / 'summary.json', result)
    lines = ['ŠTYRI RODINY LLM - POTVRDENIE FORMÁTU A NADVÄZNOSTI', result['status'],
        'Qwen, Phi a Granite dokončili po2nové tréningy vpôvodnej v3 skúške; random/TPE tiež.',
        'Ministral dokončil samostatnú2krokovú skúšku so spoločným odobratím jediného JSON code fence.',
        'Pôvodné neúspešné pokusy zostali uložené a započítané. Žiadny hlavný experiment alebo test.',
        f"Súhrnná fyzická cena tejto fázy vrátane1s/job: {allocation['charged_or_reserved_gpu_seconds']/3600:.6f}GPU-h.",
        result['scope'], result['warning']]
    (a.output / 'vysledok.txt').write_text('\n'.join(lines) + '\n')
    print(result['status'], len(checks), flush=True); return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--job', required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.job.isdigit(): raise ValueError('Numeric job ID required')
    raise SystemExit(0 if run(args) else 1)

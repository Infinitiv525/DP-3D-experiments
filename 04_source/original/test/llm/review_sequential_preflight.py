"""Six-method sequential-history and resource QC, no flow arrays or test labels."""
import argparse
import hashlib
from pathlib import Path

from configurator_contract_v3 import parse_reply, validate_feedback, messages, canonical, SYSTEM_SHA256
from preflight_support import read, save, digest, verify_freeze
from persistent_search import PersistentSearch
from settle_allocation_uid import settle


def macro(rows):
    groups = {}
    for row in rows: groups.setdefault(row['geometry'], []).append(row)
    if len(rows) != 150 or len(groups) != 15: raise ValueError('Full validation required')
    return sum(sum((r['velocity_error'] + r['pressure_error']) / 2 for r in values) / len(values) for values in groups.values()) / len(groups)


def run(a):
    freeze = verify_freeze(a.root, a.freeze); submitted = read(a.freeze.parent / 'submitted_jobs.json')
    cfg = read(a.root / 'test/config/sequential_preflight_20261021.json')
    allocation = settle(a.root / cfg['allocation_ledger_relative'], read(a.root / cfg['allocation_config_relative']))
    a.output.mkdir(parents=True, exist_ok=False)
    checks = {'five_allocations_settled': len(allocation['state']['allocations']) == 5 and all(r['status'] == 'settled' for r in allocation['state']['allocations']),
              'within_physical_limit': allocation['overrun_gpu_seconds'] == 0, 'main_not_authorized': not cfg['main_authorized']}
    reports = []; first_messages = []; initial_histories = []; norms = set()
    for job in submitted['jobs']:
        folder = a.root / 'results/llm_search_prepare' / ('sequential_' + job['job_id'])
        summary = read(folder / 'summary.json')
        checks[job['label'] + '_completed'] = summary['status'] == 'SEQUENTIAL_TECHNICAL_PREFLIGHT_COMPLETE' and all(summary['checks'].values())
        checks[job['label'] + '_immutable_identity'] = summary['freeze_sha256'] == digest(a.freeze) and summary['system_prompt_sha256'] == SYSTEM_SHA256 and not summary['test_evaluated']
        for row in summary['methods']:
            label = row['label']; method_dir = folder / label; spec = row['search']['state']['spec']
            initial_histories.append(spec['history']); norms.add(spec['identity']['normalization_sha256'])
            with PersistentSearch(method_dir / 'search', identity=spec['identity'], method=spec['method'], seed=spec['seed'],
                    history=spec['history'], training_limit=spec['training_limit'], attempt_limit=spec['attempt_limit'],
                    shared_seconds=spec['shared_seconds'], decision_limit=spec['decision_limit']) as search:
                checks[label + '_saved_search_replays'] = search.export() == row['search']
                items = search.state['attempts']
                checks[label + '_two_attempts_counted'] = len(items) == 2 and search.export()['attempts_spent'] == 5
                checks[label + '_next_ask_blocked'] = False
                try: search.ask(decision_cap=1 if spec['method'] == 'external' else None)
                except RuntimeError: checks[label + '_next_ask_blocked'] = True
                for index, item in enumerate(items):
                    identifier = 'a' + str(item['number']).zfill(2); tag = label + '_' + identifier
                    feedback_path = method_dir / (identifier + '_feedback.json'); feedback = read(feedback_path)
                    evidence = validate_feedback(feedback)
                    checks[tag + '_history_count_and_budget'] = len(feedback['history']) == 3 + index and feedback['remaining_attempts'] == 2 - index
                    if index == 0:
                        first_messages.append(hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest())
                        checks[tag + '_same_initial_training_budget'] = abs(feedback['remaining_training_seconds'] - 300) < 1e-8
                    else:
                        previous = items[index - 1]; last = feedback['history'][-1]
                        checks[tag + '_previous_trial_visible'] = (last['id'] == 'a03' and last['parameters'] == previous['parameters']
                            and last['S'] == previous['S'] and last['status'] == previous['status']
                            and last['training_seconds'] == previous['training_seconds'] and last['decision_seconds'] == previous['decision_seconds'])
                        checks[tag + '_reduced_training_budget'] = abs(feedback['remaining_training_seconds'] - (300 - previous['training_seconds'])) < 1e-8
                    checks[tag + '_complete'] = item['status'] == 'complete'
                    if item['status'] != 'complete': continue
                    training = item['metadata']['training']; dest = method_dir / (identifier + '_training'); trained = read(dest / 'summary.json')
                    score = macro(read(dest / 'best_validation_rows.json'))
                    checks[tag + '_committed_score_and_files'] = (abs(score - item['S']) < 1e-12 and abs(score - trained['best_validation']['S']) < 1e-12
                        and digest(dest / 'summary.json') == training['summary_sha256']
                        and digest(method_dir / (identifier + '_outcome.json')) == training['outcome_sha256'])
                    checks[tag + '_fresh_training_seed_data'] = (trained['config']['seed'] == 310000 + item['number']
                        and trained['config']['training']['epochs'] == 2 and not trained['test_evaluated']
                        and len(trained['training_indices']) == 700 and len(trained['validation_indices']) == 150
                        and trained['normalization_sha256'] == spec['identity']['normalization_sha256'])
                    checks[tag + '_measured_training_charged'] = training['outcome']['wall_seconds'] == item['training_seconds']
                    if label not in ('random', 'tpe'):
                        proposal = read(method_dir / (identifier + '_proposal') / 'summary.json'); metadata = item['metadata']['proposal']
                        parsed, errors = parse_reply(proposal['attempts'][-1]['raw_output'], evidence)
                        checks[tag + '_common_prompt_decode_and_valid_json'] = (not errors and parsed == proposal['configuration']
                            and parsed == metadata['configuration_with_reason'] and proposal['system_prompt_sha256'] == SYSTEM_SHA256
                            and proposal['feedback_sha256'] == digest(feedback_path) and proposal['initial_messages_sha256'] == hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest()
                            and len(proposal['attempts']) <= 2 and proposal['generation'] == read(a.root / cfg['llm_config_relative'])['generation'])
                        checks[tag + '_proposal_cost_and_provenance'] = (item['decision_seconds'] == metadata['outcome']['wall_seconds']
                            and digest(method_dir / (identifier + '_proposal') / 'summary.json') == metadata['summary_sha256'])
                if label == 'tpe':
                    trials = search.sampler.export()['trials']
                    checks['tpe_two_real_tells_restore'] = len(trials) == 5 and all(trials[3 + j]['S'] == items[j]['S'] for j in range(len(items)))
                reports.append({'label': label, 'job': job['job_id'], 'attempts': items,
                                'training_seconds_with_shared': row['search']['training_seconds'], 'decision_seconds': row['search']['decision_seconds']})
    checks['six_unique_methods'] = sorted(r['label'] for r in reports) == sorted(cfg['methods'])
    checks['identical_initial_messages_all_six'] = len(first_messages) == 6 and len(set(first_messages)) == 1
    checks['identical_shared_history_all_six'] = len(initial_histories) == 6 and all(h == initial_histories[0] for h in initial_histories)
    checks['identical_normalization'] = len(norms) == 1
    result = {'status': 'SEQUENTIAL_V3_PREFLIGHT_QC_PASSED' if all(checks.values()) else 'SEQUENTIAL_V3_PREFLIGHT_QC_NOT_PASSED',
              'checks': checks, 'system_prompt_sha256': SYSTEM_SHA256, 'freeze_sha256': digest(a.freeze), 'methods': reports,
              'allocation': allocation, 'test_evaluated': False, 'main_authorized': False,
              'warning': 'Two epochs, two adaptive attempts and one search repetition. Do not infer model-family ranking or comparative LLM utility.'}
    save(a.output / 'summary.json', result)
    lines = ['ŠTYRI LOKÁLNE LLM A DVA NADVÄZUJÚCE POKUSY', result['status'],
             'Rovnaký prompt v3, prvý vstup, rozpočet a tri spoločné štarty. Druhý návrh vidí skutočný prvý výsledok.',
             'Random/TPE používa rovnaké shared dáta a seedy. Žiadny test ani hlavný experiment.']
    for row in reports: lines.append(f"{row['label']}: completed={sum(i['status']=='complete' for i in row['attempts'])}, decision={row['decision_seconds']:.2f}s")
    lines.append(f"Fyzický ledger vrátane1s/job: {allocation['charged_or_reserved_gpu_seconds']/3600:.6f}GPU-h.")
    lines.append(result['warning']); (a.output / 'vysledok.txt').write_text('\n'.join(lines) + '\n')
    print(result['status'], 'checks', len(checks), flush=True); return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); raise SystemExit(0 if run(p.parse_args()) else 1)

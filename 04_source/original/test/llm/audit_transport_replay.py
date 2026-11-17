"""CPU-only reparse all recorded v3 outputs; never change trials or their costs."""
import argparse
from pathlib import Path
from configurator_contract_v3 import validate_feedback, parse_reply as strict_parse
from configurator_transport_v1 import parse_reply, POLICY
from preflight_support import read, save, digest


def run(a):
    submitted = read(a.freeze.parent / 'submitted_jobs.json'); rows = []
    for job in submitted['jobs']:
        if job['label'] == 'controls': continue
        folder = a.root / 'results/llm_search_prepare' / ('sequential_' + job['job_id']) / job['label']
        for identifier in ('a03', 'a04'):
            feedback_path = folder / (identifier + '_feedback.json'); feedback = read(feedback_path); evidence = validate_feedback(feedback)
            proposal_path = folder / (identifier + '_proposal') / 'summary.json'; proposal = read(proposal_path)
            for attempt in proposal['attempts']:
                raw = attempt['raw_output']; old, old_errors = strict_parse(raw, evidence)
                new, new_errors, transport = parse_reply(raw, evidence)
                rows.append({'model': job['label'], 'id': identifier, 'repair': attempt['repair'],
                             'source_summary_sha256': digest(proposal_path), 'source_feedback_sha256': digest(feedback_path),
                             'old_errors': old_errors, 'new_errors': new_errors, 'payload': new, 'transport': transport,
                             'already_valid_payload_unchanged': old == new if not old_errors else None})
    unchanged = [r['already_valid_payload_unchanged'] for r in rows if r['already_valid_payload_unchanged'] is not None]
    checks = {'four_families_present': len({r['model'] for r in rows}) == 4,
              'all_recorded_payloads_semantically_valid': all(not r['new_errors'] for r in rows),
              'already_valid_payloads_unchanged': bool(unchanged) and all(unchanged),
              'newly_accepted_only_complete_fence': all(r['transport']['single_fence_removed'] and r['old_errors'] == ['strict_json_required']
                    for r in rows if r['old_errors'] and not r['new_errors'])}
    result = {'status': 'UNIFORM_TRANSPORT_REPLAY_PASSED' if all(checks.values()) else 'UNIFORM_TRANSPORT_REPLAY_NOT_PASSED',
              'policy': POLICY, 'checks': checks, 'rows': rows, 'recorded_response_count': len(rows),
              'historical_trials_edited': False, 'new_model_generations': 0, 'new_trainings': 0,
              'warning': 'Offline parser robustness check, not a rerun or replacement of the failed Ministral study. Two real adaptive Ministral trainings remain unverified under this common transport policy.'}
    save(a.output, result); print(result['status'], len(rows)); return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); raise SystemExit(0 if run(p.parse_args()) else 1)

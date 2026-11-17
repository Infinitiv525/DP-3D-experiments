"""CPU fixtures only; no authorization of any real job or dataset access."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from main_protocol import (read, save, identity, digest, validate_manifest, shared_plan, search_plan,
                           winner, final_plan, training_seed, training_config, require_authorization)


MANIFEST = Path(__file__).resolve().parents[1] / 'config/flowbench_main_candidate_20261021.json'


def synthetic_searches(manifest, all_manual=False):
    starts = shared_plan(manifest); records = []
    for number, spec in enumerate(search_plan(manifest)):
        initial = [dict(number=r['number'], seed=r['seed'], parameters=r['parameters'], status='complete',
                        S=.1 if all_manual and r['number'] == 0 else .4 + r['number'] * .01)
                   for r in starts if r['repeat'] == spec['repeat']]
        parameters = dict(learning_rate=.000101 + number * .00001, weight_decay=.0001, pressure_weight=1, divergence_weight=0)
        initial.append(dict(number=3, seed=training_seed(manifest, spec['repeat'], 3), parameters=parameters, status='complete', S=.2))
        for attempt in range(4, 12):
            initial.append(dict(number=attempt, seed=training_seed(manifest, spec['repeat'], attempt), parameters=parameters, status='complete', S=.9))
        records.append(dict(method=spec['method'], repeat=spec['repeat'], phase='main_search', epochs=20,
                            manifest_sha256=identity(manifest), test_evaluated=False, trials=initial,
                            status='MAIN_SEARCH_COMPLETE', stop_reason='attempt_cap', training_seconds_spent=7200))
    return records


class MainProtocolTests(unittest.TestCase):
    def setUp(self): self.manifest = read(MANIFEST)

    def test_candidate_cannot_execute(self):
        with self.assertRaises(RuntimeError): require_authorization(MANIFEST, Path('MUST_NOT_READ'), None)

    def test_nine_fresh_shared_starts(self):
        plan = shared_plan(self.manifest)
        self.assertEqual(len(plan), 9); self.assertEqual(len({p['seed'] for p in plan}), 9)
        self.assertTrue(all(p['epochs'] == 20 and p['fresh_initialization'] for p in plan))
        self.assertEqual(plan, shared_plan(self.manifest))

    def test_eighteen_searches_and_common_starts(self):
        plan = search_plan(self.manifest); self.assertEqual(len(plan), 18)
        for repeat in range(3):
            group = [r for r in plan if r['repeat'] == repeat]
            self.assertEqual(len(group), 6); self.assertEqual(len({tuple(r['shared_ids']) for r in group}), 1)
            self.assertTrue(all(r['sampler_seed'] == self.manifest['search_repeat_seeds'][repeat] + 1000000 for r in group))

    def test_seed_policies_disjoint(self):
        seeds = {training_seed(self.manifest, r, j) for r in range(3) for j in range(12)}
        self.assertEqual(len(seeds), 36); self.assertFalse(seeds.intersection(self.manifest['final_training_seeds']))
        for args in ((True, 1), (3, 0), (0, 12)):
            with self.assertRaises(ValueError): training_seed(self.manifest, *args)

    def test_95_references_and95_distinct_trainings(self):
        plan = final_plan(self.manifest, synthetic_searches(self.manifest))
        self.assertEqual(plan['reference_count'], 95); self.assertEqual(plan['physical_training_count'], 95)
        self.assertEqual(sum(r['method'] == 'manual' for r in plan['references']), 5)
        self.assertTrue(all(t['fresh_initialization'] and t['initial_checkpoint'] is None for t in plan['physical_trainings']))

    def test_exact_final_duplicates_only_five_trainings(self):
        plan = final_plan(self.manifest, synthetic_searches(self.manifest, all_manual=True))
        self.assertEqual(plan['reference_count'], 95); self.assertEqual(plan['physical_training_count'], 5)
        self.assertEqual(plan['saved_duplicate_trainings'], 90)
        self.assertTrue(all(len(t['references']) == 19 for t in plan['physical_trainings']))

    def test_different_seeds_never_deduplicate(self):
        plan = final_plan(self.manifest, synthetic_searches(self.manifest, all_manual=True))
        self.assertEqual({r['seed'] for r in plan['physical_trainings']}, set(self.manifest['final_training_seeds']))

    def test_equal_numeric_categories_canonicalize_identically(self):
        studies = synthetic_searches(self.manifest, all_manual=True)
        studies[0]['trials'][0]['parameters'] = dict(self.manifest['manual'], pressure_weight=1.0, divergence_weight=0.0)
        self.assertEqual(final_plan(self.manifest, studies)['physical_training_count'], 5)

    def test_winner_earlier_attempt_on_tie(self):
        study = synthetic_searches(self.manifest)[0]; study['trials'][3]['S'] = study['trials'][0]['S']
        self.assertEqual(winner(self.manifest, study)['number'], 0)

    def test_failed_and_partial_trial_not_winner(self):
        study = synthetic_searches(self.manifest)[0]; study['trials'][3].update(status='timeout', S=None)
        self.assertEqual(winner(self.manifest, study)['number'], 0)
        study['trials'][3]['S'] = .01
        with self.assertRaises(ValueError): winner(self.manifest, study)

    def test_pilot_or_test_results_never_promoted(self):
        for changes in ({'phase': 'technical_pilot'}, {'epochs': 2}, {'test_evaluated': True}, {'manifest_sha256': 'wrong'}):
            study = dict(synthetic_searches(self.manifest)[0], **changes)
            with self.assertRaises(ValueError): winner(self.manifest, study)

    def test_all18studies_mandatory(self):
        studies = synthetic_searches(self.manifest)
        for invalid in (studies[:-1], studies[:-1] + [studies[0]]):
            with self.assertRaises(ValueError): final_plan(self.manifest, invalid)

    def test_premature_search_stop_never_finalized(self):
        for reason in ('attempt_cap', 'training_time_cap', 'global_abort', 'decision_time_cap'):
            study = synthetic_searches(self.manifest)[0]
            study.update(trials=study['trials'][:4], stop_reason=reason, training_seconds_spent=2400, decision_seconds_spent=0)
            with self.assertRaises(ValueError): winner(self.manifest, study)
        study = synthetic_searches(self.manifest)[0]; study.update(trials=study['trials'][:4], stop_reason='training_time_cap', training_seconds_spent=7199)
        self.assertEqual(winner(self.manifest, study)['number'], 3)

    def test_shared_parameters_and_seeds_must_match(self):
        for mutation in ('parameters', 'seed', 'failed'):
            study = synthetic_searches(self.manifest)[0]
            if mutation == 'parameters': study['trials'][0]['parameters'] = study['trials'][3]['parameters']
            if mutation == 'seed': study['trials'][3]['seed'] += 1
            if mutation == 'failed': study['trials'][0].update(status='failed', S=None)
            with self.assertRaises(ValueError): winner(self.manifest, study)

    def test_invalid_completed_scores_refused(self):
        for value in (float('nan'), float('inf'), True, -.1):
            study = synthetic_searches(self.manifest)[0]; study['trials'][3]['S'] = value
            with self.assertRaises(ValueError): winner(self.manifest, study)

    def test_fresh_training_config_only(self):
        job = shared_plan(self.manifest)[0]; cfg = training_config(self.manifest, job)
        self.assertFalse(cfg['pilot_only']); self.assertEqual(cfg['training']['epochs'], 20)
        for changes in ({'epochs': 2}, {'initial_checkpoint': 'pilot.pt'}, {'fresh_initialization': False}, {'stage': 'test'}):
            with self.assertRaises(ValueError): training_config(self.manifest, dict(job, **changes))
        for seed in (True, 410001, 12345):
            with self.assertRaises(ValueError): training_config(self.manifest, dict(job, seed=seed))

    def test_guard_runs_before_missing_config_or_torch_import(self):
        path = Path(__file__).resolve().parents[1] / 'training/train_flowbench_main.py'
        result = subprocess.run([sys.executable, '-B', str(path), '--root', 'UNUSED', '--manifest', str(MANIFEST),
            '--freeze', 'MUST_NOT_READ', '--config', 'MUST_NOT_READ', '--data', 'MUST_NOT_READ',
            '--normalization', 'MUST_NOT_READ', '--output', 'MUST_NOT_WRITE'], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2); self.assertIn('Main workflow not yet released', result.stdout)
        self.assertNotIn('No such file', result.stderr)

    def test_synthetic_authorization_identity_checks(self):
        # Ephemeral mock approval for gate unit testing, not real operator consent.
        with tempfile.TemporaryDirectory(prefix='main-gate-unit-', dir=Path(__file__).parent) as name:
            folder = Path(name); m = folder / 'manifest.json'; f = folder / 'freeze.json'; a = folder / 'mock_approval.json'
            manifest = copy.deepcopy(self.manifest); manifest.update(execution_ready=True, not_yet_released=[])
            save(m, manifest); save(f, dict(status='MAIN_SOURCE_FREEZE_PASSED', manifest_file_sha256=digest(m), source_files={}))
            with self.assertRaises(PermissionError): require_authorization(m, f, None)
            approval = dict(approved=True, manifest_file_sha256=digest(m), freeze_file_sha256=digest(f), gpu_hours_cap=70,
                            user_consent_reference='SYNTHETIC UNIT FIXTURE ONLY; never used outside this temporary folder')
            save(a, approval); self.assertEqual(require_authorization(m, f, a)[0], manifest)
            for patch in ({'approved': False}, {'gpu_hours_cap': 71}, {'manifest_file_sha256': 'other'}, {'user_consent_reference': ''}):
                save(a, dict(approval, **patch))
                with self.assertRaises(PermissionError): require_authorization(m, f, a)

    def test_changed_frozen_source_blocks_execution(self):
        with tempfile.TemporaryDirectory(prefix='freeze-gate-unit-', dir=Path(__file__).parent) as name:
            folder = Path(name); m = folder / 'manifest.json'; f = folder / 'freeze.json'; a = folder / 'mock_approval.json'
            source = folder / 'dummy_source.txt'; source.write_text('synthetic first version')
            manifest = copy.deepcopy(self.manifest); manifest.update(execution_ready=True, not_yet_released=[])
            save(m, manifest); save(f, dict(status='MAIN_SOURCE_FREEZE_PASSED', manifest_file_sha256=digest(m), source_files={str(source): digest(source)}))
            save(a, dict(approved=True, manifest_file_sha256=digest(m), freeze_file_sha256=digest(f), gpu_hours_cap=70,
                         user_consent_reference='SYNTHETIC TEMPORARY UNIT FIXTURE ONLY'))
            require_authorization(m, f, a)
            source.write_text('synthetic changed version')
            with self.assertRaises(ValueError): require_authorization(m, f, a)


if __name__ == '__main__': unittest.main(verbosity=2)

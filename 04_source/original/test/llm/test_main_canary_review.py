"""Small synthetic checks for the post-canary CPU gate."""
import copy
from pathlib import Path
import unittest
from main_protocol import read, identity, shared_plan, training_config
from review_main_canary import report_checks


class CanaryReviewTests(unittest.TestCase):
    def fixture(self):
        m = read(Path(__file__).resolve().parents[1] / 'config/flowbench_main_release_20261021.json')
        job = shared_plan(m)[0]; config = training_config(m, job)
        split = dict(indices=dict(train=list(range(700)), validation=list(range(700, 850)), test=list(range(850, 1000))),
                     geometry_groups=[f'g{i // 10}' for i in range(1000)])
        rows = [dict(sample=i, geometry=split['geometry_groups'][i], velocity_error=.2, pressure_error=.4,
                     velocity_below_epsilon=False, pressure_below_epsilon=False) for i in range(700, 850)]
        score = dict(S=.3, velocity_error=.2, pressure_error=.4, sample_count=150, geometry_count=15)
        summary = dict(config=config, status='FLOWBENCH_MAIN_TRIAL_COMPLETE', pilot_only=False, phase='main_training',
                       stage='shared', test_evaluated=False, best_epoch=1, best_validation=score,
                       history=[dict(epoch=i, validation=score, train_composite_loss=.3) for i in range(1, 21)],
                       training_indices=split['indices']['train'], validation_indices=split['indices']['validation'], gpu='NVIDIA H200')
        task = dict(spec=dict(job=job, manifest_sha256=identity(m)), status='complete', seconds=400)
        return m, task, config, summary, rows, split

    def test_complete_fixture_passes(self):
        checks, measured = report_checks(*self.fixture()); self.assertTrue(all(checks.values()))

    def test_pilot_or_incomplete_history_fails(self):
        args = list(self.fixture()); args[3]['pilot_only'] = True; args[3]['history'] = args[3]['history'][:2]
        checks, measured = report_checks(*args); self.assertFalse(checks['fresh_complete_main']); self.assertFalse(checks['actual20epochs'])

    def test_test_sample_leakage_rejected(self):
        args = list(self.fixture()); args[4][0]['sample'] = 900
        checks, measured = report_checks(*args); self.assertFalse(checks['all150saved_rows_validation_only'])


if __name__ == '__main__': unittest.main(verbosity=2)

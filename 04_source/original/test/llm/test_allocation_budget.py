"""Synthetic CPU checks of the global physical-allocation cap."""
import json
from pathlib import Path
import tempfile
import unittest
from allocation_budget import PhysicalBudget, wall_seconds


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='allocation-unit-', dir=Path(__file__).parent)
        self.path = Path(self.tmp.name)
        self.opts = dict(gpu_seconds_limit=960, identity={'user': 'unit', 'account': 'fixture'})
    def tearDown(self): self.tmp.cleanup()
    def scheduler(self, state='COMPLETED', runtime='00:01:20', gpus=1, requeue=0):
        return f'JobId=123 UserId=unit(1) Account=fixture JobState={state} Requeue={requeue} AllocTRES=cpu=4,gres/gpu={gpus} RunTime={runtime} TimeLimit=00:05:00'

    def test_two_reservations_and_global_limit(self):
        with PhysicalBudget(self.path, **self.opts) as ledger:
            ledger.reserve('latency', 1, 300); ledger.reserve('training', 1, 600)
            self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 902)
            with self.assertRaises(RuntimeError): ledger.reserve('extra', 1, 300)
        with PhysicalBudget(self.path, **self.opts) as ledger:
            self.assertEqual(ledger.report()['available_gpu_seconds'], 58)

    def test_uncertain_submit_stays_reserved_and_duplicate_blocked(self):
        with PhysicalBudget(self.path, **self.opts) as ledger: ledger.reserve('unknown', 1, 300)
        with PhysicalBudget(self.path, **self.opts) as ledger:
            self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 301)
            with self.assertRaises(RuntimeError): ledger.reserve('unknown', 1, 300)

    def test_actual_scheduler_cost_idempotent_includes_rounding(self):
        with PhysicalBudget(self.path, **self.opts) as ledger:
            ledger.reserve('a', 1, 300); ledger.attach('a', '123')
            ledger.settle('a', self.scheduler()); ledger.settle('a', self.scheduler())
            self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 81)

    def test_active_requeue_and_wrong_identity_cannot_refund(self):
        with PhysicalBudget(self.path, **self.opts) as ledger:
            ledger.reserve('a', 1, 300); ledger.attach('a', '123')
            for text in (self.scheduler(state='RUNNING'), self.scheduler(requeue=1),
                         self.scheduler(gpus=2), self.scheduler().replace('unit(1)', 'other(2)')):
                with self.assertRaises((RuntimeError, ValueError)): ledger.settle('a', text)
                self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 301)

    def test_overrun_visible_and_blocks_next_allocation(self):
        with PhysicalBudget(self.path, **self.opts) as ledger:
            ledger.reserve('a', 1, 300); ledger.attach('a', '123')
            ledger.settle('a', self.scheduler(runtime='00:20:00'))
            self.assertEqual(ledger.report()['overrun_gpu_seconds'], 241)
            with self.assertRaises(RuntimeError): ledger.reserve('b', 1, 1)

    def test_lock_identity_and_corruption(self):
        with PhysicalBudget(self.path, **self.opts):
            with self.assertRaises(BlockingIOError): PhysicalBudget(self.path, **self.opts)
        with self.assertRaises(ValueError): PhysicalBudget(self.path, **dict(self.opts, gpu_seconds_limit=1000))
        path = self.path / 'allocations.json'; content = json.loads(path.read_text())
        content['payload']['revision'] += 1; path.write_text(json.dumps(content))
        with self.assertRaises(ValueError): PhysicalBudget(self.path, **self.opts)

    def test_invalid_limits_and_durations(self):
        for value in (True, float('nan'), -1):
            with self.assertRaises(ValueError): PhysicalBudget(self.path, **dict(self.opts, gpu_seconds_limit=value))
        self.assertEqual(wall_seconds('1-02:03:04'), 93784)
        with self.assertRaises(ValueError): wall_seconds('UNLIMITED')


if __name__ == '__main__': unittest.main(verbosity=2)

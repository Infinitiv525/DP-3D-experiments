"""Synthetic accounting tests; never supplies fabricated scheduler measurements."""
import copy
from pathlib import Path
import tempfile
import unittest
from allocation_budget import PhysicalBudget
from close_shared_accounting_v1 import close_at_full_cap


class ConservativeAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.ledger = PhysicalBudget(Path(self.temp.name), gpu_seconds_limit=10000,
                                    identity=dict(user='lutoke823', account='perun2601449'))
        self.addCleanup(self.ledger.close)
        for i in range(2): self.ledger.reserve(f'shared:{i}', 1, 900); self.ledger.attach(f'shared:{i}', str(100 + i))
        self.labels = ['shared:0', 'shared:1']
        self.absence = dict(query_succeeded=True, user='lutoke823', active_job_ids=[], cluster='synthetic_cluster')
        self.proofs = {f'shared:{i}': dict(job_id=str(100 + i), complete_verified=True, historical_record_unavailable=True,
                                         cluster='synthetic_cluster') for i in range(2)}

    def test_full_cap_no_refund_and_idempotent(self):
        self.assertEqual(close_at_full_cap(self.ledger, self.labels, self.absence, self.proofs), self.labels)
        self.assertEqual(self.ledger.report()['charged_or_reserved_gpu_seconds'], 1802)
        self.assertEqual(close_at_full_cap(self.ledger, self.labels, self.absence, self.proofs), [])
        for row in self.ledger.state['allocations']:
            self.assertIsNone(row['actual_allocated_gpu_seconds']); self.assertNotIn('reported_runtime_seconds', row)
            self.assertNotIn('scheduler_fields', row); self.assertEqual(row['refund_gpu_seconds'], 0)

    def test_live_owner_blocks_entire_transaction(self):
        self.absence['active_job_ids'] = ['101']
        with self.assertRaises(RuntimeError): close_at_full_cap(self.ledger, self.labels, self.absence, self.proofs)
        self.assertTrue(all(r['status'] == 'submitted_on_hold' for r in self.ledger.state['allocations']))

    def test_unavailable_scheduler_not_absence(self):
        self.absence['query_succeeded'] = False
        with self.assertRaises(ValueError): close_at_full_cap(self.ledger, self.labels, self.absence, self.proofs)

    def test_bad_or_incomplete_evidence_blocks_all(self):
        for patch in (dict(job_id='OTHER'), dict(complete_verified=False), dict(historical_record_unavailable=False), dict(cluster='OTHER')):
            proofs = copy.deepcopy(self.proofs); proofs['shared:1'].update(patch)
            with self.assertRaises(ValueError): close_at_full_cap(self.ledger, self.labels, self.absence, proofs)
            self.assertTrue(all(r['status'] == 'submitted_on_hold' for r in self.ledger.state['allocations']))

    def test_changed_reservation_is_not_refunded_or_overwritten(self):
        self.ledger.state['allocations'][1]['charged_gpu_seconds'] = 800
        with self.assertRaises(RuntimeError): close_at_full_cap(self.ledger, self.labels, self.absence, self.proofs)
        self.assertEqual(self.ledger.state['allocations'][0]['status'], 'submitted_on_hold')

    def test_measured_settlement_is_preserved(self):
        row = self.ledger.state['allocations'][0]
        row.update(status='settled', charged_gpu_seconds=551, reported_runtime_seconds=550)
        before = copy.deepcopy(row)
        close_at_full_cap(self.ledger, self.labels, self.absence, self.proofs)
        self.assertEqual(self.ledger.state['allocations'][0], before)
        self.assertEqual(self.ledger.report()['charged_or_reserved_gpu_seconds'], 1452)


if __name__ == '__main__': unittest.main(verbosity=2)

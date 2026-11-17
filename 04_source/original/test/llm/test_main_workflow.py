"""CPU-only end-to-end fixtures. Synthetic scores are NOT research results."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from main_protocol import read, shared_plan, search_plan, final_plan, identity
from main_workflow import fixed_task, run_search, final_catalog, atomic
from main_execution import load_task, stage_jobs, ProcessBackend, verify_runtime_freeze
from main_submit import submit_batch, reconcile
from allocation_budget import PhysicalBudget
from main_protocol import digest, training_config

MANIFEST = Path(__file__).resolve().parents[1] / 'config/flowbench_main_candidate_20261021.json'


class FakeBackend:
    def __init__(self): self.trained = []; self.feedback = []; self.fail = None; self.invalid = False; self.seconds = 10
    def train(self, job, config, folder, cap):
        self.trained.append(copy.deepcopy(job))
        if self.fail: raise RuntimeError('Injected crash before commit')
        self.last_config = config
        score = .5 if job['stage'] == 'shared' else .2
        return dict(status='complete', seconds=min(self.seconds, cap),
                    measurement=dict(S=score, velocity_error=score, pressure_error=score, divergence=.1),
                    checkpoint_path='SYNTHETIC_NOT_A_CHECKPOINT', checkpoint_sha256='SYNTHETIC')
    def propose(self, method, feedback, folder, cap):
        self.feedback.append(copy.deepcopy(feedback))
        if self.invalid: return dict(status='invalid_reply', seconds=1)
        return dict(status='complete', seconds=min(1, cap), parameters=dict(learning_rate=.001, weight_decay=.001, pressure_weight=1, divergence_weight=0))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.m = read(MANIFEST); self.backend = FakeBackend()
        self.provenance = dict(freeze_sha256='SYNTHETIC', authorization_sha256='NO_REAL_APPROVAL')
    def shared(self, repeat=0):
        return [fixed_task(self.m, j, self.root / 'shared' / j['id'], self.backend, self.provenance, 'process')
                for j in shared_plan(self.m) if j['repeat'] == repeat]
    def search(self, method='qwen3_8b', repeat=0, records=None):
        return run_search(self.m, method, repeat, records or self.shared(repeat), self.root / 'search' / f'{method}_r{repeat}',
                          self.backend, self.provenance, 'process')

    def test_all_stages_nine_eighteen_95_aliases(self):
        records = {r: self.shared(r) for r in range(3)}
        results = [self.search(s['method'], s['repeat'], records[s['repeat']]) for s in search_plan(self.m)]
        self.assertEqual(len(self.backend.trained), 9 + 18 * 9)
        self.assertTrue(all(len(s['trials']) == 12 for s in results))
        plan = final_plan(self.m, results)
        finals = {j['id']: fixed_task(self.m, j, self.root / 'final' / j['id'], self.backend, self.provenance, 'process')
                  for j in plan['physical_trainings']}
        catalog = final_catalog(self.m, plan, finals)
        self.assertEqual(catalog['reference_count'], 95); self.assertFalse(catalog['test_evaluated'])
        self.assertEqual(len(self.backend.trained), 171 + plan['physical_training_count'])
        self.assertTrue(all(j['seed'] >= 410001 for j in self.backend.trained[171:]))

    def test_completed_search_resume_no_children_or_new_attempt(self):
        records = self.shared(); first = self.search(records=records); count = len(self.backend.trained)
        second = self.search(records=records)
        self.assertEqual(first, second); self.assertEqual(len(self.backend.trained), count)

    def test_completed_fixed_task_idempotent(self):
        first = self.shared(); count = len(self.backend.trained)
        self.assertEqual(first, self.shared()); self.assertEqual(len(self.backend.trained), count)

    def test_shared_cost_charged_identically_all_six(self):
        records = self.shared()
        results = [self.search(m, records=records) for m in self.m['methods']]
        self.assertTrue(all(r['training_seconds_spent'] == 120 for r in results))
        self.assertEqual(len(self.backend.trained), 3 + 6 * 9)

    def test_feedback_has_real_committed_previous_attempt(self):
        self.search(); feedback = self.backend.feedback
        self.assertEqual(len(feedback), 9)
        self.assertEqual(feedback[0]['phase'], 'main_search'); self.assertEqual(feedback[0]['epochs_per_trial'], 20)
        self.assertEqual(len(feedback[0]['history']), 3); self.assertEqual(len(feedback[1]['history']), 4)
        self.assertEqual(feedback[1]['history'][-1]['id'], 'a03'); self.assertEqual(feedback[1]['history'][-1]['S'], .2)

    def test_invalid_decision_counts_attempt_no_training(self):
        records = self.shared(); self.backend.invalid = True
        result = self.search(records=records)
        self.assertEqual(len(self.backend.trained), 3); self.assertEqual(len(result['trials']), 12)
        self.assertEqual(result['decision_seconds_spent'], 9)
        self.assertTrue(all(t['status'] == 'invalid_reply' and t['S'] is None for t in result['trials'][3:]))

    def test_training_budget_stops_without_extra_attempt(self):
        self.backend.seconds = 600; records = self.shared(); self.backend.seconds = 840
        result = self.search('random', records=records)
        self.assertEqual(result['training_seconds_spent'], 7200)
        self.assertEqual(result['stop_reason'], 'training_time_cap')
        self.assertLessEqual(len(result['trials']), 12)

    def test_fixed_crash_retains_full_reservation_and_never_retries(self):
        job = shared_plan(self.m)[0]; folder = self.root / 'fixed'; self.backend.fail = True
        with self.assertRaises(RuntimeError): fixed_task(self.m, job, folder, self.backend, self.provenance, 'process')
        self.assertEqual(load_task(folder)['seconds'], 840)
        with patch('main_workflow.owner_active', return_value=False):
            with self.assertRaises(RuntimeError): fixed_task(self.m, job, folder, self.backend, self.provenance, 'process')
        self.assertEqual(load_task(folder)['status'], 'interrupted_reserved_cost'); self.assertEqual(len(self.backend.trained), 1)

    def test_search_crash_consumes_attempt_and_reserved_cost_on_recovery(self):
        records = self.shared(); self.backend.fail = True
        with self.assertRaises(RuntimeError): self.search('random', records=records)
        self.backend.fail = False
        with patch('persistent_search.owner_active', return_value=False): result = self.search('random', records=records)
        self.assertEqual(result['trials'][3]['status'], 'interrupted_reserved_cost')
        self.assertEqual(len(result['trials']), 12); self.assertEqual(result['training_seconds_spent'], 30 + 840 + 8 * 10)

    def test_shared_checksum_or_identity_tamper_rejected(self):
        records = self.shared(); records[0]['spec']['manifest_sha256'] = 'other'
        with self.assertRaises(ValueError): self.search(records=records)
        path = self.root / 'shared' / 'r0_s00' / 'task.json'; value = read(path)
        value['payload']['seconds'] = 0; atomic(path, value)
        with self.assertRaises(ValueError): load_task(path.parent)

    def test_foreign_shared_freeze_not_reused(self):
        records = self.shared(); records[0]['spec']['provenance']['freeze_sha256'] = 'OTHER_FREEZE'
        with self.assertRaises(ValueError): self.search(records=records)

    def test_missing_final_training_not_sealed(self):
        from test_main_protocol import synthetic_searches
        plan = final_plan(self.m, synthetic_searches(self.m))
        with self.assertRaises(KeyError): final_catalog(self.m, plan, {})

    def test_incomplete_shared_blocks_stage(self):
        with self.assertRaises(FileNotFoundError): stage_jobs(self.m, 'search', self.root)

    def test_production_entry_candidate_denied_before_output_or_data(self):
        script = Path(__file__).with_name('main_submit.py')
        out = self.root / 'MUST_NOT_CREATE'
        result = subprocess.run([sys.executable, '-B', str(script), 'submit', '--root', str(self.root),
            '--manifest', str(MANIFEST), '--freeze', 'MUST_NOT_READ', '--authorization', 'NO_APPROVAL', '--output', str(out)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0); self.assertIn('not yet released', result.stderr)
        self.assertFalse(out.exists())

    def ledger(self, cap=10000):
        return PhysicalBudget(self.root / 'budget', gpu_seconds_limit=cap, identity=dict(user='lutoke823', account='perun2601449'))
    def jobs(self): return [dict(stage='final', label=str(i), seconds=900) for i in range(3)]

    def test_batch_reserves_and_attaches_everything_before_release(self):
        released = []; events = []
        with self.ledger() as ledger:
            def submit(job):
                self.assertEqual(len(ledger.state['allocations']), 3); events.append(job['label']); return str(100 + int(job['label']))
            def release(job):
                self.assertTrue(all(r['job_id'] is not None for r in ledger.state['allocations'])); released.append(job)
            submit_batch(ledger, self.jobs(), submit, release)
            self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 2703)
        self.assertEqual(released, ['100', '101', '102'])

    def test_budget_shortfall_submits_nothing(self):
        with self.ledger(2702) as ledger:
            with self.assertRaises(RuntimeError): submit_batch(ledger, self.jobs(), lambda j: self.fail('must not submit'), lambda j: self.fail('must not release'))
            self.assertEqual(ledger.state['allocations'], [])

    def test_partial_submission_keeps_held_jobs_and_full_reservations(self):
        released = []
        with self.ledger() as ledger:
            def submit(job):
                if job['label'] == '1': raise RuntimeError('Injected scheduler failure')
                return '101'
            with self.assertRaises(RuntimeError): submit_batch(ledger, self.jobs(), submit, released.append)
            self.assertFalse(released); self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 2703)
            self.assertEqual(ledger.state['allocations'][0]['job_id'], '101')
            with self.assertRaises(RuntimeError): submit_batch(ledger, self.jobs(), submit, released.append)

    def test_duplicate_job_id_rejected_before_any_release(self):
        released = []
        with self.ledger() as ledger:
            with self.assertRaises(RuntimeError): submit_batch(ledger, self.jobs(), lambda j: '101', released.append)
        self.assertFalse(released)

    def test_active_allocation_keeps_full_reservation(self):
        with self.ledger() as ledger:
            ledger.reserve('test', 1, 900); ledger.attach('test', '101')
            raw = f'JobId=101 UserId={os.getuid()}({os.getuid()}) Account=perun2601449 JobState=RUNNING'
            reconcile(ledger, lambda argv: raw)
            self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 901)

    def backend_args(self):
        a = SimpleNamespace(root=self.root, manifest=self.root / 'manifest.json', freeze=self.root / 'freeze.json',
                            authorization=self.root / 'approval.json')
        for path in (a.manifest, a.freeze, a.authorization): atomic(path, dict(synthetic=True, not_real_authorization=True))
        atomic(a.root / self.m['normalization_source_relative'], dict(synthetic=True))
        return a

    def test_real_training_adapter_command_and_result_validation(self):
        a = self.backend_args(); backend = ProcessBackend(a, self.m)
        job = shared_plan(self.m)[0]; config = training_config(self.m, job); folder = self.root / 'adapter'
        folder.mkdir(); calls = []
        def process(argv, cap, log):
            calls.append(argv); dest = folder / 'training'; dest.mkdir()
            atomic(dest / 'best_model.pt', dict(synthetic_not_real_checkpoint=True))
            atomic(dest / 'summary.json', dict(status='FLOWBENCH_MAIN_TRIAL_COMPLETE', test_evaluated=False, config=config,
                split_sha256=self.m['split_sha256'], normalization_sha256=digest(a.root / self.m['normalization_source_relative']),
                freeze_file_sha256=digest(a.freeze), authorization_file_sha256=digest(a.authorization),
                checkpoint_sha256=digest(dest / 'best_model.pt'), best_validation=dict(S=.2, velocity_error=.1, pressure_error=.3),
                predicted_divergence_rms_mean=.1))
            return dict(exit_code=0, timed_out=False, wall_seconds=2)
        with patch('main_execution.run_bounded_process', side_effect=process):
            result = backend.train(job, config, folder, 840)
        self.assertEqual(result['status'], 'complete')
        self.assertIn(str(self.root / 'test/training/train_flowbench_main.py'), calls[0])
        self.assertIn('830', calls[0]); self.assertIn('--authorization', calls[0])

    def test_real_training_adapter_rejects_false_success_provenance(self):
        a = self.backend_args(); backend = ProcessBackend(a, self.m); job = shared_plan(self.m)[0]
        folder = self.root / 'adapter'; folder.mkdir()
        def process(argv, cap, log):
            atomic(folder / 'training/summary.json', dict(status='FLOWBENCH_MAIN_TRIAL_COMPLETE', test_evaluated=True))
            return dict(exit_code=0, timed_out=False, wall_seconds=2)
        with patch('main_execution.run_bounded_process', side_effect=process):
            with self.assertRaises(ValueError): backend.train(job, training_config(self.m, job), folder, 840)

    def test_real_proposer_uses_common_transport_and_preserves_reason(self):
        a = self.backend_args(); cfg_path = Path(__file__).resolve().parents[1] / 'config/llm_configurator_v3_20261021.json'
        cfg = read(cfg_path); atomic(a.root / self.m['llm_config_relative'], cfg)
        backend = ProcessBackend(a, self.m); folder = self.root / 'proposal_adapter'; folder.mkdir()
        records = self.shared(); self.search(records=records); feedback = self.backend.feedback[0]
        atomic(folder / 'feedback.json', feedback)
        from configurator_contract_v3 import SYSTEM_SHA256, canonical, messages
        from configurator_transport_v1 import POLICY
        source = Path(__file__).with_name('configurator_transport_v1.py')
        # An inert temporary copy permits exact source-hash validation, not execution.
        atomic(a.root / 'test/llm/configurator_transport_v1.py', dict(synthetic_source=True))
        params = dict(self.m['manual'], reason='Synthetic public reason.', expected_effect='Synthetic effect.', risk='Synthetic risk.', evidence_ids=['r0_s00'])
        calls = []
        def process(argv, cap, log):
            calls.append(argv)
            atomic(folder / 'proposal/summary.json', dict(status='LOCAL_CONFIGURATOR_V3_COMPLETE', system_prompt_sha256=SYSTEM_SHA256,
                feedback_sha256=digest(folder / 'feedback.json'), phase='main_search', test_evaluated=False,
                model=cfg['models'][0], transport_policy=POLICY, generation=cfg['generation'], configuration=params,
                contract_version=3, transport_source_sha256=digest(a.root / 'test/llm/configurator_transport_v1.py'),
                initial_messages_sha256=__import__('hashlib').sha256(canonical(messages(feedback)).encode()).hexdigest()))
            return dict(exit_code=0, timed_out=False, wall_seconds=2)
        with patch('main_execution.run_bounded_process', side_effect=process): result = backend.propose('qwen3_8b', feedback, folder, 210)
        self.assertEqual(result['configuration_with_reason']['reason'], params['reason'])
        self.assertIn(str(self.root / 'test/llm/propose_configuration_transport_v1.py'), calls[0])

    def test_real_adapter_timeout_is_not_partial_success(self):
        a = self.backend_args(); job = shared_plan(self.m)[0]; folder = self.root / 'adapter'; folder.mkdir()
        with patch('main_execution.run_bounded_process', return_value=dict(exit_code=-15, timed_out=True, wall_seconds=840)):
            result = ProcessBackend(a, self.m).train(job, training_config(self.m, job), folder, 840)
        self.assertEqual(result['status'], 'timeout'); self.assertNotIn('measurement', result)

    def test_finished_allocation_settles_uid_and_retains_original_proof(self):
        with self.ledger() as ledger:
            ledger.reserve('test', 1, 900); ledger.attach('test', '101')
            raw = (f'JobId=101 UserId={os.getuid()}({os.getuid()}) Account=perun2601449 JobState=COMPLETED '
                   'Requeue=0 AllocTRES=cpu=4,gres/gpu=1 TimeLimit=00:15:00 RunTime=00:02:00')
            reconcile(ledger, lambda argv: raw)
            self.assertEqual(ledger.report()['charged_or_reserved_gpu_seconds'], 121)
            self.assertEqual(ledger.state['allocations'][0]['scheduler_uid_proof']['original_scheduler_text'], raw)

    def test_incomplete_runtime_or_input_freeze_rejected(self):
        with self.assertRaises(ValueError): verify_runtime_freeze(self.root, self.m, dict(source_files={}))

    def test_live_fixed_owner_cannot_be_recovered(self):
        job = shared_plan(self.m)[0]; folder = self.root / 'live'; self.backend.fail = True
        with self.assertRaises(RuntimeError): fixed_task(self.m, job, folder, self.backend, self.provenance, 'process')
        with self.assertRaisesRegex(RuntimeError, 'live job/process'):
            fixed_task(self.m, job, folder, self.backend, self.provenance, 'process')
        self.assertEqual(load_task(folder)['status'], 'running'); self.assertEqual(len(self.backend.trained), 1)

    def test_global_70h_ledger_includes_every_reserved_stage(self):
        with self.ledger(252000) as ledger:
            ledger.reserve('already_spent', 1, 251099)
            with self.assertRaises(RuntimeError): submit_batch(ledger, self.jobs(), lambda j: self.fail('must not submit'), lambda j: None)
            self.assertEqual(len(ledger.state['allocations']), 1)


if __name__ == '__main__': unittest.main(verbosity=2)

"""CPU fixtures for balanced scheduling; no real submissions or model data."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from allocation_budget import PhysicalBudget
from main_protocol import read, digest
from main_workflow import atomic
from final_window_engine_v1 import reserve_all, advance, release_window
from final_window_controller_v1 import job_id, locked_plan


class FakeBackend:
    def __init__(self, ledger, expected=70):
        self.ledger=ledger; self.expected=expected; self.queue=1; self.counter=1000
        self.events=[]; self.finished=set(); self.fail=None
    def next(self, kind):
        if self.fail==kind: raise RuntimeError('Injected '+kind+' failure')
        self.queue+=1
        if self.queue>50: raise AssertionError('Scheduler limit exceeded')
        self.counter+=1; self.events.append((kind,str(self.counter))); return str(self.counter)
    def queue_count(self): return self.queue
    def submit_gpu(self, job):
        if len(self.ledger.state['allocations'])!=self.expected: raise AssertionError('Whole phase not reserved first')
        return self.next('gpu')
    def submit_capture(self, job, gpu): return self.next('capture')
    def submit_followup(self, dependencies): return self.next('followup')
    def release(self, gpu):
        if self.fail=='release': raise RuntimeError('Injected release failure')
        self.events.append(('release',gpu))
    def review(self, job):
        if job['label'] not in self.finished: raise RuntimeError('Incomplete application')
    def finish_window(self, state, jobs, good=True):
        w=state['windows'][-1]
        for job in jobs[w['start']:w['end']]:
            r=next(x for x in self.ledger.state['allocations'] if x['label']=='final:'+job['label'])
            r.update(status='settled',charged_gpu_seconds=501)
            if good: self.finished.add(job['label'])
        self.ledger.save(); self.queue=1


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.ledger=PhysicalBudget(self.root/'budget',gpu_seconds_limit=100000,identity=dict(user='test',account='test'))
        self.addCleanup(self.ledger.close)
        self.jobs=[dict(stage='final',label=f'job{i:02}',seconds=900) for i in range(70)]
        self.state=dict(phase='new');self.backend=FakeBackend(self.ledger);self.commits=0
    def save(self): self.commits+=1
    def advance(self): return advance(self.state,self.jobs,self.ledger,self.backend,self.save)
    def dispatch(self):
        ids=self.advance();release_window(self.state,ids,self.backend,self.save);return ids

    def test_all70_reserved_before_first_release(self):
        ids=self.advance()
        self.assertEqual(len(self.ledger.state['allocations']),70)
        self.assertEqual(self.ledger.report()['charged_or_reserved_gpu_seconds'],63070)
        self.assertEqual(len(ids),20);self.assertFalse(any(x[0]=='release' for x in self.backend.events))
        self.assertEqual(len(self.state['windows'][0]['capture_ids']),20)
        self.assertIsNotNone(self.state['windows'][0]['next_controller_id'])

    def test_full70_run_20_20_20_10_no_duplicates(self):
        windows=[]
        while self.state['phase']!='ready_to_seal':
            ids=self.dispatch()
            if ids:
                windows.append(len(ids));self.backend.finish_window(self.state,self.jobs)
        self.assertEqual(windows,[20,20,20,10]);self.assertEqual(self.state['next_index'],70)
        self.assertEqual(len([e for e in self.backend.events if e[0]=='gpu']),70)
        self.assertEqual(len([e for e in self.backend.events if e[0]=='release']),70)
        self.assertTrue(all(w['review']=='ALL_COMPLETE_AND_VERIFIED' for w in self.state['windows']))
        before=copy.deepcopy(self.backend.events);self.assertEqual(self.advance(),[]);self.assertEqual(before,self.backend.events)

    def test_whole_phase_budget_shortfall_no_submission(self):
        self.ledger.spec['gpu_seconds_limit']=63069
        with self.assertRaises(RuntimeError): self.advance()
        self.assertFalse(self.backend.events);self.assertFalse(self.ledger.state['allocations'])

    def test_existing_reservation_never_reserved_again(self):
        self.ledger.reserve('final:job00',1,900)
        with self.assertRaises(RuntimeError): reserve_all(self.ledger,self.jobs)
        self.assertEqual(len(self.ledger.state['allocations']),1)

    def test_pending_capture_blocks_next_window(self):
        self.dispatch();count=len(self.backend.events)
        with self.assertRaises(RuntimeError): self.advance()
        self.assertEqual(len(self.backend.events),count)

    def test_failed_training_blocks_next_even_when_accounting_settled(self):
        self.dispatch();self.backend.finish_window(self.state,self.jobs,good=False)
        count=len(self.backend.events)
        with self.assertRaises(RuntimeError): self.advance()
        self.assertEqual(len(self.backend.events),count)

    def test_capture_submission_failure_keeps_every_gpu_held(self):
        self.backend.fail='capture'
        with self.assertRaises(RuntimeError): self.advance()
        self.assertEqual(self.state['phase'],'submitting')
        self.assertFalse(any(x[0]=='release' for x in self.backend.events))
        count=len(self.backend.events);self.backend.fail=None
        with self.assertRaises(RuntimeError): self.advance()
        self.assertEqual(len(self.backend.events),count)

    def test_gpu_submission_failure_not_refunded_or_retried(self):
        self.backend.fail='gpu'
        with self.assertRaises(RuntimeError): self.advance()
        self.assertEqual(self.ledger.report()['charged_or_reserved_gpu_seconds'],63070)
        self.backend.fail=None
        with self.assertRaises(RuntimeError): self.advance()
        self.assertFalse(self.backend.events)

    def test_followup_failure_keeps_entire_window_held(self):
        self.backend.fail='followup'
        with self.assertRaises(RuntimeError): self.advance()
        self.assertFalse(any(x[0]=='release' for x in self.backend.events))

    def test_uncertain_release_blocks_automatic_resubmission(self):
        ids=self.advance();self.backend.fail='release'
        with self.assertRaises(RuntimeError): release_window(self.state,ids,self.backend,self.save)
        self.assertEqual(self.state['windows'][0]['release_requested'],ids[:1])
        with self.assertRaises(RuntimeError): self.advance()

    def test_queue_occupancy_reduces_window_without_dropping_models(self):
        self.backend.queue=20
        ids=self.dispatch();self.assertEqual(len(ids),14)
        self.assertLessEqual(self.backend.queue,50);self.assertEqual(len(self.ledger.state['allocations']),70)

    def test_full_queue_preserves_ready_reservations_no_submissions(self):
        self.backend.queue=49
        with self.assertRaises(RuntimeError): self.advance()
        self.assertEqual(self.state['phase'],'ready');self.assertFalse(self.backend.events)
        self.backend.queue=1;self.assertEqual(len(self.dispatch()),20)

    def test_nonfinal_and_duplicate_plans_rejected(self):
        for jobs in ([dict(self.jobs[0],stage='search')], [self.jobs[0],self.jobs[0]]):
            with self.assertRaises(ValueError): reserve_all(self.ledger,jobs)
        self.assertFalse(self.ledger.state['allocations'])

    def test_job_id_rejects_array_or_nonlocal_ambiguous_reply(self):
        self.assertEqual(job_id('123'),'123')
        for v in ('123_1','123;cluster','unknown',''): 
            with self.assertRaises(ValueError): job_id(v)

    def test_candidate_guard_before_output_or_missing_review(self):
        root=Path(__file__).resolve().parents[1]
        r=subprocess.run([sys.executable,'-B',str(root/'llm/final_window_controller_v1.py'),
            '--root',str(self.root),'--manifest',str(root/'config/flowbench_main_candidate_20261021.json'),
            '--freeze','MUST_NOT_READ','--authorization','MUST_NOT_READ','--output',str(self.root/'MUST_NOT_CREATE'),
            '--extension-review','MUST_NOT_READ'],capture_output=True,text=True,timeout=20)
        self.assertNotEqual(r.returncode,0);self.assertIn('not yet released',r.stderr)
        self.assertFalse((self.root/'MUST_NOT_CREATE').exists())

    def test_locked_plan_keeps95_aliases_and_all5seeds(self):
        from test_main_protocol import synthetic_searches
        from types import SimpleNamespace
        m=read(Path(__file__).resolve().parents[1]/'config/flowbench_main_candidate_20261021.json')
        provenance=dict(freeze_sha256='SYNTHETIC',authorization_sha256='NO_REAL_APPROVAL')
        for s in synthetic_searches(m,all_manual=True):
            s['provenance']=provenance
            atomic(self.root/'search'/f"{s['method']}_r{s['repeat']}"/'summary.json',s)
        plan,jobs,inputs=locked_plan(SimpleNamespace(output=self.root),m,provenance)
        self.assertEqual(plan['reference_count'],95);self.assertEqual(len(jobs),5);self.assertEqual(len(inputs),18)
        self.assertEqual({j['job']['seed'] for j in jobs},set(m['final_training_seeds']))
        with self.assertRaises(ValueError): locked_plan(SimpleNamespace(output=self.root),m,dict(freeze_sha256='OTHER'))


if __name__=='__main__': unittest.main(verbosity=2)

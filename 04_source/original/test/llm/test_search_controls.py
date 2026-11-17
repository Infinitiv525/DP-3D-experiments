"""CPU-only synthetic checks, including a real bounded subprocess timeout."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from search_controls import RandomConfigurator,TPEConfigurator,TrainingBudget,run_bounded_process,validate_parameters


def history():
    generator=RandomConfigurator(20261101)
    return [{'parameters':generator.ask(),'S':.4+i*.1} for i in range(3)]


class ControlTests(unittest.TestCase):
    def test_random_reproducibility_and_space(self):
        a=RandomConfigurator(1);b=RandomConfigurator(1)
        for _ in range(30):
            x=a.ask();self.assertEqual(x,b.ask());validate_parameters(x)
    def test_tpe_reproducibility(self):
        a=TPEConfigurator(1,history());b=TPEConfigurator(1,history())
        self.assertEqual(a.ask(),b.ask());a.tell(.3);b.tell(.3)
        self.assertEqual(a.ask(),b.ask());self.assertEqual(len(a.study.trials),5)
    def test_tpe_failure_penalty_not_free_retry(self):
        a=TPEConfigurator(1,history());a.ask();a.tell(failed=True)
        self.assertEqual(a.study.trials[-1].value,float('inf'))
        self.assertEqual(len(a.study.trials),4)
        self.assertTrue(a.export()['trials'][-1]['positive_infinity_penalty'])
        json.dumps(a.export(),allow_nan=False)
        a.ask();a.tell(.2)
    def test_budget_counts_failed_and_invalid_attempts(self):
        budget=TrainingBudget(10,5,shared_seconds=3,shared_attempts=3)
        self.assertEqual(budget.begin('bad_json'),7)
        budget.finish('bad_json',0,status='invalid_proposal')
        budget.begin('oom');budget.finish('oom',2,status='failed')
        self.assertFalse(budget.can_start());self.assertEqual(budget.spent_seconds,5)
        self.assertEqual(budget.attempts,5)
    def test_time_budget_and_visible_overrun(self):
        budget=TrainingBudget(1,12);budget.begin('x');budget.finish('x',1.01,status='timed_out')
        self.assertFalse(budget.can_start());self.assertGreater(budget.export()['overrun_seconds'],0)
    def test_success_score_and_guard(self):
        budget=TrainingBudget(2,1);budget.begin('x')
        with self.assertRaises(RuntimeError):budget.begin('y')
        budget.finish('x',1,.3,'complete');self.assertEqual(budget.export()['events'][-1]['S'],.3)
    def test_actual_subprocess_timeout(self):
        with tempfile.TemporaryDirectory(prefix='budget_unit_',dir=Path(__file__).parent) as folder:
            outcome=run_bounded_process([sys.executable,'-c','import time;time.sleep(5)'],.5,Path(folder)/'log',cleanup_seconds=.15)
            self.assertTrue(outcome['timed_out']);self.assertNotEqual(outcome['exit_code'],0)
            self.assertLess(outcome['wall_seconds'],2)
    def test_actual_subprocess_success(self):
        with tempfile.TemporaryDirectory(prefix='budget_unit_',dir=Path(__file__).parent) as folder:
            outcome=run_bounded_process([sys.executable,'-c','print("ok")'],2,Path(folder)/'log',cleanup_seconds=.2)
            self.assertFalse(outcome['timed_out']);self.assertEqual(outcome['exit_code'],0)
    def test_nonfinite_and_invalid_parameters(self):
        x=RandomConfigurator(1).ask();x['learning_rate']=float('nan')
        with self.assertRaises(ValueError):validate_parameters(x)
        with self.assertRaises(ValueError):TrainingBudget(float('inf'),12)


if __name__=='__main__':unittest.main(verbosity=2)

"""Small CPU-only tests, including actual abrupt process termination."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from persistent_search import PersistentSearch, owner_active
from search_controls import RandomConfigurator, TPEConfigurator


def options(method, mode='process'):
    sampler = RandomConfigurator(17)
    history = [{'parameters': sampler.ask(), 'S': .4 + i * .1} for i in range(3)]
    return dict(identity={'purpose': 'synthetic_cpu_recovery_not_main', 'fixture': 1},
                method=method, seed=91, history=history, training_limit=30,
                attempt_limit=7, shared_seconds=3, decision_limit=20, owner_mode=mode)


def stage_crash(root, mode):
    for method in ('random', 'tpe', 'external'):
        with PersistentSearch(Path(root) / method, **options(method, mode)) as search:
            search.ask(decision_cap=4 if method == 'external' else None)
            if method == 'external':
                search.selected(options(method)['history'][0]['parameters'], 1, {'reason': 'synthetic'})
            search.reserve_training(5)
            search.complete(2, .3)
            search.ask(decision_cap=4 if method == 'external' else None)
            if method != 'external': search.reserve_training(5)
    # Close locks deliberately, then die without committing trial outcomes.
    # Slurm owner stays active until the surrounding shell job has terminated.
    os._exit(77)


def verify_crash(root, mode):
    reports = {}
    for method in ('random', 'tpe', 'external'):
        with PersistentSearch(Path(root) / method, **options(method, mode)) as search:
            assert search.recover_interrupted()
            assert not search.recover_interrupted()
            report = search.export()
            assert report['attempts_spent'] == 5
            assert report['training_seconds'] == (5 if method == 'external' else 10)
            assert report['decision_seconds'] == (5 if method == 'external' else 0)
            assert report['state']['attempts'][-1]['status'] == 'interrupted_reserved_cost'
            if method != 'external':
                ref = RandomConfigurator(91) if method == 'random' else TPEConfigurator(91, options(method)['history'])
                for index, row in enumerate(report['state']['attempts']):
                    assert ref.ask() == row['parameters']
                    if method == 'tpe': ref.tell(.3 if index == 0 else None, failed=index == 1)
                assert search.ask()['parameters'] == ref.ask()
            reports[method] = search.export()
    return reports


class PersistentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='recovery-unit-', dir=Path(__file__).parent)
        self.path = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_replay_ready_and_completed(self):
        for method in ('random', 'tpe'):
            folder = self.path / method
            with PersistentSearch(folder, **options(method)) as search:
                first = search.ask()['parameters']
                search.reserve_training(5)
                search.complete(2, .3)
                second = search.ask()['parameters']
                saved = search.export()
            with PersistentSearch(folder, **options(method)) as resumed:
                self.assertEqual(resumed.pending['parameters'], second)
                self.assertEqual(resumed.export(), saved)
                resumed.reserve_training(5)
                resumed.complete(1, status='oom')
                third = resumed.ask()['parameters']
            reference = RandomConfigurator(91) if method == 'random' else TPEConfigurator(91, options(method)['history'])
            self.assertEqual(reference.ask(), first)
            if method == 'tpe': reference.tell(.3)
            self.assertEqual(reference.ask(), second)
            if method == 'tpe': reference.tell(failed=True)
            self.assertEqual(reference.ask(), third)

    def test_real_abrupt_exit_and_recovery(self):
        result = subprocess.run([sys.executable, '-B', __file__, '--crash', str(self.path)], timeout=30)
        self.assertEqual(result.returncode, 77)
        reports = verify_crash(self.path, 'process')
        self.assertEqual(set(reports), {'random', 'tpe', 'external'})

    def test_live_owner_and_writer_lock(self):
        with PersistentSearch(self.path, **options('random')) as search:
            with self.assertRaises(BlockingIOError): PersistentSearch(self.path, **options('random'))
            search.ask()
            search.reserve_training(5)
            with self.assertRaises(RuntimeError): search.recover_interrupted()

    def test_identity_and_software_checksum_guards(self):
        opts = options('random')
        with PersistentSearch(self.path, **opts): pass
        other = copy.deepcopy(opts)
        other['identity']['fixture'] = 2
        with self.assertRaises(ValueError): PersistentSearch(self.path, **other)
        with patch('persistent_search.fingerprint', return_value={'different': True}):
            with self.assertRaises(ValueError): PersistentSearch(self.path, **opts)
        path = self.path / 'state.json'
        content = json.loads(path.read_text())
        content['payload']['revision'] += 1
        path.write_text(json.dumps(content))
        with self.assertRaises(ValueError): PersistentSearch(self.path, **opts)

    def test_external_saved_proposal_and_budget(self):
        opts = options('external')
        with PersistentSearch(self.path, **opts) as search:
            search.ask(decision_cap=4)
            with self.assertRaises(RuntimeError): search.ask(decision_cap=4)
            parameters = opts['history'][0]['parameters']
            search.selected(parameters, 1.5, {'reason': 'saved, not regenerated'})
        with PersistentSearch(self.path, **opts) as resumed:
            self.assertEqual(resumed.pending['parameters'], parameters)
            self.assertEqual(resumed.export()['decision_seconds'], 1.5)
            self.assertEqual(resumed.reserve_training(100), 27)
            resumed.complete(29, .3)
            self.assertEqual(resumed.export()['training_overrun_seconds'], 2)
            with self.assertRaises(RuntimeError): resumed.ask(decision_cap=4)

    def test_invalid_duration_attempt_limit_and_score(self):
        for value in (True, -1, float('nan'), float('inf')):
            opts = options('random'); opts['shared_seconds'] = value
            with self.assertRaises(ValueError): PersistentSearch(self.path, **opts)
        opts = options('random'); opts['attempt_limit'] = 4
        with PersistentSearch(self.path, **opts) as search:
            search.ask(); search.reserve_training(1)
            with self.assertRaises(ValueError): search.complete(1, float('nan'))
            search.complete(1, .4)
            with self.assertRaises(RuntimeError): search.ask()

    def test_scheduler_failure_and_live_job_fail_closed(self):
        value = {'mode': 'slurm', 'job': '123', 'user': 'nobody', 'cluster': os.environ.get('SLURM_CLUSTER_NAME', '')}
        with patch('persistent_search.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'squeue')):
            with self.assertRaises(subprocess.CalledProcessError): owner_active(value)
        with patch('persistent_search.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '123 RUNNING\n', '')):
            self.assertTrue(owner_active(value))
        with patch('persistent_search.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')):
            self.assertFalse(owner_active(value))

    def test_terminal_job_not_yet_absent_blocks_recovery(self):
        value = {'mode': 'slurm', 'job': '123', 'user': 'nobody', 'cluster': os.environ.get('SLURM_CLUSTER_NAME', '')}
        with patch('persistent_search.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '123 PREEMPTED\n', '')):
            self.assertTrue(owner_active(value))

    def test_invalid_external_reply_is_counted(self):
        with PersistentSearch(self.path, **options('external')) as search:
            search.ask(decision_cap=4)
            search.selection_failed(2, metadata={'reason': 'schema rejected'})
            report = search.export()
            self.assertEqual(report['decision_seconds'], 2)
            self.assertEqual(report['training_seconds'], 3)
            self.assertEqual(report['attempts_spent'], 4)
            self.assertIsNone(search.pending)
            search.ask(decision_cap=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crash')
    parser.add_argument('--verify')
    parser.add_argument('--owner-mode', default='process')
    args, rest = parser.parse_known_args()
    if args.crash: stage_crash(args.crash, args.owner_mode)
    elif args.verify:
        report = {'status': 'PERSISTENT_SEARCH_SLURM_RECOVERY_PASSED', 'GPU_used': False,
                  'methods': verify_crash(args.verify, args.owner_mode)}
        Path(args.verify, 'recovery_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(report['status'])
    else: unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)

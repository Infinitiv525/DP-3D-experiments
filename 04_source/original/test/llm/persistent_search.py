"""JSON-only crash recovery for sequential search; no data/model access.

Reserve before launching a process. A lost running attempt keeps its entire
reservation. Recovery never re-launches it or refunds unmeasured consumption.
This per-search ledger is not a global Slurm allocation controller.
"""
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile

import psutil
from search_controls import RandomConfigurator, TPEConfigurator, validate_parameters


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprint():
    return {'python': platform.python_version(), 'numpy': importlib.metadata.version('numpy'),
            'optuna': importlib.metadata.version('optuna'),
            'sources': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('persistent_search.py', 'search_controls.py')}}


def duration(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Invalid duration')
    return float(value)


def owner(mode):
    if mode == 'slurm':
        job = os.environ['SLURM_JOB_ID']
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            job = os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
        return {'mode': mode, 'job': job, 'user': os.environ['SLURM_JOB_USER'],
                'cluster': os.environ.get('SLURM_CLUSTER_NAME', '')}
    if mode != 'process': raise ValueError('Unknown owner mode')
    return {'mode': mode, 'host': socket.gethostname(), 'pid': os.getpid(),
            'created': psutil.Process().create_time()}


def owner_active(value):
    """Fail closed on an unavailable scheduler or an uncheckable remote PID."""
    if value['mode'] == 'slurm':
        if value['cluster'] != os.environ.get('SLURM_CLUSTER_NAME', ''):
            raise RuntimeError('Cannot establish same Slurm cluster')
        result = subprocess.run(['squeue', '-h', '-r', '--states=all', '-u', value['user'], '-o', '%i %T'],
                                capture_output=True, text=True, timeout=20, check=True)
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) != 2: raise RuntimeError('Unexpected squeue output')
            # Even a terminal-looking entry can be in teardown/requeue. Require
            # absence, successful scheduler query, and --no-requeue owner jobs.
            if fields[0] == value['job']: return True
        return False
    if value['host'] != socket.gethostname(): raise RuntimeError('Cannot inspect remote owner process')
    try:
        process = psutil.Process(value['pid'])
        return process.create_time() == value['created'] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class PersistentSearch:
    def __init__(self, directory, *, identity, method, seed, history, training_limit,
                 attempt_limit, shared_seconds, decision_limit=0, owner_mode=None):
        if method not in ('random', 'tpe', 'external'): raise ValueError('Unknown method')
        if type(seed) is not int: raise ValueError('Integer seed required')
        if type(attempt_limit) is not int or attempt_limit < len(history): raise ValueError('Invalid attempt limit')
        if not identity: raise ValueError('Explicit data/training/prompt identity required')
        for row in history:
            validate_parameters(row['parameters'])
            if isinstance(row['S'], bool) or not math.isfinite(row['S']): raise ValueError('Invalid initial score')
        spec = {'identity': identity, 'method': method, 'seed': seed, 'history': history,
                'training_limit': duration(training_limit, True), 'attempt_limit': attempt_limit,
                'shared_seconds': duration(shared_seconds), 'decision_limit': duration(decision_limit),
                'software': fingerprint()}
        if spec['shared_seconds'] > spec['training_limit']: raise ValueError('Shared cost exceeds budget')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.mode = owner_mode or ('slurm' if 'SLURM_JOB_ID' in os.environ else 'process')
            self.broken = False
            path = self.directory / 'state.json'
            if path.exists():
                envelope = json.loads(path.read_text())
                self.state = envelope['payload']
                if hashlib.sha256(canonical(self.state)).hexdigest() != envelope['sha256']:
                    raise ValueError('Checkpoint checksum mismatch')
                if self.state['schema'] != 1 or self.state['spec'] != spec:
                    raise ValueError('Checkpoint identity, software or limits mismatch')
            else:
                self.state = {'schema': 1, 'revision': 0, 'spec': copy.deepcopy(spec), 'attempts': []}
                self._commit()
            self._replay()
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, 'lock', None) is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _check(self):
        if self.lock is None or self.broken: raise RuntimeError('Closed or failed writer; reopen checkpoint')

    def _commit(self):
        self._check()
        self.broken = True
        self.state['revision'] += 1
        payload = canonical(self.state)
        envelope = canonical({'payload': self.state, 'sha256': hashlib.sha256(payload).hexdigest()})
        descriptor, temporary = tempfile.mkstemp(prefix='.commit-', dir=self.directory)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(envelope)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / 'state.json')
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        self.broken = False

    def _replay(self):
        spec = self.state['spec']
        self.sampler = (RandomConfigurator(spec['seed']) if spec['method'] == 'random' else
                        TPEConfigurator(spec['seed'], spec['history']) if spec['method'] == 'tpe' else None)
        for item in self.state['attempts']:
            if self.sampler:
                if self.sampler.ask() != item['parameters']: raise RuntimeError('Sampler replay mismatch')
                if spec['method'] == 'tpe' and item['phase'] == 'finished':
                    self.sampler.tell(item['S'], failed=item['status'] != 'complete')

    @property
    def pending(self):
        items = self.state['attempts']
        return items[-1] if items and items[-1]['phase'] != 'finished' else None

    def export(self):
        spec = self.state['spec']
        training = spec['shared_seconds'] + sum(x['training_seconds'] for x in self.state['attempts'])
        decision = sum(x['decision_seconds'] for x in self.state['attempts'])
        return {'training_seconds': training, 'decision_seconds': decision,
                'remaining_training_seconds': max(0., spec['training_limit'] - training),
                'remaining_decision_seconds': max(0., spec['decision_limit'] - decision),
                'training_overrun_seconds': max(0., training - spec['training_limit']),
                'decision_overrun_seconds': max(0., decision - spec['decision_limit']),
                'attempts_spent': len(spec['history']) + len(self.state['attempts']),
                'state': copy.deepcopy(self.state)}

    def ask(self, decision_cap=None):
        self._check()
        if self.pending: raise RuntimeError('Resume or finish existing attempt first')
        report = self.export()
        if report['attempts_spent'] >= self.state['spec']['attempt_limit'] or report['remaining_training_seconds'] <= 0:
            raise RuntimeError('Budget exhausted')
        external = self.sampler is None
        if external:
            cap = duration(decision_cap, True)
            if cap > report['remaining_decision_seconds']: raise RuntimeError('Decision budget exhausted')
        else: cap = 0.
        item = {'number': report['attempts_spent'], 'phase': 'selecting' if external else 'ready',
                'parameters': None if external else self.sampler.ask(), 'metadata': {},
                'owner': owner(self.mode), 'training_seconds': 0., 'decision_seconds': cap,
                'S': None, 'status': None}
        self.state['attempts'].append(item)
        self._commit()
        return copy.deepcopy(item)

    def selected(self, parameters, seconds, metadata):
        self._check()
        if not self.pending or self.pending['phase'] != 'selecting': raise RuntimeError('No pending selection')
        validate_parameters(parameters)
        measured = duration(seconds)
        canonical(metadata)
        if owner(self.mode) != self.pending['owner']: raise RuntimeError('Selection owned by another process/job')
        self.pending.update(parameters=copy.deepcopy(parameters), metadata=copy.deepcopy(metadata),
                            decision_seconds=measured, phase='ready')
        self._commit()

    def reserve_training(self, cap_seconds):
        self._check()
        if not self.pending or self.pending['phase'] != 'ready': raise RuntimeError('No ready proposal')
        # A ready proposal has no running child and can be claimed after restart.
        old = self.pending['owner']
        if old != owner(self.mode) and owner_active(old): raise RuntimeError('Previous owner still active')
        cap = min(duration(cap_seconds, True), self.export()['remaining_training_seconds'])
        if cap <= 0: raise RuntimeError('Training budget exhausted')
        self.pending.update(phase='running', training_seconds=cap, owner=owner(self.mode))
        self._commit()
        return cap

    def selection_failed(self, seconds, status='invalid_reply', metadata=None):
        self._check()
        if not self.pending or self.pending['phase'] != 'selecting': raise RuntimeError('No pending selection')
        if owner(self.mode) != self.pending['owner']: raise RuntimeError('Selection owned by another process/job')
        if status not in ('invalid_reply', 'decision_timeout', 'decision_failed'): raise ValueError('Unknown failure')
        measured = duration(seconds)
        canonical(metadata)
        self.pending.update(phase='finished', decision_seconds=measured, status=status, metadata=copy.deepcopy(metadata))
        self._commit()

    def complete(self, seconds, score=None, status='complete'):
        self._check()
        if not self.pending or self.pending['phase'] != 'running': raise RuntimeError('No running trial')
        if owner(self.mode) != self.pending['owner']: raise RuntimeError('Trial owned by another process/job')
        measured = duration(seconds)
        if status not in ('complete', 'failed', 'timeout', 'oom'): raise ValueError('Unknown outcome')
        if status == 'complete' and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError('Finite score required')
        self.pending.update(phase='finished', training_seconds=measured, status=status,
                            S=score if status == 'complete' else None)
        self._commit()
        self._replay()

    def recover_interrupted(self):
        self._check()
        if not self.pending or self.pending['phase'] == 'ready': return False
        if owner_active(self.pending['owner']): raise RuntimeError('Previous owner still active; recovery refused')
        self.pending.update(phase='finished', status='interrupted_reserved_cost', S=None)
        self._commit()
        self._replay()
        return True

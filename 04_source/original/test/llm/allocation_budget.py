"""Conservative, shared physical GPU-allocation ledger. No training-budget refund.

Reserve a complete Slurm wall limit BEFORE submission; submit on hold, attach
job ID, then release it. Interrupted submission keeps the reservation. An
operator may reconcile it, never silently refund an uncertain allocation.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def wall_seconds(value):
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', value)
    if not match: raise ValueError('Unrecognized Slurm duration: ' + value)
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    if minutes >= 60 or seconds >= 60: raise ValueError('Invalid time')
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


class PhysicalBudget:
    def __init__(self, directory, *, gpu_seconds_limit, identity):
        if isinstance(gpu_seconds_limit, bool) or not isinstance(gpu_seconds_limit, (int, float)) or not math.isfinite(gpu_seconds_limit) or gpu_seconds_limit <= 0:
            raise ValueError('Positive finite GPU-seconds required')
        if not identity: raise ValueError('Explicit benchmark identity required')
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'allocation.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.spec = {'schema': 1, 'gpu_seconds_limit': gpu_seconds_limit, 'identity': identity,
                         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            path = self.directory / 'allocations.json'
            if path.exists():
                envelope = json.loads(path.read_text()); self.state = envelope['payload']
                if hashlib.sha256(canonical(self.state)).hexdigest() != envelope['sha256']: raise ValueError('Ledger checksum mismatch')
                if self.state['spec'] != self.spec: raise ValueError('Budget identity/code/limit mismatch')
            else:
                self.state = {'spec': self.spec, 'allocations': [], 'revision': 0}
                self.save()
        except BaseException:
            self.close(); raise

    def close(self):
        if getattr(self, 'lock', None): self.lock.close(); self.lock = None
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def save(self):
        if self.lock is None: raise RuntimeError('Closed ledger')
        self.state['revision'] += 1
        content = canonical({'payload': self.state, 'sha256': hashlib.sha256(canonical(self.state)).hexdigest()})
        fd, temporary = tempfile.mkstemp(prefix='.alloc-', dir=self.directory)
        try:
            with os.fdopen(fd, 'wb') as stream: stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.directory / 'allocations.json')
            fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    def report(self):
        used = sum(row['charged_gpu_seconds'] for row in self.state['allocations'])
        return {'gpu_seconds_limit': self.spec['gpu_seconds_limit'], 'charged_or_reserved_gpu_seconds': used,
                'available_gpu_seconds': max(0, self.spec['gpu_seconds_limit'] - used),
                'overrun_gpu_seconds': max(0, used - self.spec['gpu_seconds_limit']), 'state': self.state}

    def reserve(self, label, gpus, seconds):
        if type(gpus) is not int or gpus < 1 or type(seconds) is not int or seconds < 1: raise ValueError('Positive integer GPU count and wall seconds required')
        if not isinstance(label, str) or not label: raise ValueError('Label required')
        if any(row['label'] == label for row in self.state['allocations']): raise RuntimeError('Duplicate allocation label; inspect saved state before retry')
        reservation = gpus * (seconds + 1)  # one-second scheduler granularity reserve
        if reservation > self.report()['available_gpu_seconds']: raise RuntimeError('Physical GPU budget exhausted')
        self.state['allocations'].append({'label': label, 'gpus': gpus, 'wall_limit_seconds': seconds,
            'charged_gpu_seconds': reservation, 'reservation_gpu_seconds': reservation, 'job_id': None, 'status': 'reserved_before_submit'})
        self.save()

    def attach(self, label, job_id):
        if not re.fullmatch(r'[0-9]+', job_id): raise ValueError('Non-array numeric job ID required')
        row = next(r for r in self.state['allocations'] if r['label'] == label)
        if row['job_id'] is not None or any(r['job_id'] == job_id for r in self.state['allocations']): raise RuntimeError('Already attached')
        row.update(job_id=job_id, status='submitted_on_hold'); self.save()

    def settle(self, label, scheduler_text):
        row = next(r for r in self.state['allocations'] if r['label'] == label)
        fields = dict(re.findall(r'(?:^|\s)(\w+)=([^\s]+)', scheduler_text))
        expected = self.spec['identity']
        if fields.get('JobId') != row['job_id'] or fields.get('UserId', '').split('(')[0] != expected['user'] or fields.get('Account') != expected['account']:
            raise ValueError('Scheduler job identity differs')
        if fields.get('JobState') not in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE'}:
            raise RuntimeError('Allocation not definitively ended')
        if fields.get('Requeue') != '0': raise RuntimeError('Requeue-enabled job requires manual reconciliation')
        allocated = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', fields.get('AllocTRES', ''))
        if allocated is None or int(allocated[1]) != row['gpus']: raise ValueError('Allocated GPU count not verified')
        if wall_seconds(fields['TimeLimit']) != row['wall_limit_seconds']: raise ValueError('Wall limit differs')
        runtime = wall_seconds(fields['RunTime']); charged = (runtime + 1) * row['gpus']
        if row['status'] == 'settled':
            if row['charged_gpu_seconds'] != charged: raise RuntimeError('Changed scheduler measurement')
            return
        row.update(status='settled', charged_gpu_seconds=charged, reported_runtime_seconds=runtime,
                   scheduler_fields=fields, scheduler_sha256=hashlib.sha256(scheduler_text.encode()).hexdigest())
        self.save()


def main(a):
    config = json.loads(a.config.read_text())
    with PhysicalBudget(a.ledger, **config) as ledger:
        if a.action == 'submit':
            if a.gpus != 1: raise ValueError('This pilot submitter supports exactly one GPU')
            if a.seconds % 60: raise ValueError('Whole-minute allocation required')
            script = a.script.resolve(strict=True)
            # Reservation is intentionally NOT refunded when any submission step fails.
            ledger.reserve(a.label, a.gpus, a.seconds)
            result = subprocess.run(['sbatch', '--hold', '--parsable', '--no-requeue', '--export=NIL',
                '--account=' + config['identity']['account'], '--qos=' + config['identity']['account'],
                '--partition=gpu_short', '--nodes=1', '--ntasks=1', '--gres=gpu:1',
                '--time=' + str(a.seconds // 60), str(script), *a.script_args], check=True, capture_output=True, text=True)
            job = result.stdout.strip()
            ledger.attach(a.label, job)
            subprocess.run(['scontrol', 'release', job], check=True)
            print(json.dumps({'job_id': job, 'label': a.label, 'budget': ledger.report()}, allow_nan=False))
        elif a.action == 'settle':
            for row in ledger.state['allocations']:
                if row['job_id'] is None or row['status'] == 'settled': continue
                # On query failure/uncertain state the full reservation remains charged.
                result = subprocess.run(['scontrol', 'show', 'job', '-o', row['job_id']], capture_output=True, text=True, check=True, timeout=20)
                ledger.settle(row['label'], result.stdout)
            print(json.dumps(ledger.report(), allow_nan=False))
        else: print(json.dumps(ledger.report(), allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['submit', 'settle', 'report'])
    p.add_argument('--config', type=Path, required=True); p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--label'); p.add_argument('--script', type=Path); p.add_argument('--gpus', type=int, default=1); p.add_argument('--seconds', type=int)
    p.add_argument('script_args', nargs='*')
    main(p.parse_args())

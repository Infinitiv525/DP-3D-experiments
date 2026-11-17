"""Main train/validation workflow. Backend is injected; tests need no GPU/data.

Fixed tasks never automatically retry. Search resumes committed sampler state;
an interrupted process retains its full reservation and consumes an attempt.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import fcntl

from main_protocol import (identity, read, shared_plan, training_config,
                           training_seed, winner, PARAMETERS)
from persistent_search import PersistentSearch, owner, owner_active, duration
from sequential_feedback import build_feedback
from configurator_contract_v3 import validate_feedback


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    fd, temporary = tempfile.mkstemp(prefix='.main-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def checked_result(result, cap):
    duration(result['seconds'])
    if result['seconds'] > cap + 5: raise ValueError('Process exceeded cleanup margin; retain reservation')
    if result['status'] not in ('complete', 'failed', 'timeout', 'oom'): raise ValueError('Unknown outcome')
    if result['status'] == 'complete':
        m = result['measurement']
        for key in ('S', 'velocity_error', 'pressure_error', 'divergence'):
            v = m[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError('Invalid measurement')
        if abs(m['S'] - .5 * (m['velocity_error'] + m['pressure_error'])) > 1e-10:
            raise ValueError('Objective differs')
    return result


def fixed_task(manifest, job, directory, backend, provenance, owner_mode=None):
    """Exactly-once fixed job; crash is an explicit blocked failure, not a retry."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    mode = owner_mode or ('slurm' if 'SLURM_JOB_ID' in os.environ else 'process')
    spec = dict(manifest_sha256=identity(manifest), job=job, provenance=provenance)
    with (directory / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'task.json'
        def commit(state): atomic(path, dict(payload=state, sha256=identity(state)))
        if path.exists():
            envelope = read(path); state = envelope['payload']
            if identity(state) != envelope['sha256'] or state['spec'] != spec: raise ValueError('Fixed-task identity/checksum differs')
            if state['status'] == 'running':
                if owner_active(state['owner']): raise RuntimeError('Fixed task still owned by a live job/process')
                state['status'] = 'interrupted_reserved_cost'; commit(state)
            if state['status'] != 'complete': raise RuntimeError('Fixed task failed/interrupted; no automatic retry')
            return copy.deepcopy(state)
        state = dict(spec=spec, status='running', owner=owner(mode),
                     seconds=manifest['trial_process_seconds_cap'], result=None)
        commit(state)  # Before starting any child process.
        result = checked_result(backend.train(job, training_config(manifest, job), directory,
                                             state['seconds']), state['seconds'])
        state.update(status=result['status'], seconds=result['seconds'], result=result); commit(state)
        if state['status'] != 'complete': raise RuntimeError('Fixed task did not complete; downstream phase blocked')
        return copy.deepcopy(state)


def shared_rows(manifest, records, repeat):
    expected = [j for j in shared_plan(manifest) if j['repeat'] == repeat]
    if len(records) != 3: raise ValueError('Three newly completed shared starts required')
    rows = []
    for job, record in zip(expected, records, strict=True):
        if (record['spec']['manifest_sha256'] != identity(manifest) or record['spec']['job'] != job
                or record['status'] != 'complete'): raise ValueError('Shared provenance/status differs')
        checked_result(record['result'], manifest['trial_process_seconds_cap'])
        if record['seconds'] != record['result']['seconds']: raise ValueError('Shared cost differs')
        rows.append(dict(id=job['id'], **job['parameters'], **record['result']['measurement'],
                         training_gpu_hours=record['seconds'] / 3600))
    return rows


def run_search(manifest, method, repeat, shared_records, directory, backend, provenance, owner_mode=None):
    if method not in manifest['methods'] or type(repeat) is not int or not 0 <= repeat < 3: raise ValueError('Unknown arm')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rows = shared_rows(manifest, shared_records, repeat)
    if any(r['spec']['provenance'] != provenance for r in shared_records):
        raise ValueError('Shared source freeze/authorization differs')
    history = [dict(parameters={k: row[k] for k in PARAMETERS}, S=row['S']) for row in rows]
    llm = method not in ('random', 'tpe')
    spec = dict(manifest_sha256=identity(manifest), method=method, repeat=repeat,
                shared_sha256=identity(shared_records), provenance=provenance, phase='main_search')
    with PersistentSearch(directory / 'search', identity=spec, method='external' if llm else method,
            seed=manifest['search_repeat_seeds'][repeat] + manifest['sampler_seed_offset'], history=history,
            training_limit=manifest['training_seconds_per_method_repeat'], attempt_limit=12,
            shared_seconds=sum(r['seconds'] for r in shared_records),
            decision_limit=manifest['proposal_seconds_per_llm_search_cap'] if llm else 0, owner_mode=owner_mode) as search:
        search.recover_interrupted()
        while True:
            report = search.export()
            if search.pending is None:
                if report['attempts_spent'] >= 12: reason = 'attempt_cap'; break
                if report['remaining_training_seconds'] <= 5: reason = 'training_time_cap'; break
                if llm and report['remaining_decision_seconds'] <= 5: reason = 'decision_time_cap'; break
                search.ask(decision_cap=min(manifest['proposal_process_seconds_cap'], report['remaining_decision_seconds']) if llm else None)
            item = search.pending; number = item['number']; attempt_dir = directory / f'a{number:02}'
            attempt_dir.mkdir(exist_ok=True)
            measurements = {f"a{i['number']:02}": i['metadata']['training']['measurement']
                            for i in search.state['attempts'] if i['status'] == 'complete'}
            feedback = build_feedback(rows, search.export(), measurements, 20)
            feedback['phase'] = 'main_search'; validate_feedback(feedback)
            atomic(attempt_dir / 'feedback.json', feedback)
            if item['phase'] == 'selecting':
                decision = backend.propose(method, feedback, attempt_dir, item['decision_seconds'])
                duration(decision['seconds'])
                if decision['seconds'] > item['decision_seconds'] + 5: raise ValueError('Decision overrun; retain reservation')
                if decision['status'] == 'complete':
                    search.selected(decision['parameters'], decision['seconds'], dict(proposal=decision))
                else:
                    search.selection_failed(decision['seconds'], decision['status'], dict(proposal=decision))
            if search.pending and search.pending['phase'] == 'ready':
                cap = search.reserve_training(manifest['trial_process_seconds_cap'])
                job = dict(id=f'{method}_r{repeat}_a{number:02}', stage='search', seed=training_seed(manifest, repeat, number),
                           parameters=search.pending['parameters'], epochs=20, fresh_initialization=True)
                result = checked_result(backend.train(job, training_config(manifest, job), attempt_dir, cap), cap)
                search.pending['metadata']['training'] = result
                search.complete(result['seconds'], result['measurement']['S'] if result['status'] == 'complete' else None, result['status'])
            atomic(directory / 'search_report.json', search.export())
        report = search.export()
        if report['training_overrun_seconds'] or report['decision_overrun_seconds']:
            raise RuntimeError('Logical budget exceeded; do not publish a complete benchmark arm')
        trials = [dict(number=j, seed=training_seed(manifest, repeat, j), parameters=h['parameters'], status='complete', S=h['S'])
                  for j, h in enumerate(history)]
        trials += [dict(number=i['number'], seed=training_seed(manifest, repeat, i['number']),
                        parameters=i['parameters'], status=i['status'], S=i['S']) for i in report['state']['attempts']]
        result = dict(status='MAIN_SEARCH_COMPLETE', phase='main_search', epochs=20, test_evaluated=False,
                      manifest_sha256=identity(manifest), method=method, repeat=repeat, trials=trials,
                      provenance=provenance,
                      training_seconds_spent=report['training_seconds'], decision_seconds_spent=report['decision_seconds'],
                      stop_reason=reason, ledger_revision=report['state']['revision'])
        winner(manifest, result)  # Reject incomplete/premature completion before publishing.
        atomic(directory / 'summary.json', result); return result


def final_catalog(manifest, plan, records):
    if plan['manifest_sha256'] != identity(manifest) or plan['reference_count'] != 95 or plan['test_evaluated']:
        raise ValueError('Balanced untested main plan required')
    entries = {}
    for job in plan['physical_trainings']:
        record = records[job['id']]
        if record['status'] != 'complete' or record['spec']['job'] != job or record['spec']['manifest_sha256'] != identity(manifest):
            raise ValueError('Incomplete/mismatched final training')
        entries[job['id']] = dict(result=record['result'], task_sha256=identity(record))
    return dict(status='MAIN_FINAL_TRAINING_COMPLETE_NOT_TESTED', manifest_sha256=identity(manifest),
                reference_count=95, references=plan['references'], trainings=entries, test_evaluated=False)

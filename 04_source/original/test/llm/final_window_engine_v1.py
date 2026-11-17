"""Bounded final-job scheduling; no model choice, dataset access, or statistics.

Reserve the whole balanced final phase before any GPU submission. Each window
fits GPU+capture+next-controller jobs under MaxSubmitJobs50. A partial/uncertain
submission is never automatically repeated or refunded.
"""
MAX_WINDOW = 20
MAX_SUBMIT = 50


def reserve_all(ledger, jobs):
    if not jobs or any(j['stage'] != 'final' or j['seconds'] != 900 for j in jobs):
        raise ValueError('Nonempty fixed900s final allocation plan required')
    labels = ['final:' + j['label'] for j in jobs]
    if len(set(labels)) != len(labels): raise ValueError('Unique final allocations required')
    if any(r['label'] in labels for r in ledger.state['allocations']):
        raise RuntimeError('Existing final reservation: inspect partial transaction, never reserve again')
    if sum(j['seconds'] + 1 for j in jobs) > ledger.report()['available_gpu_seconds']:
        raise RuntimeError('Whole balanced final phase does not fit; no submission')
    for label, job in zip(labels, jobs, strict=True): ledger.reserve(label, 1, job['seconds'])


def advance(state, jobs, ledger, backend, save):
    """Returns GPU IDs for release AFTER the caller closes the physical lock."""
    phase = state['phase']
    if phase in ('complete', 'ready_to_seal'): return []
    if phase == 'new':
        if any(r['status'] != 'settled' for r in ledger.state['allocations']):
            raise RuntimeError('All previous allocations must be closed')
        state['phase'] = 'reserving'; save()
        reserve_all(ledger, jobs)
        state.update(phase='ready', next_index=0, windows=[]); save()
    elif phase == 'window_submitted':
        window = state['windows'][-1]
        if window['released_gpu_ids'] != window['gpu_ids']:
            raise RuntimeError('Uncertain/partial release; inspect manually')
        for job in jobs[window['start']:window['end']]:
            row = next(r for r in ledger.state['allocations'] if r['label'] == 'final:' + job['label'])
            if row['status'] != 'settled': raise RuntimeError('Previous capture incomplete; keep all reservations')
            backend.review(job)
        window['review'] = 'ALL_COMPLETE_AND_VERIFIED'; state['phase'] = 'ready'; save()
    elif phase != 'ready':
        raise RuntimeError('Uncertain previous transaction; no automatic retry')
    if state['next_index'] == len(jobs):
        state['phase'] = 'ready_to_seal'; save(); return []
    capacity = (MAX_SUBMIT - backend.queue_count() - 1) // 2
    count = min(MAX_WINDOW, capacity, len(jobs) - state['next_index'])
    if count <= 0: raise RuntimeError('No scheduler capacity; resume ready state after unrelated jobs finish')
    start = state['next_index']; end = start + count
    window = dict(start=start, end=end, gpu_ids=[], capture_ids=[], next_controller_id=None,
                  release_requested=[], released_gpu_ids=[], review=None)
    state['windows'].append(window); state['phase'] = 'submitting'; save()
    for job in jobs[start:end]:
        gpu = backend.submit_gpu(job)
        ledger.attach('final:' + job['label'], gpu)
        window['gpu_ids'].append(gpu); save()
    for job, gpu in zip(jobs[start:end], window['gpu_ids'], strict=True):
        capture = backend.submit_capture(job, gpu)
        window['capture_ids'].append(capture); save()
    window['next_controller_id'] = backend.submit_followup(window['capture_ids'])
    state.update(phase='window_submitted', next_index=end); save()
    return list(window['gpu_ids'])


def release_window(state, ids, backend, save):
    if not ids: return
    window = state['windows'][-1]
    if state['phase'] != 'window_submitted' or ids != window['gpu_ids']:
        raise RuntimeError('Matching fully attached window required')
    for job in ids:
        window['release_requested'].append(job); save()  # Before external action.
        backend.release(job)
        window['released_gpu_ids'].append(job); save()

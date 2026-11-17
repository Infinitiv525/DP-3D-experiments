"""Read-only inference of all locked finals, after explicit heldout authorization."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm'))
from final_measurement_gate_v2 import (ROOT, MAIN, DEST, POLICY, authorized, own_allocation,
                                       exclusive_claim, TestOnly, atomic)
from main_protocol import read, digest, identity, METHODS


def geometry_rows(rows):
    result = []
    for g in sorted({r['geometry'] for r in rows}):
        group = [r for r in rows if r['geometry'] == g]
        if len(group) != 10: raise ValueError('Exactly10Re per geometry required')
        values = {k: sum(r[k] for r in group) / 10 for k in ('velocity_error', 'pressure_error',
                    'predicted_divergence_rms', 'reference_divergence_rms')}
        result.append(dict(geometry=g, count=10, S=.5*(values['velocity_error']+values['pressure_error']), **values))
    if len(result) != 15: raise ValueError('Exactly15 test geometries required')
    return result


def latency_pass(per_round):
    if set(per_round) != {'1', '2', '3'} or any(v['n'] != 1000 for v in per_round.values()):
        raise ValueError('Three complete1000sample rounds required')
    return all(v['wall_ms']['p99'] <= 5 for v in per_round.values())


def run(stage):
    # Both checks precede numerical imports, array mapping, CUDA and checkpoint loading.
    manifest, lock = authorized(); own_allocation(stage)
    exclusive_claim(DEST / (stage + '.claim.json'), dict(stage=stage, job_id=os.environ['SLURM_JOB_ID'],
        authorization_sha256=digest(DEST / 'authorization.json'), started_unix=time.time()))
    out = DEST / stage; out.mkdir(exist_ok=False)
    started = time.perf_counter(); cpu = time.process_time()
    import numpy as np
    import torch
    from train_flowbench_smoke import SteadyFNO, gauge_target
    from flowbench_protocol_metrics import fluid_errors, aggregate_geometry, validate_split
    from flowbench_physics import divergence_values
    from audit_protocol_latency import summarize
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != manifest['required_gpu_name']:
        raise ValueError('NVIDIA H200 required')
    info = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,compute_mode',
        '--format=csv'], capture_output=True, text=True, timeout=10)
    if info.returncode: raise RuntimeError('Cannot archive GPU identity')
    environment = dict(gpu=torch.cuda.get_device_name(), gpu_info=info.stdout, torch=torch.__version__,
        cuda=torch.version.cuda, numpy=np.__version__, python=sys.version, slurm_job_id=os.environ['SLURM_JOB_ID'],
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_threads=4, tf32=False, cudnn_benchmark=False)
    atomic(out / 'environment.json', environment)
    data = ROOT / manifest['dataset_relative']; split = read(data / 'split.json'); validate_split(split)
    stats = read(ROOT / manifest['normalization_source_relative'])['stats']
    rawx = np.load(data / 'X_pilot.npy', mmap_mode='r', allow_pickle=False)
    if rawx.shape != (1000, 2, 64, 64, 64): raise ValueError('Unexpected X array')
    # Y is mapped only by the separately authorized accuracy stage, never latency.
    if stage == 'accuracy':
        rawy = np.load(data / 'Y_pilot.npy', mmap_mode='r', allow_pickle=False)
        if rawy.shape != (1000, 4, 64, 64, 64): raise ValueError('Unexpected Y array')
        x = TestOnly(rawx, lock['test_ids']); y = TestOnly(rawy, lock['test_ids'])
    def timer():
        if time.perf_counter() - started > POLICY['wall_seconds_per_stage'] - 120:
            raise TimeoutError('Reserved stage wall budget; no automatic replay')
    def tensor(value): return torch.tensor(value, dtype=torch.float32, device='cuda').view(1, -1, 1, 1, 1)
    xm, xs = tensor(stats['X']['mean']), tensor(stats['X']['std'])
    ym, ys = tensor(stats['Y']['mean']), tensor(stats['Y']['std'])
    coords = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, 64, device='cuda')]*3, indexing='ij'))[None]
    if stage == 'latency':
        selected = lock['latency_input_ids']
        if any(i not in split['indices']['validation'] for i in selected): raise PermissionError('Validation-only latency inputs')
        host = [torch.from_numpy(np.array(rawx[i], copy=True))[None] for i in selected]
        device = [v.cuda() for v in host]; features = [torch.cat(((v-xm)/xs, coords), 1) for v in device]
    model_reports = {}
    with torch.inference_mode():
        for number, entry in enumerate(lock['models']):
            timer(); model_start = time.perf_counter()
            checkpoint = torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
            source = read(entry['summary']); tc = checkpoint['config']['training']
            if checkpoint['config'] != source['config'] or checkpoint['normalization'] != stats:
                raise ValueError('Checkpoint config/norm identity differs')
            model = SteadyFNO(tc['width'], tc['layers'], tc['modes']).cuda().eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            folder = out / entry['id']; folder.mkdir(exist_ok=False)
            torch.cuda.reset_peak_memory_stats()
            if stage == 'accuracy':
                rows = []
                for i in lock['test_ids']:
                    timer(); rx = x[i]
                    a = torch.from_numpy(np.array(rx, copy=True))[None].cuda()
                    target = torch.from_numpy(gauge_target(rx, y[i]))[None].cuda()
                    prediction = model(torch.cat(((a-xm)/xs, coords), 1))*ys+ym
                    if not torch.isfinite(prediction).all(): raise ValueError('Nonfinite prediction')
                    fluid = a[:, 0] > 0
                    row = fluid_errors(prediction[0], target[0], fluid[0])
                    if row['velocity_below_epsilon'] or row['pressure_below_epsilon']:
                        raise ValueError('Degenerate reference norm; no silent dropping')
                    row.update(sample=i, geometry=split['geometry_groups'][i],
                        predicted_divergence_rms=float(divergence_values(prediction[:,:3], fluid).square().mean().sqrt()),
                        reference_divergence_rms=float(divergence_values(target[:,:3], fluid).square().mean().sqrt()))
                    rows.append(row)
                geos = geometry_rows(rows); metrics = aggregate_geometry(rows)
                atomic(folder / 'test_rows.json', rows); atomic(folder / 'geometry_rows.json', geos)
                report = dict(metrics=metrics, geometry_rows=geos, test_rows_sha256=digest(folder / 'test_rows.json'),
                              geometry_rows_sha256=digest(folder / 'geometry_rows.json'))
            else:
                calls = dict(model_only=lambda i: model(features[i]),
                    gpu_pipeline=lambda i: model(torch.cat(((device[i]-xm)/xs, coords), 1))*ys+ym,
                    host_to_host=lambda i: (model(torch.cat(((host[i].to('cuda')-xm)/xs, coords), 1))*ys+ym).cpu())
                for i in range(len(selected)):
                    expected = (calls['model_only'](i)*ys+ym).cpu()
                    for mode in ('gpu_pipeline', 'host_to_host'):
                        actual = calls[mode](i).cpu()
                        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                            raise ValueError('Timing wrappers changed predictions')
                records = []
                for k in range(3):
                    order = POLICY['modes'][k:] + POLICY['modes'][:k]
                    for mode in order:
                        call = calls[mode]
                        for j in range(100):
                            output = call(j % len(selected)); torch.cuda.synchronize()
                        begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                        for j in range(1000):
                            timer(); index = j % len(selected); torch.cuda.synchronize()
                            t = time.perf_counter(); begin.record(); output = call(index)
                            end.record(); torch.cuda.synchronize()
                            elapsed = (time.perf_counter()-t)*1000
                            records.append(dict(round=k+1, mode=mode, sample=selected[index],
                                wall_ms=elapsed, cuda_event_ms=begin.elapsed_time(end)))
                    # Preserve completed rounds even if a later round fails.
                    atomic(folder / 'measurements.json', records)
                per_round = {m: {str(k): summarize([r for r in records if r['mode']==m and r['round']==k], [5])
                    for k in (1,2,3)} for m in POLICY['modes']}
                per_mode = {m: summarize([r for r in records if r['mode']==m], [5]) for m in POLICY['modes']}
                report = dict(per_round=per_round, per_mode=per_mode, measured_count=len(records),
                    primary_pass=latency_pass(per_round['gpu_pipeline']), wrappers_bit_identical=True,
                    measurements_sha256=digest(folder / 'measurements.json'), latency_input_ids=selected)
            torch.cuda.synchronize()
            report.update(status='FINAL_MODEL_MEASUREMENT_COMPLETE', stage=stage, **entry,
                elapsed_seconds=time.perf_counter()-model_start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            atomic(folder / 'summary.json', report); model_reports[entry['id']] = report
            atomic(out / 'progress.json', dict(completed=len(model_reports), total=70, latest=entry['id']))
            print('MODEL_MEASUREMENT_COMPLETE', stage, number+1, '/70', entry['id'], flush=True)
            del model, checkpoint
    result = dict(status='FINAL_MEASUREMENT_COMPLETE', stage=stage, model_count=70,
        logical_reference_count=95, models=model_reports, references=lock['references'],
        test_targets_loaded=stage=='accuracy', authorization_sha256=digest(DEST / 'authorization.json'),
        catalog_sha256=lock['catalog_sha256'], policy=POLICY, environment=environment,
        elapsed_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
        linux_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if stage == 'accuracy':
        # Every alias uses exactly the same physical result, not a new independent draw.
        geometries = [g['geometry'] for g in next(iter(model_reports.values()))['geometry_rows']]
        tensors = {metric: np.full((6,15,3,5), np.nan) for metric in ('S','velocity_error','pressure_error')}
        refs = []
        for ref in lock['references']:
            report = model_reports[ref['training_id']]
            refs.append(dict(**ref, test_metrics=report['metrics']))
            if ref['method'] == 'manual': continue
            mi=METHODS.index(ref['method']); ri=ref['repeat']; fi=ref['seed']-410001
            bygeo = {v['geometry']: v for v in report['geometry_rows']}
            for gi, g in enumerate(geometries):
                for metric in tensors: tensors[metric][mi,gi,ri,fi] = bygeo[g][metric]
        if any(not np.isfinite(t).all() for t in tensors.values()): raise ValueError('Incomplete reference matrix')
        from main_statistics import primary_contrasts
        atomic(out / 'paired_scores.json', dict(methods=list(METHODS), geometries=geometries,
            search_repeats=[0,1,2], final_seeds=list(range(410001,410006)),
            tensors={k:v.tolist() for k,v in tensors.items()}, references=refs))
        atomic(out / 'predeclared_statistics.json', primary_contrasts(tensors['S']))
        result['method_mean_S']={m:float(tensors['S'][i].mean()) for i,m in enumerate(METHODS)}
        result['manual_mean_S']=sum(r['test_metrics']['S'] for r in refs if r['method']=='manual')/5
    else:
        result['primary_pass_count'] = sum(r['primary_pass'] for r in model_reports.values())
        result['warning'] = 'Repeated timings are not independent scientific replicas. Instrumentation included; GPU co-tenancy/clocks may affect tails. No whole-application5ms guarantee.'
    atomic(out / 'summary.json', result)
    print('FINAL_MEASUREMENT_COMPLETE', stage, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['accuracy', 'latency'])
    run(p.parse_args().stage)

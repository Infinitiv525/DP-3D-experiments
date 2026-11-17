"""Integration check on OLD validation only. No heldout labels or tuning."""
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm'))
from final_measurement_gate_v3 import ROOT, DEST, authorized, own_allocation, exclusive_claim, TestOnly, atomic
from main_protocol import read, digest
from measure_final_models_v4 import sample_metrics, geometry_rows


def run():
    manifest, lock = authorized(); own_allocation('validation_smoke')
    exclusive_claim(DEST/'validation_smoke.claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],
        authorization_sha256=digest(DEST/'authorization.json'),test_targets_loaded=False))
    started=time.perf_counter();out=DEST/'validation_smoke';out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from train_flowbench_smoke import SteadyFNO, gauge_target
    from flowbench_protocol_metrics import aggregate_geometry, validate_split
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA H200':
        raise ValueError('H200 required for numerical integration check')
    data=ROOT/manifest['dataset_relative'];split=read(data/'split.json');validate_split(split)
    ids=sorted(split['indices']['validation'])
    if len(ids)!=150 or set(ids).intersection(split['indices']['test']):raise ValueError('Validation-only inputs required')
    # These guards permit validation IDs ONLY, even though arrays contain all splits.
    x=TestOnly(np.load(data/'X_pilot.npy',mmap_mode='r',allow_pickle=False),ids)
    y=TestOnly(np.load(data/'Y_pilot.npy',mmap_mode='r',allow_pickle=False),ids)
    entry=lock['models'][0];source=read(entry['summary'])
    cp=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
    stats=read(ROOT/manifest['normalization_source_relative'])['stats']
    if cp['config']!=source['config'] or cp['normalization']!=stats:raise ValueError('Checkpoint provenance differs')
    tc=cp['config']['training'];model=SteadyFNO(tc['width'],tc['layers'],tc['modes']).cuda().eval()
    model.load_state_dict(cp['model'],strict=True)
    def tensor(v):return torch.tensor(v,device='cuda',dtype=torch.float32).view(1,-1,1,1,1)
    xm,xs=tensor(stats['X']['mean']),tensor(stats['X']['std'])
    ym,ys=tensor(stats['Y']['mean']),tensor(stats['Y']['std'])
    coords=torch.stack(torch.meshgrid(*[torch.linspace(0,1,64,device='cuda')]*3,indexing='ij'))[None]
    rows=[]
    with torch.inference_mode():
        for i in ids:
            if time.perf_counter()-started>780:raise TimeoutError('Validation preflight wall budget')
            rx=x[i];a=torch.from_numpy(np.array(rx,copy=True))[None].cuda()
            target=torch.from_numpy(gauge_target(rx,y[i]))[None].cuda()
            prediction=model(torch.cat(((a-xm)/xs,coords),1))*ys+ym
            if not torch.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
            row=sample_metrics(prediction,target,a)
            row.update(sample=i,geometry=split['geometry_groups'][i]);rows.append(row)
        geos=geometry_rows(rows);metrics=aggregate_geometry(rows)
        differences={k:abs(metrics[k]-source['best_validation'][k]) for k in ('S','velocity_error','pressure_error')}
        if any(v>2e-6 for v in differences.values()):raise ValueError('Adapter differs from original frozen validation metrics')
        div={k:sum(r[k] for r in rows)/150 for k in ('predicted_divergence_rms','reference_divergence_rms')}
        for k,v in div.items():
            if abs(v-source[k+'_mean'])>2e-6:raise ValueError('Diagnostic divergence differs from frozen trainer')
        selected=lock['latency_input_ids']
        host=[torch.from_numpy(np.array(x[i],copy=True))[None] for i in selected]
        device=[v.cuda() for v in host];features=[torch.cat(((v-xm)/xs,coords),1) for v in device]
        calls=dict(model_only=lambda i:model(features[i]),
            gpu_pipeline=lambda i:model(torch.cat(((device[i]-xm)/xs,coords),1))*ys+ym,
            host_to_host=lambda i:(model(torch.cat(((host[i].to('cuda')-xm)/xs,coords),1))*ys+ym).cpu())
        for i in range(15):
            expected=(calls['model_only'](i)*ys+ym).cpu()
            for mode in ('gpu_pipeline','host_to_host'):
                actual=calls[mode](i).cpu()
                if not torch.isfinite(actual).all() or not torch.equal(expected,actual):raise ValueError('Latency wrappers differ')
        timing_checks=[]
        for mode,call in calls.items():
            for j in range(2):output=call(j);torch.cuda.synchronize()
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            for j in range(5):
                torch.cuda.synchronize();t=time.perf_counter();begin.record();output=call(j)
                end.record();torch.cuda.synchronize()
                timing_checks.append(dict(mode=mode,wall_ms=(time.perf_counter()-t)*1000,cuda_event_ms=begin.elapsed_time(end)))
    atomic(out/'validation_rows.json',rows)
    atomic(out/'summary.json',dict(status='FINAL_ADAPTER_VALIDATION_PREFLIGHT_PASSED',
        test_targets_loaded=False,validation_ids=ids,checkpoint_sha256=entry['checkpoint_sha256'],
        authorization_sha256=digest(DEST/'authorization.json'),metrics=metrics,geometry_rows=geos,
        absolute_difference_from_frozen_validation=differences,tolerance=2e-6,divergence=div,
        wrappers_bit_identical=True,wrapper_checks=timing_checks,
        warning='Technical wrapper checks only, not final latency certification or new test evidence.',
        elapsed_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__))
    print('FINAL_ADAPTER_VALIDATION_PREFLIGHT_PASSED',differences,flush=True)


if __name__=='__main__':run()

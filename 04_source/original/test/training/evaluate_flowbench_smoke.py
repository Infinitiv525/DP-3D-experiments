"""Validation-only field diagnostics and resolution scaling timing, not 128^3 accuracy."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from train_flowbench_smoke import SteadyFNO, gauge_target


def divergence_descriptor(velocity, fluid):
    """Central FD in unit-cube index coordinates; no dimensional NS claim."""
    n=fluid.shape[0]
    spacing=1.0/(n-1)
    stencil=fluid[1:-1,1:-1,1:-1].copy()
    for axis in range(3):
        for side in (slice(None,-2),slice(2,None)):
            sl=[slice(1,-1)]*3;sl[axis]=side;stencil &= fluid[tuple(sl)]
    div=(velocity[0,2:,1:-1,1:-1]-velocity[0,:-2,1:-1,1:-1]
         +velocity[1,1:-1,2:,1:-1]-velocity[1,1:-1,:-2,1:-1]
         +velocity[2,1:-1,1:-1,2:]-velocity[2,1:-1,1:-1,:-2])/(2*spacing)
    if not stencil.any():raise ValueError('Empty fluid divergence stencil')
    return float(np.sqrt(np.mean(div[stencil]**2))),int(stencil.sum())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(a.run/'best_model.pt',map_location='cpu',weights_only=False)
    summary=json.loads((a.run/'summary.json').read_text())
    cfg=checkpoint['config']['training'];stats=checkpoint['normalization']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=SteadyFNO(cfg['width'],cfg['layers'],cfg['modes']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    x=np.load(a.data/'X_pilot.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(a.data/'Y_pilot.npy',mmap_mode='r',allow_pickle=False)
    def tensor(v):return torch.tensor(v,device='cuda',dtype=torch.float32).view(1,-1,1,1,1)
    xm,xs=tensor(stats['X']['mean']),tensor(stats['X']['std'])
    ym,ys=tensor(stats['Y']['mean']),tensor(stats['Y']['std'])
    n=x.shape[-1];coords=torch.stack(torch.meshgrid(*[torch.linspace(0,1,n,device='cuda')]*3,indexing='ij'))[None]
    rows=[];timings={};features=None
    with torch.inference_mode():
        for j,i in enumerate(summary['validation_indices']):
            raw=np.array(x[i],copy=True);target=gauge_target(raw,y[i]);fluid=raw[0]>0
            features=torch.cat(((torch.from_numpy(raw)[None].cuda()-xm)/xs,coords),dim=1)
            prediction=(model(features)*ys+ym)[0].cpu().numpy()
            # Gauge-align each prediction on the same evaluation fluid domain.
            prediction[3,fluid]-=prediction[3,fluid].mean(dtype=np.float64)
            e=prediction-target
            rows.append({'sample':i,'fluid_velocity_relative_l2':float(np.linalg.norm(e[:3,fluid])/max(np.linalg.norm(target[:3,fluid]),1e-12)),'fluid_centered_pressure_relative_l2':float(np.linalg.norm(e[3,fluid])/max(np.linalg.norm(target[3,fluid]),1e-12)),'fluid_channel_rmse':np.sqrt(np.mean(e[:,fluid]**2,axis=1)).tolist()})
            reference_div,points=divergence_descriptor(target[:3],fluid)
            predicted_div,_=divergence_descriptor(prediction[:3],fluid)
            rows[-1].update(Re=float(raw[1,0,0,0]),reference_divergence_rms_unit_cube=reference_div,predicted_divergence_rms_unit_cube=predicted_div,divergence_stencil_points=points)
            if j<3:np.savez_compressed(a.output/f'validation_{i}.npz',sdf=raw[0],target=target,prediction=prediction)
        for size in (32,64,128):
            # Features are interpolated solely to measure compute scaling. No fine-grid truth is used.
            inp=torch.nn.functional.interpolate(features,size=(size,)*3,mode='trilinear',align_corners=True)
            for _ in range(50):model(inp)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            gpu=[];wall=[]
            for _ in range(1000):
                s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                started=time.perf_counter();s.record();model(inp);e.record();e.synchronize()
                wall.append((time.perf_counter()-started)*1000);gpu.append(s.elapsed_time(e))
            quant=lambda v:dict(zip(('p50','p95','p99'),np.percentile(v,[50,95,99]).tolist()))
            timings[str(size)]={'gpu_ms':quant(gpu),'synchronized_wall_ms':quant(wall),'max_allocated_bytes':torch.cuda.max_memory_allocated(),'samples':1000,'gpu_samples_ms':gpu,'wall_samples_ms':wall}
            print('TIMING',size,timings[str(size)]['synchronized_wall_ms'],flush=True)
    result={'status':'FLOWBENCH_SMOKE_EVALUATION_COMPLETE','gpu':torch.cuda.get_device_name(),'source_run':str(a.run),'checkpoint_sha256':hashlib.sha256((a.run/'best_model.pt').read_bytes()).hexdigest(),'validation_metrics':rows,'test_evaluated':False,'timings':timings,'warning':'Validation accuracy is 32^3 only. 64^3/128^3 use interpolated inputs for GPU-resident FP32 batch1 timing only, excluding preprocessing, transfer and visualization; no fine-resolution accuracy or physics claim.','production_accepted':False}
    result['validation_accuracy_resolution']=[n,n,n]
    result['warning']=f'Validation accuracy is {n}^3 only. Other timing resolutions use interpolated inputs for GPU-resident FP32 batch1 timing only, excluding preprocessing, transfer and visualization; no fine-resolution accuracy or physics claim.'
    (a.output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

"""New draft-protocol trainer; historical smoke trainer remains untouched."""
import argparse
import hashlib
import json
import random
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from train_flowbench_smoke import SteadyFNO,gauge_target
from flowbench_physics import relative_divergence_penalty,divergence_values
from flowbench_protocol_metrics import weighted_data_loss,fluid_errors,aggregate_geometry,validate_split,TrainValidationOnly,validate_hyperparameters


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(a):
    started=time.perf_counter();cpu=time.process_time()
    a.output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','test_evaluated':False,'pilot_only':True,'history':[]}
    save(a.output/'summary.json',result)
    try:
        cfg=json.loads(a.config.read_text());tc=cfg['training']
        validate_hyperparameters(tc)
        if cfg.get('initial_checkpoint'):raise ValueError('Fresh initialization only')
        audit=json.loads((a.data/'pressure_gauge_audit.json').read_text())
        if audit['status']!='FLOWBENCH_NATIVE_GAUGE64_CACHE_COMPLETE' or audit['source_revision']!=cfg['revision']:
            raise ValueError('Corrected native-gauge cache required')
        if digest(a.data/'split.json')!=audit['split_sha256']:raise ValueError('Split hash mismatch')
        split=json.loads((a.data/'split.json').read_text());validate_split(split)
        train_ids=sorted(split['indices']['train']);val_ids=sorted(split['indices']['validation'])
        if len(train_ids)!=700 or len(val_ids)!=150:raise ValueError('Full train/validation expected')
        rawx=np.load(a.data/'X_pilot.npy',mmap_mode='r',allow_pickle=False)
        rawy=np.load(a.data/'Y_pilot.npy',mmap_mode='r',allow_pickle=False)
        if rawx.shape!=(1000,2,64,64,64) or rawy.shape!=(1000,4,64,64,64):raise ValueError('64^3 required')
        x=TrainValidationOnly(rawx,split);y=TrainValidationOnly(rawy,split)
        def check_time():
            if time.perf_counter()-started>a.max_wall_seconds:raise TimeoutError('Trial time budget exhausted')
        identity={'train_ids':train_ids,'audit_sha256':digest(a.data/'pressure_gauge_audit.json'),
                  'split_sha256':audit['split_sha256'],'method':'float64 canonical sorted train-only whole-grid moments; gauge_target'}
        norm_start=time.perf_counter()
        if a.normalization.exists():
            norm=json.loads(a.normalization.read_text())
            if norm['identity']!=identity:raise ValueError('Normalization cache provenance differs')
            stats=norm['stats']
        else:
            stats={}
            for name,arr in (('X',x),('Y',y)):
                channels=2 if name=='X' else 4
                total=np.zeros(channels,dtype=np.float64);sq=total.copy();count=0
                for i in train_ids:
                    check_time()
                    v=np.asarray(gauge_target(x[i],arr[i]) if name=='Y' else arr[i],dtype=np.float64)
                    total+=v.sum(axis=(1,2,3));sq+=(v*v).sum(axis=(1,2,3));count+=v[0].size
                mean=total/count;std=np.maximum(np.sqrt(np.maximum(sq/count-mean**2,0)),1e-6)
                stats[name]={'mean':mean.tolist(),'std':std.tolist()}
            save(a.normalization,{'identity':identity,'stats':stats})
        result.update(normalization_seconds=time.perf_counter()-norm_start,normalization_train_only=stats,
                      normalization_sha256=digest(a.normalization),normalization_identity=identity)
        seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        def tensor(v):return torch.tensor(v,device='cuda',dtype=torch.float32).view(1,-1,1,1,1)
        xm,xs=tensor(stats['X']['mean']),tensor(stats['X']['std']);ym,ys=tensor(stats['Y']['mean']),tensor(stats['Y']['std'])
        coords=torch.stack(torch.meshgrid(*[torch.linspace(0,1,64,device='cuda')]*3,indexing='ij'))[None]
        def sample(i):
            check_time();raw=x[i];a0=torch.from_numpy(np.array(raw,copy=True))[None].cuda()
            target=torch.from_numpy(gauge_target(raw,y[i]))[None].cuda()
            return torch.cat(((a0-xm)/xs,coords),1),(target-ym)/ys,target,a0[:,:1]>0
        model=SteadyFNO(tc['width'],tc['layers'],tc['modes']).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
        rng=np.random.default_rng(seed);best=float('inf');best_rows=[];training_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(tc['epochs']):
            t=time.perf_counter();model.train();losses=[]
            for i in rng.permutation(train_ids):
                features,target,physical,fluid=sample(int(i));optimizer.zero_grad(set_to_none=True)
                prediction=model(features)
                data_loss=weighted_data_loss(prediction,target,tc['pressure_weight'])
                div=relative_divergence_penalty(prediction*ys+ym,physical,fluid,1e-4)
                loss=data_loss+tc['divergence_weight']*div
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                losses.append(float(loss.detach()))
            torch.cuda.synchronize();train_seconds=time.perf_counter()-t
            t=time.perf_counter();model.eval();rows=[]
            with torch.inference_mode():
                for i in val_ids:
                    features,_,target,fluid=sample(i);prediction=model(features)*ys+ym
                    metrics=fluid_errors(prediction[0],target[0],fluid[0,0])
                    metrics.update(sample=i,geometry=split['geometry_groups'][i],
                        predicted_divergence_rms=float(divergence_values(prediction[:,:3],fluid).square().mean().sqrt()),
                        reference_divergence_rms=float(divergence_values(target[:,:3],fluid).square().mean().sqrt()))
                    if metrics['velocity_below_epsilon'] or metrics['pressure_below_epsilon']:
                        raise ValueError('Degenerate relative-error denominator; revise before freeze')
                    rows.append(metrics)
            score=aggregate_geometry(rows);validation_seconds=time.perf_counter()-t
            entry={'epoch':epoch+1,'train_composite_loss':float(np.mean(losses)),
                   'validation':score,'train_seconds':train_seconds,'validation_seconds':validation_seconds}
            result['history'].append(entry)
            with (a.output/'history.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
            print(json.dumps(entry),flush=True)
            if score['S']<best:
                best=score['S'];best_rows=rows;result['best_epoch']=epoch+1;result['best_validation']=score
                torch.save({'model':model.state_dict(),'config':cfg,'normalization':stats},a.output/'best_model.pt')
                save(a.output/'best_validation_rows.json',rows)
            save(a.output/'summary.json',result)
        result.update(status='FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE',config=cfg,
            training_indices=train_ids,validation_indices=val_ids,config_sha256=digest(a.config),
            split_sha256=audit['split_sha256'],checkpoint_sha256=digest(a.output/'best_model.pt'),
            model_selection='minimum geometry-macro validation S=0.5*E_U+0.5*E_p; earlier epoch on exact tie',
            training_and_validation_seconds=time.perf_counter()-training_start,
            gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            predicted_divergence_rms_mean=float(np.mean([r['predicted_divergence_rms'] for r in best_rows])),
            reference_divergence_rms_mean=float(np.mean([r['reference_divergence_rms'] for r in best_rows])),
            warning=cfg.get('purpose','Technical pilot, not main search. No latency certification or test evaluation.'))
    except Exception:
        result['status']='FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_FAILED';result['error']=traceback.format_exc();print(result['error'],flush=True)
    finally:
        result.update(elapsed_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                      linux_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(a.output/'summary.json',result)
    return result['status']=='FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=240)
    raise SystemExit(0 if run(p.parse_args()) else 1)

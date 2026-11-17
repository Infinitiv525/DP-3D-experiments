"""Supervised steady 3D FNO wiring test; no time stepping or physical acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fno_model import FNOBlock3d
from flowbench_physics import relative_divergence_penalty


def gauge_target(features, target):
    """Remove each field's arbitrary pressure constant on SDF-positive points."""
    result = np.array(target, dtype=np.float32, copy=True)
    fluid = features[0] > 0
    if not fluid.any():
        raise ValueError('No fluid points')
    result[3, fluid] -= result[3, fluid].mean(dtype=np.float64)
    result[3, ~fluid] = 0
    return result


class SteadyFNO(nn.Module):
    def __init__(self, width=12, layers=2, modes=4):
        super().__init__()
        self.lift = nn.Conv3d(5, width, 1)
        self.blocks = nn.ModuleList([FNOBlock3d(width, (modes, modes, modes)) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv3d(width, width*2, 1), nn.GELU(), nn.Conv3d(width*2, 4, 1))

    def forward(self, features):
        # Nonperiodic cavity: symmetric padding on every spatial axis, no inlet enforcement.
        x = self.lift(features)
        x = nn.functional.pad(x, (4, 4, 4, 4, 4, 4))
        for block in self.blocks:
            x = block(x)
        return self.project(x[..., 4:-4, 4:-4, 4:-4])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    audit = json.loads((args.data / cfg.get('cache_audit', 'audit.json')).read_text())
    accepted = audit['status'].startswith('STRUCTURAL_AUDIT_COMPLETE') or audit['status'] in ('FLOWBENCH_UNIFORM64_CACHE_COMPLETE','FLOWBENCH_NATIVE_GAUGE64_CACHE_COMPLETE')
    if audit.get('revision',audit.get('source_revision')) != cfg['revision'] or not accepted:
        raise RuntimeError('Pinned successful structural audit required')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke test requires CUDA')
    args.output.mkdir(parents=True, exist_ok=False)
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    x = np.load(args.data / 'X_pilot.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(args.data / 'Y_pilot.npy', mmap_mode='r', allow_pickle=False)
    n=cfg['pilot_resolution']
    if x.shape!=(1000,2,n,n,n) or y.shape!=(1000,4,n,n,n):
        raise ValueError('Cache shapes differ from configured pilot')
    split_path = args.data / 'split.json'
    split = json.loads(split_path.read_text())['indices']
    train_cfg = cfg['training']
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(split['train'])[:train_cfg['max_train_samples']].tolist()
    val_ids = rng.permutation(split['validation'])[:train_cfg['max_validation_samples']].tolist()
    # No test indices/target values used, including for normalization or latency.
    stats = {}
    for name, arr in [('X', x), ('Y', y)]:
        total = np.zeros(arr.shape[1], dtype=np.float64)
        sq = total.copy(); count = 0
        for i in train_ids:
            v = np.asarray(gauge_target(x[i], arr[i]) if name == 'Y' else arr[i], dtype=np.float64)
            total += v.sum(axis=(1, 2, 3)); sq += (v*v).sum(axis=(1, 2, 3)); count += v[0].size
        mean = total / count
        std = np.maximum(np.sqrt(np.maximum(sq/count - mean**2, 0)), 1e-6)
        stats[name] = {'mean': mean.tolist(), 'std': std.tolist()}
    def tensor(v):
        return torch.tensor(v, device='cuda', dtype=torch.float32).view(1, -1, 1, 1, 1)
    xm, xs = tensor(stats['X']['mean']), tensor(stats['X']['std'])
    ym, ys = tensor(stats['Y']['mean']), tensor(stats['Y']['std'])
    n = x.shape[-1]
    axes = [torch.linspace(0, 1, n, device='cuda') for _ in range(3)]
    coords = torch.stack(torch.meshgrid(*axes, indexing='ij'))[None]
    def sample(i):
        a = torch.from_numpy(np.array(x[i], copy=True))[None].to('cuda')
        b = torch.from_numpy(gauge_target(x[i], y[i]))[None].to('cuda')
        return torch.cat(((a-xm)/xs, coords), dim=1), (b-ym)/ys, a[:,:1]>0
    model = SteadyFNO(train_cfg['width'], train_cfg['layers'], train_cfg['modes']).cuda()
    initial_provenance=None
    if cfg.get('initial_checkpoint'):
        initial=Path(cfg['initial_checkpoint'])
        source=torch.load(initial,map_location='cpu',weights_only=False)
        source_summary=json.loads((initial.parent/'summary.json').read_text())
        for name in ('X','Y'):
            for key in ('mean','std'):
                if not np.allclose(stats[name][key],source['normalization'][name][key],rtol=1e-6,atol=1e-8):
                    raise ValueError('Initial model and current normalization differ')
        for key in ('width','layers','modes'):
            if train_cfg[key]!=source['config']['training'][key]:raise ValueError('Architecture changed')
        if set(train_ids)!=set(source_summary['training_indices']) or set(val_ids)!=set(source_summary['validation_indices']):
            raise ValueError('Refinement must preserve source train/validation membership')
        model.load_state_dict(source['model'])
        initial_provenance={'path':str(initial),'sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'source_best_epoch':source_summary['best_epoch'],'optimizer_reset':True}
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'])
    history = []; best = float('inf'); started = time.perf_counter()
    for epoch in range(train_cfg['epochs']):
        model.train(); losses = []; composite_losses=[]; divergence_losses=[]
        for i in rng.permutation(train_ids):
            a, b, fluid = sample(int(i)); optimizer.zero_grad(set_to_none=True)
            prediction=model(a)
            data_loss = (prediction-b).square().mean()
            div_loss=data_loss.new_zeros(())
            if train_cfg.get('measure_divergence',False):
                div_loss=relative_divergence_penalty(prediction*ys+ym,b*ys+ym,fluid,train_cfg.get('reference_divergence_mse_floor',1e-4))
            loss=data_loss+train_cfg.get('divergence_weight',0.0)*div_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(data_loss.detach()));composite_losses.append(float(loss.detach()));divergence_losses.append(float(div_loss.detach()))
        model.eval(); errors = []; baselines = []; channels = []
        with torch.inference_mode():
            for i in val_ids:
                a, b, _ = sample(i); err = model(a)-b
                errors.append(float(err.square().mean())); baselines.append(float(b.square().mean()))
                channels.append((err*ys).square().mean(dim=(0, 2, 3, 4)).cpu().numpy())
        entry = {'epoch': epoch+1, 'train_standardized_mse': float(np.mean(losses)), 'validation_standardized_mse': float(np.mean(errors)), 'validation_train_mean_baseline_mse': float(np.mean(baselines)), 'validation_whole_grid_channel_rmse_raw_units': np.sqrt(np.mean(channels, axis=0)).tolist()}
        entry.update(train_composite_loss=float(np.mean(composite_losses)),train_relative_divergence_penalty=float(np.mean(divergence_losses)))
        history.append(entry); print(json.dumps(entry), flush=True)
        with (args.output / 'history.jsonl').open('a') as log:
            log.write(json.dumps(entry,allow_nan=False)+'\n')
        if entry['validation_standardized_mse'] < best:
            best = entry['validation_standardized_mse']
            torch.save({'model':model.state_dict(), 'config':cfg, 'normalization':stats}, args.output / 'best_model.pt')
    model.load_state_dict(torch.load(args.output / 'best_model.pt', weights_only=False)['model'])
    model.eval(); a, _, _ = sample(val_ids[0])
    gpu_ms = []; wall_ms = []
    with torch.inference_mode():
        for _ in range(50): model(a)
        torch.cuda.synchronize()
        for _ in range(1000):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter(); start.record(); model(a); end.record(); end.synchronize()
            wall_ms.append((time.perf_counter()-wall)*1000); gpu_ms.append(start.elapsed_time(end))
    def quantiles(values):
        return dict(zip(('p50','p95','p99'), np.percentile(values, [50,95,99]).tolist()))
    result = {'status':'FLOWBENCH_SUPERVISED_SMOKE_COMPLETE', 'revision':cfg['revision'], 'config':cfg, 'seed':seed, 'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(), 'training_indices':train_ids, 'validation_indices':val_ids, 'test_evaluated':False, 'normalization_train_only':stats, 'history':history, 'resolution':[n,n,n], 'latency_gpu_resident_fp32_batch1_ms':quantiles(gpu_ms), 'latency_synchronized_wall_ms':quantiles(wall_ms), 'latency_samples':1000, 'latency_excludes':['loading','input_normalization','host_device_copy','visualization'], 'timing_warning':'synchronized isolated latency, not end-to-end or native 128^3 certification', 'latency_gpu_samples_ms':gpu_ms, 'latency_wall_samples_ms':wall_ms, 'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'numpy':np.__version__, 'cuda':torch.version.cuda, 'elapsed_seconds':time.perf_counter()-started, 'physics_loss_enabled':False, 'metrics_warning':'whole-grid including solids; fluid polarity, gauge and units require separate audit; stride cache is pilot only', 'production_accepted':False}
    result['pressure_target_convention'] = 'Per-sample mean pressure removed only on SDF>0 points; solid pressure set to zero. Each validation target uses its own mean, no train/test statistics.'
    result['status']=cfg.get('run_stage',result['status'])
    result['initial_checkpoint_provenance']=initial_provenance
    result['physics_loss_enabled']=train_cfg.get('divergence_weight',0.0)>0
    result['physics_penalty_convention']='central FD unit-cube coordinates, all-fluid stencil; predicted divergence MSE / max(reference divergence MSE, configured floor); not full Navier-Stokes residual'
    result['best_epoch']=min(history,key=lambda row:row['validation_standardized_mse'])['epoch']
    result['model_selection']='minimum validation standardized whole-grid MSE, fixed before this run'
    result['metrics_warning'] = f'Whole-grid including solids; pressure channel RMSE uses gauge-centered targets, not raw absolute pressure. Resolution {n}^3; '+cfg.get('cache_description','stride-4 technical cache')+'; no physics acceptance.'
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output / 'normalization.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()

"""FP32 latency of an existing corrected-cache checkpoint. Never loads target Y."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from train_flowbench_smoke import SteadyFNO


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(rows, thresholds):
    result = {'n': len(rows)}
    for key in ('wall_ms', 'cuda_event_ms'):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        result[key] = {'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)),
                       'p99': float(np.percentile(values, 99)), 'min': float(values.min()), 'max': float(values.max()),
                       'mean': float(values.mean()), 'percentile_method': 'numpy linear'}
    result['wall_fraction_at_or_below'] = {str(t): float(np.mean([r['wall_ms'] <= t for r in rows])) for t in thresholds}
    return result


def run(a):
    started = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    cfg = read(a.config); source = a.root / cfg['checkpoint_relative']; data = a.root / cfg['data_relative']
    summary = read(source / 'summary.json'); audit = read(data / 'pressure_gauge_audit.json')
    if summary['status'] != 'FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE' or summary['test_evaluated']:
        raise ValueError('Successful train/validation-only source required')
    if digest(source / 'best_model.pt') != summary['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    if digest(data / 'split.json') != cfg['split_sha256'] or audit['split_sha256'] != cfg['split_sha256']:
        raise ValueError('Split hash mismatch')
    split = read(data / 'split.json'); selected = []; seen = set()
    for i in sorted(split['indices']['validation']):
        group = split['geometry_groups'][i]
        if group not in seen: selected.append(i); seen.add(group)
        if len(selected) == cfg['validation_geometries']: break
    if len(selected) != cfg['validation_geometries']: raise ValueError('Not enough validation geometries')
    raw = np.load(data / 'X_pilot.npy', mmap_mode='r', allow_pickle=False)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(source / 'best_model.pt', map_location='cpu', weights_only=True)
    if checkpoint['config'] != summary['config'] or checkpoint['normalization'] != summary['normalization_train_only']:
        raise ValueError('Checkpoint provenance differs')
    tc = checkpoint['config']['training']; stats = checkpoint['normalization']
    model = SteadyFNO(tc['width'], tc['layers'], tc['modes']).cuda().eval(); model.load_state_dict(checkpoint['model'])
    def tensor(v): return torch.tensor(v, device='cuda', dtype=torch.float32).view(1, -1, 1, 1, 1)
    xm, xs = tensor(stats['X']['mean']), tensor(stats['X']['std'])
    ym, ys = tensor(stats['Y']['mean']), tensor(stats['Y']['std'])
    coordinates = torch.stack(torch.meshgrid(*[torch.linspace(0, 1, 64, device='cuda')] * 3, indexing='ij'))[None]
    host = [torch.from_numpy(np.array(raw[i], copy=True))[None] for i in selected]
    device = [value.cuda() for value in host]
    features = [torch.cat(((value - xm) / xs, coordinates), 1) for value in device]
    calls = {'model_only': lambda i: model(features[i]),
             'gpu_pipeline': lambda i: model(torch.cat(((device[i] - xm) / xs, coordinates), 1)) * ys + ym,
             'host_to_host': lambda i: (model(torch.cat(((host[i].to('cuda') - xm) / xs, coordinates), 1)) * ys + ym).cpu()}
    records = []; checks = []
    with torch.inference_mode():
        # Confirm the three timing wrappers compute the same result before timing.
        for i in range(len(selected)):
            expected = (calls['model_only'](i) * ys + ym).cpu()
            for mode in ('gpu_pipeline', 'host_to_host'):
                actual = calls[mode](i).cpu()
                if not torch.isfinite(actual).all() or not torch.equal(actual, expected): raise ValueError('Timing modes changed outputs')
            checks.append({'sample': selected[i], 'wrappers_bit_identical': True})
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for round_index in range(cfg['rounds']):
            # Rotate order to avoid giving any mode the same thermal/time position.
            order = cfg['mode_order'][round_index:] + cfg['mode_order'][:round_index]
            for mode in order:
                call = calls[mode]
                for j in range(cfg['warmup_per_mode_per_round']):
                    output = call(j % len(selected)); torch.cuda.synchronize()
                begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                for j in range(cfg['measured_per_mode_per_round']):
                    if time.perf_counter() - started > cfg['max_process_seconds']: raise TimeoutError('Latency audit wall cap')
                    i = j % len(selected); torch.cuda.synchronize()
                    t = time.perf_counter(); begin.record(); output = call(i); end.record(); torch.cuda.synchronize()
                    wall = (time.perf_counter() - t) * 1000
                    records.append({'round': round_index + 1, 'mode': mode, 'sample': selected[i],
                                    'wall_ms': wall, 'cuda_event_ms': begin.elapsed_time(end)})
                save(a.output / 'measurements.json', records)
                print('LATENCY_MODE_COMPLETE', round_index + 1, mode, flush=True)
    per_mode = {mode: summarize([r for r in records if r['mode'] == mode], cfg['thresholds_ms']) for mode in cfg['mode_order']}
    per_round = {mode: {str(k): summarize([r for r in records if r['mode'] == mode and r['round'] == k], cfg['thresholds_ms'])
                for k in range(1, cfg['rounds'] + 1)} for mode in cfg['mode_order']}
    for mode in per_mode:
        per_mode[mode]['all_round_wall_p99_le_5ms'] = all(v['wall_ms']['p99'] <= 5 for v in per_round[mode].values())
    gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,compute_mode', '--format=csv'], capture_output=True, text=True, timeout=10)
    result = {'status': 'FLOWBENCH_CURRENT_CHECKPOINT_LATENCY_AUDIT_COMPLETE', 'technical_pilot': True,
              'test_evaluated': False, 'test_targets_loaded': False, 'config': cfg, 'selected_validation_ids': selected,
              'checkpoint_sha256': summary['checkpoint_sha256'], 'split_sha256': cfg['split_sha256'],
              'config_sha256': digest(a.config), 'normalization_sha256': summary['normalization_sha256'],
              'source_validation_S': summary['best_validation']['S'], 'output_equivalence_checks': checks,
              'gpu': torch.cuda.get_device_name(), 'gpu_info': gpu_info.stdout, 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'numpy': np.__version__, 'per_mode': per_mode, 'per_round': per_round,
              'elapsed_seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'warning': 'One checkpoint, one GPU allocation. No accuracy/test certification, no statistical independence claim for repeated timings; timing instrumentation included. Host GPU co-tenancy and clock variability remain limitations.'}
    save(a.output / 'summary.json', result)
    lines = ['LATENCIA AKTUÁLNEHO MODELU - FP32', result['status'], 'Batch1,64^3,tri kolá po1000meraní/režim. Bez disk IO, SDF generovania a vizualizácie.']
    for mode, row in per_mode.items():
        lines.append(f"{mode}: wall p50={row['wall_ms']['p50']:.4f}ms, p95={row['wall_ms']['p95']:.4f}ms, p99={row['wall_ms']['p99']:.4f}ms; všetky3kolá p99<=5ms: {row['all_round_wall_p99_le_5ms']}")
    lines.append(result['warning']); (a.output / 'vysledok.txt').write_text('\n'.join(lines) + '\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())

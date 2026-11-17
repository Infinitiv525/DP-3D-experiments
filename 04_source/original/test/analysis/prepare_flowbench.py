"""Pinned remote-only download, streaming audit, and explicitly coarse pilot cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(root, config, role):
    item = config['files'][role]
    final = root / Path(item['path']).name
    partial = final.with_suffix('.npz.part')
    if not final.exists():
        for attempt in range(5):
            start = partial.stat().st_size if partial.exists() else 0
            if start == item['bytes']:
                break
            if start > item['bytes']:
                raise ValueError('Partial file too large; inspect manually')
            url = (f"https://huggingface.co/datasets/{config['repository']}/resolve/"
                   f"{config['revision']}/{item['path']}?download=true&nonce={time.time_ns()}")
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-', 'User-Agent': 'dissertation-flowbench-audit'})
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    if start and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                        raise RuntimeError('Server did not honor resume range')
                    with partial.open('ab') as f:
                        last = start
                        while block := response.read(8 << 20):
                            f.write(block)
                            if f.tell() - last >= 256 << 20:
                                last = f.tell()
                                print(role, 'download_bytes', last, flush=True)
                if partial.stat().st_size == item['bytes']:
                    break
            except Exception as exc:
                print('download retry', role, attempt, type(exc).__name__, str(exc), flush=True)
                if attempt == 4:
                    raise
                time.sleep(3)
        if partial.stat().st_size != item['bytes']:
            raise RuntimeError('Incomplete download')
        if digest(partial) != item['sha256']:
            raise RuntimeError('Downloaded SHA256 differs from pinned LFS SHA256')
        partial.rename(final)
    if final.stat().st_size != item['bytes'] or digest(final) != item['sha256']:
        raise RuntimeError('Existing archive failed integrity check; not overwriting')
    print(role, 'SHA256_VERIFIED', flush=True)
    return final


def header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f'Unsupported npy header {version}')
    if order or dtype.hasobject or dtype.kind != 'f':
        raise ValueError('Require C-order numeric floating array, no pickle')
    return shape, dtype


def read_exact(stream, count):
    data = bytearray(count)
    view = memoryview(data)
    done = 0
    while done < count:
        part = stream.read(min(count - done, 8 << 20))
        if not part:
            raise EOFError('Truncated numpy payload')
        view[done:done+len(part)] = part
        done += len(part)
    return data


def audit_archive(archive, role, out, resolution):
    rows = []
    groups = []
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('.npy')]
        if len(names) != 1:
            raise ValueError(f'Expected one array, got {names}')
        with z.open(names[0]) as stream:
            shape, dtype = header(stream)
            expected_channels = 2 if role == 'X' else 4
            if len(shape) != 5 or shape[1:] != (expected_channels, 128, 128, 128):
                raise ValueError(f'Unexpected {role} shape {shape}; do not guess channel axes')
            if shape[0] != 1000:
                raise ValueError(f'Pinned card and archive disagree on samples: {shape}')
            cache = np.lib.format.open_memmap(out / f'{role}_pilot.npy', mode='w+', dtype='float32', shape=(shape[0], shape[1], resolution, resolution, resolution))
            step = 128 // resolution
            count = int(np.prod(shape[1:])) * dtype.itemsize
            for i in range(shape[0]):
                sample = np.frombuffer(read_exact(stream, count), dtype=dtype).reshape(shape[1:])
                if not np.isfinite(sample).all():
                    raise ValueError(f'Non-finite {role} sample {i}')
                rows.append({'sample': i, 'min': sample.min(axis=(1, 2, 3)).tolist(), 'max': sample.max(axis=(1, 2, 3)).tolist()})
                if role == 'X':
                    sdf = sample[0]
                    g = sdf <= 0
                    if not g.any() or g.all():
                        raise ValueError('SDF does not distinguish inside and outside')
                    # Group exact voxelized geometry regardless of Re or SDF values.
                    groups.append(hashlib.sha256(np.packbits(g.astype(bool)).tobytes()).hexdigest())
                    if np.ptp(sample[1]) > max(1e-5, abs(float(sample[1].mean())) * 1e-5):
                        raise ValueError('Re channel is not spatially constant; inspect semantics')
                    if not 10 <= float(sample[1].mean()) <= 1000:
                        raise ValueError('Re outside range declared by updated author paper')
                cache[i] = sample[:, ::step, ::step, ::step]
                if i % 50 == 0:
                    print(role, 'audited_samples', i + 1, flush=True)
            if stream.read(1):
                raise ValueError('Unexpected bytes after numpy payload')
            cache.flush()
    return {'member': names[0], 'shape': list(shape), 'dtype': str(dtype), 'sample_ranges': rows}, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    out = root / 'pilot32'
    if (out / 'audit.json').exists():
        raise RuntimeError('Prepared audit already exists; inspect/reuse instead of overwriting')
    out.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 40_000_000_000:
        raise RuntimeError('Less than 40 GB filesystem free; project quota also needs checking')
    save(root / 'source_manifest.json', config)
    for name in ('README.md', 'info.txt'):
        target = root / name
        if not target.exists():
            url = f"https://huggingface.co/datasets/{config['repository']}/resolve/{config['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=60) as r:
                target.write_bytes(r.read(1 << 20))
    report = {'status': 'AUDIT_IN_PROGRESS', 'revision': config['revision'], 'license': config['license'], 'arrays': {}}
    groups = None
    for role in ('X', 'Y'):
        archive = download(root, config, role)
        report['arrays'][role], found = audit_archive(archive, role, out, config['pilot_resolution'])
        if role == 'X':
            groups = found
        save(out / 'audit_progress.json', report)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 20:
        raise ValueError('Too few exact geometry groups for the proposed split')
    rng = np.random.default_rng(config['seed'])
    rng.shuffle(unique_groups)
    n = len(unique_groups)
    labels = {g: ('train' if i < int(.7*n) else 'validation' if i < int(.85*n) else 'test') for i, g in enumerate(unique_groups)}
    split = {key: [i for i, g in enumerate(groups) if labels[g] == key] for key in ('train', 'validation', 'test')}
    if min(map(len, split.values())) < 2:
        raise ValueError('Insufficient samples in a split')
    save(out / 'split.json', {'seed': config['seed'], 'method': config['split'], 'geometry_groups': groups, 'indices': split, 'near_duplicate_geometry_audit_pending': True})
    report.update(status='STRUCTURAL_AUDIT_COMPLETE_NOT_PHYSICS_ACCEPTANCE', exact_geometry_groups=n, split_counts={k:len(v) for k,v in split.items()}, cache_method='stride_4_no_interpolation_or_averaging_pilot_only', channel_semantics_source='updated author paper arxiv:2503.17289 section 2.1: X=SDF,Re; Y=u,v,w,p; SDF signs and Re invariants checked; fluid polarity still to be checked against output fields', physics_training_enabled=False, production_accepted=False)
    for role in ('X', 'Y'):
        report[f'{role}_pilot_sha256'] = digest(out / f'{role}_pilot.npy')
    save(out / 'audit.json', report)
    print(report['status'], report['split_counts'], flush=True)


if __name__ == '__main__':
    main()

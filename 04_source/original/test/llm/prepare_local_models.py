"""Pinned, license-aware download and SHA256 verification; no model execution."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    args.models.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=False)
    selected = []
    total = 0
    patterns = ('*.safetensors', '*.json', '*.model', '*.jinja', '*.txt',
                '*.tiktoken', 'README.md', 'LICENSE*', 'NOTICE*')
    import fnmatch
    for spec in cfg['models']:
        info = api.model_info(spec['repo'], revision=spec['revision'], files_metadata=True)
        if info.sha != spec['revision'] or info.gated:
            raise RuntimeError('Pinned ungated model required: ' + spec['repo'])
        card = info.card_data.to_dict() if info.card_data is not None else {}
        if card.get('license') != spec['license']:
            raise RuntimeError('License differs from approved manifest')
        files = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, pat) for pat in patterns)]
        size = sum(f.size or 0 for f in files)
        if not any(f.rfilename.endswith('.safetensors') for f in files):
            raise RuntimeError('No weights')
        total += size
        selected.append((spec, files, size))
    if total > cfg['download_max_bytes']:
        raise RuntimeError('Download size exceeds hard cap')
    save(args.output/'plan.json', {'status':'DOWNLOAD_PLAN_VERIFIED','total_bytes':total,
         'models':[{'spec':s,'download_bytes':n} for s, _, n in selected]})
    results = []
    for spec, files, size in selected:
        start = time.perf_counter()
        target = args.models/spec['label']/spec['revision']
        print('DOWNLOAD_START', spec['label'], size, flush=True)
        snapshot_download(repo_id=spec['repo'], revision=spec['revision'], token=False,
                          local_dir=target, allow_patterns=patterns, max_workers=4)
        checked = []
        for f in files:
            path = target/f.rfilename
            if f.size is not None and path.stat().st_size != f.size:
                raise RuntimeError('Size mismatch: '+f.rfilename)
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8<<20), b''):
                    h.update(chunk)
            digest = h.hexdigest()
            expected = (f.lfs.get('sha256') if isinstance(f.lfs,dict) else getattr(f.lfs,'sha256',None)) if f.lfs else None
            if f.lfs and not expected:
                raise RuntimeError('Missing official LFS hash: '+f.rfilename)
            if expected and digest != expected:
                raise RuntimeError('LFS SHA256 mismatch: '+f.rfilename)
            checked.append({'file':f.rfilename,'bytes':path.stat().st_size,
                            'sha256':digest,'official_lfs_sha256':expected})
        record = {'status':'LOCAL_MODEL_VERIFIED','spec':spec,'path':str(target),
                  'download_bytes':size,'download_and_verification_seconds':time.perf_counter()-start,
                  'files':checked}
        save(args.output/(spec['label']+'.json'),record)
        results.append(record)
        print('MODEL_VERIFIED', spec['label'], flush=True)
    save(args.output/'summary.json', {'status':'LOCAL_MODELS_DOWNLOAD_COMPLETE',
         'total_bytes':total,'models':results,'no_test_or_cfd_data_used':True})


if __name__ == '__main__':
    main()

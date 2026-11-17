"""Preserve local research files byte-for-byte in an isolated Git index."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiment_3d_llm_flow'
AUDIT = EXP / 'test/github_main_20261105_v40'
BRANCH = 'refs/heads/codex/local-artifacts-main-20261105'
SECRET = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}')


def git(args, data=None, env=None):
    return subprocess.run(['git', '-c', 'core.longpaths=true', '-c', 'core.quotepath=false', *args], cwd=ROOT, input=data, capture_output=True,
                          check=True, env=env).stdout


def run():
    AUDIT.mkdir(exist_ok=True)
    internal = AUDIT / 'internal'
    internal.mkdir(exist_ok=True)
    previous = subprocess.run(['git','rev-parse','--verify',BRANCH],cwd=ROOT,capture_output=True)
    base = previous.stdout.decode().strip() if previous.returncode == 0 else git(['rev-parse','origin/codex/pinn-backup-20261105']).decode().strip()
    git(['merge-base','--is-ancestor','origin/main',base])
    excluded = []
    paths = []
    manifest_path = AUDIT / 'local_files_manifest.json'
    for folder, dirs, files in os.walk(EXP):
        keep = []
        for name in dirs:
            p = Path(folder) / name
            if name in {'review_checkout','.git','__pycache__','node_modules','.venv'} or p == internal:
                excluded.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),reason='Generated checkout, cache, or Git staging metadata'))
            else:
                keep.append(name)
        dirs[:] = keep
        for name in files:
            p = Path(folder) / name
            if p == manifest_path:
                continue
            if p.is_symlink():
                excluded.append(dict(path=str(p.relative_to(ROOT)),reason='Symlink; original preserved locally'))
                continue
            paths.append(p)
    paths.append(ROOT / 'experiment_3d_llm_flow.zip')
    paths = sorted(paths)
    records = []
    binary = {'.png','.jpg','.jpeg','.mp4','.pdf','.pt','.npz','.npy','.zip','.gif','.vtu','.vtp','.ico'}
    for p in paths:
        data = p.read_bytes()
        if len(data) >= 100 * 1024**2:
            raise ValueError('Use release artifact for large file: ' + str(p))
        if p.suffix.lower() not in binary and SECRET.search(data):
            raise ValueError('Credential-like content, review without printing it: ' + str(p))
        records.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(source='Local research files, existing thesis files already in Git',
                    files=records, file_count=len(records), bytes=sum(r['bytes'] for r in records),
                    exclusions=excluded, byte_preservation='Nested .gitattributes * -text disables newline conversion',
                    self_hash_excluded=True, original_working_tree_and_index_unchanged=True,
                    no_hpc_archive_or_large_models_downloaded=True, parent=base)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    paths.append(manifest_path)
    relative = [p.relative_to(ROOT).as_posix() for p in paths]
    env = dict(os.environ, GIT_INDEX_FILE=str(internal / 'stage.index'))
    git(['read-tree',base],env=env)
    hashes = git(['hash-object','-w','--no-filters','--stdin-paths'],data=('\n'.join(relative)+'\n').encode(),env=env).decode().splitlines()
    if len(hashes) != len(relative):
        raise ValueError('Git blob count mismatch')
    attributes = git(['hash-object','-w','--stdin'],data=b'* -text\n',env=env).decode().strip()
    ignore = git(['hash-object','-w','--stdin'],data=b'**/__pycache__/\n**/*.pyc\ntest/github_backup_20261105_v38/review_checkout/\ntest/github_main_20261105_v40/internal/\n',env=env).decode().strip()
    entries = ''.join('100644 '+sha+'\t'+path+'\0' for sha,path in zip(hashes,relative))
    entries += '100644 '+attributes+'\texperiment_3d_llm_flow/.gitattributes\0'
    entries += '100644 '+ignore+'\texperiment_3d_llm_flow/.gitignore\0'
    git(['update-index','-z','--index-info'],data=entries.encode(),env=env)
    tree = git(['write-tree'],env=env).decode().strip()
    changed = git(['diff-tree','--name-only','-r',base,tree],env=env).decode().splitlines()
    if any(not (p.startswith('experiment_3d_llm_flow/') or p=='experiment_3d_llm_flow.zip') for p in changed):
        raise ValueError('Unexpected change outside authorized research files')
    if not changed:
        print('NO_CHANGES')
        return
    commit = git(['commit-tree',tree,'-p',base],data=b'Preserve local research files, 3D results and PINN architecture documentation\n',env=env).decode().strip()
    git(['update-ref',BRANCH,commit,base if previous.returncode == 0 else '0'*40])
    (internal / 'prepared_commit.json').write_text(json.dumps(dict(commit=commit,parent=base,branch=BRANCH,
        file_count=len(records),bytes=manifest['bytes'],tree=tree,changed_paths=len(changed)),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(commit=commit,parent=base,files=len(records),bytes=manifest['bytes'],changed_paths=len(changed)),indent=2))


if __name__ == '__main__':
    run()

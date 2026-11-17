"""Prepare NEW paired TRAIN CFD cases and validate their terminal schedule."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK=ROOT/'results/native_expansion/cylinder_refinement_20261023_v1'


def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify():
    freeze=read(WORK/'source_freeze.json')
    for p,sha in freeze['files'].items():
        if digest(Path(p))!=sha:raise ValueError('Quality source changed: '+p)
    return read(ROOT/'test/config/native_cylinder_refinement_20261023_v1.json')
def item_for(index):return next(x for x in verify()['missing_long_window_families'] if x['index']==index)
def paths(index,branch,job):
    if branch not in ['coarse','fine'] or not re.fullmatch(r'[0-9]+',str(job)):
        raise ValueError('Explicit branch and numeric Slurm array identity required')
    item=item_for(index);name=f"{item['scenario_id']}_{branch}_{job}"
    return ROOT/'dataset_a/v2/cylinder_refinement_20261023_v1'/name,WORK/'family_runs'/name
def scenario(sid):
    with (ROOT/'test/config/dataset_a_v2_scenarios.csv').open(newline='') as f:
        row=next(x for x in csv.DictReader(f) if x['scenario_id']==sid)
    if row['split']!='train' or row['test_locked']!='false':raise PermissionError('TRAIN geometry only')
    return row


def prepare(index,branch,job,source):
    cfg=verify();item=item_for(index);row=scenario(item['scenario_id'])
    expected=f"{item['scenario_id']}_solvgate_{job}_{index}_native_refined"
    if branch!='coarse':raise ValueError('Only new refinement branch allowed')
    if source!=ROOT/'dataset_a/v2/gates/solver'/expected:raise ValueError('Unexpected source case')
    case,out=paths(index,branch,job)
    if case.exists() or out.exists():raise RuntimeError('Never overwrite an existing CFD attempt')
    case.parent.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=False)
    for name in ['0','constant','system']:shutil.copytree(source/name,case/name)
    u=float(row['u_inlet_m_s']);length=float(row['reference_length_m'])
    text=(ROOT/'test/cases/dataset_a_v2/controlDict_fidelity.template').read_text()
    values={'__END_TIME__':2*length/u,'__INITIAL_DT__':.0001*length/u,
            '__MAX_DT__':.02*length/u,'__WRITE_INTERVAL__':.4*length/u,'__MAX_CO__':.2}
    for k,v in values.items():
        if text.count(k)!=1:raise ValueError('Unexpected controlDict template')
        text=text.replace(k,f'{v:.14g}')
    for old,new in [('magUInf 0.02375;',f'magUInf {u:.14g};'),('lRef 0.01;',f'lRef {length:.14g};'),
                    ('Aref 0.0001;',f'Aref {length*length:.14g};')]:
        if text.count(old)!=1:raise ValueError('Force normalization template differs')
        text=text.replace(old,new)
    (case/'system/controlDict').write_text(text)
    save(out/'metadata.json',dict(scenario=row,branch=branch,source_case=str(source),case_dir=str(case),
        solve_tau=[0,40],coarse_policy=item,fine_policy=cfg['fine_configuration'],source_freeze_sha256=digest(WORK/'source_freeze.json'),
        purpose='TRAIN family reference-quality screen only, no ML test and no automatic production acceptance'))
    print(str(case),str(out),sep='\n')


def segment(index,branch,job,tau):
    case,out=paths(index,branch,job);row=read(out/'metadata.json')['scenario'];end=tau*float(row['reference_length_m'])/float(row['u_inlet_m_s'])
    p=case/'system/controlDict';s=p.read_text()
    s,n=re.subn(r'(?m)^endTime\s+[^;]+;',f'endTime {end:.14g};',s);assert n==1
    s,n=re.subn(r'(?m)^startFrom\s+[^;]+;','startFrom latestTime;',s);assert n==1
    p.write_text(s)


def finish(index,branch,job,wall_seconds):
    cfg=verify();case,out=paths(index,branch,job);meta=read(out/'metadata.json');row=meta['scenario']
    log=(out/'log.pimpleFoam').read_text()
    if len(re.findall(r'(?m)^End$',log))!=2:raise ValueError('Both solver segments must finish')
    co=[float(x) for x in re.findall(r'Courant Number mean:\s*\S+\s+max:\s*(\S+)',log)]
    continuity=[float(x) for x in re.findall(r'cumulative\s*=\s*([^\s,]+)',log)]
    if not co or not continuity or not max(co)<=.55 or not abs(continuity[-1])<1e-5:
        raise ValueError('Frozen numerical gate failed')
    u=float(row['u_inlet_m_s']);length=float(row['reference_length_m']);available=[]
    for p in (case/'processor0').iterdir():
        try:t=float(p.name)
        except ValueError:continue
        if p.is_dir():available.append((t,p.name))
    times=[]
    for k in range(101):
        tau=k*.4;expected=tau*length/u
        matches=[name for t,name in available if abs(t-expected)<1e-6]
        if len(matches)!=1:raise ValueError('Missing/ambiguous retained CFD time')
        name=matches[0]
        for rank in range(32):
            p=case/f'processor{rank}'/name
            if any(not (p/f).exists() and not (p/(f+'.gz')).exists() for f in ['U','p']):
                raise ValueError('Incomplete parallel field output')
        times.append(dict(tau=tau,time_s=expected,directory=name))
    save(out/'retained_times.json',times)
    save(out/'summary.json',dict(status='NATIVE_FAMILY_LONG_WINDOW_SOLVER_COMPLETE',scenario_id=row['scenario_id'],
        branch=branch,case_dir=str(case),time_states=101,tau_range=[0,40],max_courant=max(co),
        final_cumulative_continuity=continuity[-1],wall_seconds=wall_seconds,mpi_ranks=32,
        production_accepted=False,model_test_metrics_computed=False,source_freeze_sha256=digest(WORK/'source_freeze.json')))
    print('NATIVE_FAMILY_LONG_WINDOW_SOLVER_COMPLETE',row['scenario_id'],branch,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','prepare','segment','finish'])
    p.add_argument('--index',type=int);p.add_argument('--branch',choices=['coarse','fine']);p.add_argument('--job')
    p.add_argument('--source',type=Path);p.add_argument('--tau',type=float);p.add_argument('--wall-seconds',type=int)
    a=p.parse_args()
    if a.action=='verify':verify();print('NATIVE_QUALITY_FREEZE_VERIFIED')
    elif a.action=='prepare':prepare(a.index,a.branch,a.job,a.source)
    elif a.action=='segment':segment(a.index,a.branch,a.job,a.tau)
    else:finish(a.index,a.branch,a.job,a.wall_seconds)

"""Private checkpoint continuation, gated segments, complete461-state manifest."""
import argparse
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
from native_quality_case_v1 import ROOT,read,digest
from native_review_serialization_v1 import save
from native_force_dictionary_v1 import force_every_step
from evaluate_box_fidelity import histories
from evaluate_fidelity_windows import load

WORK=ROOT/'results/native_expansion/cylinder_full_20261025_v3'
CASE=ROOT/'dataset_a/v2/cylinder_full_20261025_v3/a2_sc_001'
PREFIX=ROOT/'results/native_expansion/cylinder_temporal_20261023_v1'
SOURCE=ROOT/'dataset_a/v2/cylinder_temporal_20261023_v1/a2_sc_001_coarse_128179'
U=.0175
L=.0085


def schedule():
    return [round(k*.4,6) for k in range(201)]+[80+4*k for k in range(1,61)]+[round(320+.4*k,6) for k in range(1,201)]


def canonical_tau(value):
    return round(float(value),6)


def verify():
    f=read(WORK/'source_freeze.json')
    for p,sha in f['files'].items():
        if digest(Path(p))!=sha:raise ValueError('Full pilot source/provenance changed: '+p)
    return read(ROOT/'test/config/native_cylinder_full_20261025_v1.json')


def control(text,end,interval):
    entries={'startFrom':'latestTime','stopAt':'endTime','endTime':f'{end*L/U:.14g}',
        'deltaT':f'{.0001*L/U:.14g}','maxCo':'0.1','maxDeltaT':f'{.01*L/U:.14g}',
        'writeControl':'adjustableRunTime','writeInterval':f'{interval*L/U:.14g}','purgeWrite':'0'}
    for key,value in entries.items():
        text,n=re.subn(r'(?m)^'+key+r'\s+[^;]+;',key+' '+value+';',text)
        if n!=1:raise ValueError('Exactly one top-level setting required: '+key)
    return text


def locate_times(case,taus,nranks=32):
    available=[]
    for p in (case/'processor0').iterdir():
        try:t=float(p.name)
        except ValueError:continue
        if p.is_dir():available.append((t,p.name))
    result=[]
    for tau in taus:
        expected=tau*L/U;matches=[name for t,name in available if abs(t-expected)<1e-6]
        if len(matches)!=1:raise ValueError('Missing/ambiguous saved state: '+str(tau))
        name=matches[0]
        for rank in range(nranks):
            folder=case/f'processor{rank}'/name
            for field in ['U','p']:
                if not (folder/field).is_file() and not (folder/(field+'.gz')).is_file():raise ValueError('Incomplete parallel state')
        result.append(dict(tau=tau,time_s=expected,directory=name))
    return result


def init():
    cfg=verify()
    if CASE.exists() or (WORK/'initialization.json').exists():raise RuntimeError('Never overwrite an existing pilot')
    if cfg['scenario_id']!='a2_sc_001':raise ValueError('TRAIN pilot only')
    end=read(PREFIX/'family_runs/a2_sc_001_coarse_128179/retained_times.json')[-1]
    if end['tau']!=40:raise ValueError('Checkpoint40 required')
    roots=['0','constant','system']+[f'processor{i}/{name}' for i in range(32) for name in ['constant',end['directory']]]
    proof={};CASE.mkdir(parents=True,exist_ok=False)
    for relative in roots:
        src=SOURCE/relative
        if any(p.is_symlink() for p in src.rglob('*')):raise ValueError('Unexpected source symlink')
        before={str(p.relative_to(SOURCE)):digest(p) for p in src.rglob('*') if p.is_file()}
        shutil.copytree(src,CASE/relative)
        for name,sha in before.items():
            if digest(SOURCE/name)!=sha or digest(CASE/name)!=sha:raise ValueError('Checkpoint copy differs or source changed')
        proof.update(before)
    locate_times(CASE,[40])
    save(WORK/'checkpoint_copy_sha256.json',proof)
    save(WORK/'initialization.json',dict(status='PRIVATE_CHECKPOINT40_CLONED',source_case=str(SOURCE),case=str(CASE),
        checkpoint=end,source_modified=False,prefix_states_reused=101,copy_proof_sha256=digest(WORK/'checkpoint_copy_sha256.json')))


def prepare(index,job):
    cfg=verify();start,end,interval=cfg['stages'][index]
    if index==0:init()
    else:
        previous=read(WORK/'segments'/f'{index-1:02d}'/'summary.json')
        if previous['status']!='FULL_PILOT_SEGMENT_NUMERICALLY_PASSED' or previous['tau_end']!=start:raise ValueError('Previous segment did not pass')
    locate_times(CASE,[start])
    latest=max(float(p.name) for p in (CASE/'processor0').iterdir() if p.is_dir() and re.fullmatch(r'[0-9.eE+-]+',p.name))
    if abs(latest-start*L/U)>1e-6:raise ValueError('Unexpected latest checkpoint')
    out=WORK/'segments'/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    text=control((CASE/'system/controlDict').read_text(),end,interval)
    (CASE/'system/controlDict').write_text(text);(out/'controlDict.used').write_text(text)
    save(out/'metadata.json',dict(job=job,index=index,tau_start=start,tau_end=end,delta_tau=interval,
        case=str(CASE),control_sha256=digest(out/'controlDict.used'),source_freeze_sha256=digest(WORK/'source_freeze.json')))
    print(str(CASE),flush=True);print(str(out),flush=True)


def flux_audit():
    patches={}
    for name in ['inletFlux','outletFlux','frontFlux','backFlux']:
        h,n=histories(CASE,name)
        if not np.isfinite(h).all():raise ValueError('Nonfinite flux')
        patches[name]={r[0]:r[1] for r in h}
    times=sorted(set.intersection(*(set(x) for x in patches.values())))
    if not times:raise ValueError('No common flux times')
    area=.1*.04
    io=max(abs(patches['inletFlux'][t]+patches['outletFlux'][t])/(U*area) for t in times)
    periodic=max(abs(patches['frontFlux'][t]+patches['backFlux'][t])/(U*area) for t in times)
    if io>=1e-4 or periodic>=1e-4:raise ValueError('Mass flux gate failed')
    return dict(max_inlet_outlet_relative_imbalance=io,max_periodic_relative_imbalance=periodic,samples=len(times))


def finish(index,seconds):
    cfg=verify();start,end,interval=cfg['stages'][index];out=WORK/'segments'/f'{index:02d}'
    log=(out/'log.pimpleFoam').read_text()
    if len(re.findall(r'(?m)^End$',log))!=1:raise ValueError('Solver did not finish normally')
    co=[float(x) for x in re.findall(r'Courant Number mean:\s*\S+\s+max:\s*(\S+)',log)]
    cont=[float(x) for x in re.findall(r'cumulative\s*=\s*([^\s,]+)',log)]
    if not co or not cont or not all(math.isfinite(x) for x in co+cont) or max(co)>.11 or abs(cont[-1])>=1e-5:raise ValueError('Frozen numerical gate failed')
    states=locate_times(CASE,[t for t in schedule() if start<t<=end])
    save(out/'retained_times.json',states)
    save(out/'summary.json',dict(status='FULL_PILOT_SEGMENT_NUMERICALLY_PASSED',tau_start=start,tau_end=end,
        new_states=len(states),max_co=max(co),continuity=cont[-1],flux=flux_audit(),solver_wall_seconds=seconds,
        mpi_ranks=32,production_accepted=False,metadata_sha256=digest(out/'metadata.json'),log_sha256=digest(out/'log.pimpleFoam')))
    print('FULL_PILOT_SEGMENT_NUMERICALLY_PASSED',index,start,end,flush=True)


def endpoint400(out,entry):
    folder=out/'offline_force400';folder.mkdir();copy=folder/'case'
    for name in ['0','constant','system',entry['directory']]:shutil.copytree(CASE/name,copy/name)
    hashes={str(CASE/entry['directory']/k):digest(CASE/entry['directory']/k) for k in ['U','p','phi']}
    p=copy/'system/controlDict';p.write_text(force_every_step(p.read_text()))
    with (folder/'postprocess.log').open('w') as log:
        subprocess.run(['pimpleFoam','-case',str(copy),'-postProcess','-time',entry['directory']],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    h,n=histories(copy,'obstacleForces',True)
    if len(h)!=1 or abs(h[0,0]-entry['time_s'])>1e-6 or not np.isfinite(h).all():raise ValueError('Invalid final force')
    if any(digest(Path(p))!=sha for p,sha in hashes.items()):raise ValueError('Final source fields changed')
    save(folder/'endpoint.json',dict(tau=400,time_s=float(h[0,0]),Cd=float(h[0,1]),Cl=float(h[0,2]),offline_only=True,source_hashes=hashes))
    return np.array([400.,h[0,1],h[0,2]])


def export(job):
    cfg=verify();out=WORK/'export'/job;out.mkdir(parents=True,exist_ok=False)
    compact=out/'compact';compact.mkdir();manifest=[];reference=None
    freeze=read(WORK/'source_freeze.json');prefix_times=read(PREFIX/'family_runs/a2_sc_001_coarse_128179/retained_times.json')
    entries=[]
    for entry in prefix_times:
        p=PREFIX/'reviews/a2_sc_001_128181/coarse/compact'/f"a2_sc_001_t{float(entry['directory']):g}.npz"
        if digest(p)!=freeze['prefix_exports'][str(p)]:raise ValueError('Prefix export SHA changed')
        entries.append((entry,p,SOURCE))
    new_times=[]
    for index in range(len(cfg['stages'])):
        segment=WORK/'segments'/f'{index:02d}';s=read(segment/'summary.json')
        if s['status']!='FULL_PILOT_SEGMENT_NUMERICALLY_PASSED':raise ValueError('Incomplete full solver')
        new_times.extend(read(segment/'retained_times.json'))
    for entry in new_times:
        name=entry['directory']
        with (out/'export.log').open('a') as log:
            subprocess.run(['reconstructPar','-case',str(CASE),'-time',name],stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,str(ROOT/'test/analysis/export_uniform_snapshot.py'),'--case-dir',str(CASE),
                '--time',name,'--diameter',str(L),'--scenario-id','a2_sc_001','--output-dir',str(compact),
                '--dimensions','128','64','32','--bounds','-.05','.15','-.05','.05','-.02','.02'],stdout=log,stderr=subprocess.STDOUT,check=True)
        entries.append((entry,compact/f'a2_sc_001_t{float(name):g}.npz',CASE))
    for entry,p,source in entries:
        arr=load(p);meta=read(p.with_suffix('.json'))
        if meta['source_case']!=str(source) or abs(meta['time_s']-entry['time_s'])>1e-6 or meta['npz_bytes']!=p.stat().st_size:raise ValueError('Export provenance differs')
        if not np.isfinite(arr['U']).all() or not np.isfinite(arr['p']).all() or abs(float(arr['time_s'])-entry['time_s'])>1e-6:raise ValueError('Nonfinite/wrong-time fields')
        if reference is None:reference={k:arr[k].copy() for k in ['x','y','z','valid']}
        if any(not np.array_equal(arr[k],reference[k]) for k in reference):raise ValueError('Grid or static mask changed')
        manifest.append(dict(**dict(entry,tau=canonical_tau(entry['tau'])),npz=str(p),npz_sha256=digest(p),metadata_sha256=digest(p.with_suffix('.json')),source_case=str(source)))
    if [e['tau'] for e in manifest]!=schedule():raise ValueError('Full461-state schedule differs')
    point=endpoint400(out,new_times[-1]);old,n=histories(SOURCE,'obstacleForces',True);old[:,0]*=U/L
    prefix_end=read(PREFIX/'reviews/a2_sc_001_128181/coarse/offline_endpoint/endpoint.json')
    from native_force_endpoint_support_v2 import append_endpoint
    old,audit=append_endpoint(old,prefix_end,U,L)
    new,n=histories(CASE,'obstacleForces',True);new[:,0]*=U/L
    if not np.isfinite(new).all() or new[-1,0]>400+1e-6:raise ValueError('Invalid continuation forces')
    if abs(new[-1,0]-400)<1e-8:
        if not np.allclose(new[-1,1:],point[1:],rtol=1e-6,atol=1e-8):raise ValueError('Final forces disagree')
        new[-1,0]=400
    else:new=np.vstack([new,point])
    full=np.vstack([old[old[:,0]<new[0,0]],new])
    if not (np.diff(full[:,0])>0).all():raise ValueError('Nonmonotonic merged force history')
    save(out/'full_force_history.json',dict(columns=['tau','Cd','Cl'],values=full.tolist(),prefix_source=str(SOURCE),continuation_source=str(CASE),offline_final_endpoint=True))
    save(out/'manifest.json',manifest)
    save(out/'summary.json',dict(status='FULL_LENGTH_PILOT_NUMERICALLY_COMPLETE',scenario_id='a2_sc_001',split='train',
        states=len(manifest),tau_range=[0,400],prefix_reused=101,new_exports=len(new_times),
        manifest_sha256=digest(out/'manifest.json'),force_history_sha256=digest(out/'full_force_history.json'),
        source_freeze_sha256=digest(WORK/'source_freeze.json'),late_fine_comparator_available=False,
        production_accepted=False,LLM_training_started=False,limitations=cfg['limitations']))
    print('FULL_LENGTH_PILOT_NUMERICALLY_COMPLETE',len(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','finish','export']);p.add_argument('--index',type=int);p.add_argument('--job');p.add_argument('--seconds',type=int)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.index,a.job)
    elif a.action=='finish':finish(a.index,a.seconds)
    else:export(a.job)

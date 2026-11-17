"""Review completed box configurations; retain restart provenance and clip old history tails."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from evaluate_fidelity_windows import load

U=.02125
L=.0115


def histories(case,function,force=False):
    files=sorted((case/'postProcessing'/function).glob('*/*.dat'),key=lambda p:(float(p.parent.name),p.name))
    starts=sorted({float(p.parent.name) for p in files});rows={};discarded=0
    for p in files:
        start=float(p.parent.name);next_start=next((v for v in starts if v>start),float('inf'))
        text=p.read_text().splitlines()
        if force:
            h=next(v.lstrip('#').split() for v in text if v.lstrip('#').split()[:1]==['Time'])
            columns=[0,h.index('Cd'),h.index('Cl')]
        else:columns=[0,1]
        for line in text:
            if not line.strip() or line.lstrip().startswith('#'):continue
            values=line.split();t=float(values[0])
            if t>=next_start-1e-10:
                discarded+=1;continue
            row=[float(values[i]) for i in columns]
            if not np.isfinite(row).all():raise RuntimeError('Nonfinite history')
            rows[t]=row
    if not rows:raise RuntimeError(f'No history {function}')
    return np.array([rows[t] for t in sorted(rows)]),discarded


def exports(project,case,out):
    (out/'compact').mkdir(parents=True,exist_ok=True)
    times=[p for p in (case/'processor0').iterdir() if p.is_dir() and p.name.replace('.','',1).isdigit()]
    paths={}
    for tau in np.arange(101)*.4:
        t=tau*L/U
        matches=[p for p in times if abs(float(p.name)-t)<1e-6]
        if len(matches)!=1:raise RuntimeError(f'Missing or ambiguous tau {tau}: {matches}')
        selected=matches[0]
        if any(not (case/f'processor{r}'/selected.name).is_dir() for r in range(32)):
            raise RuntimeError('Incomplete rank schedule')
        path=out/'compact'/f'a2_bx_001_t{float(selected.name):g}.npz'
        if not path.exists():
            with (out/'reconstruct_export.log').open('a') as log:
                if tau>0:subprocess.run(['reconstructPar','-case',str(case),'-time',selected.name],stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,str(project/'test/analysis/export_uniform_snapshot.py'),'--case-dir',str(case),'--time',selected.name,'--diameter',str(L),'--scenario-id','a2_bx_001','--output-dir',str(out/'compact'),'--dimensions','128','64','32','--bounds','-.05','.15','-.05','.05','-.02','.02'],stdout=log,stderr=subprocess.STDOUT,check=True)
        d=load(path)
        if abs(float(d['time_s'])-t)>1e-6:raise RuntimeError('Export physical time mismatch')
        paths[round(float(tau),6)]=path
    return paths


def raw_pressure(case):
    import pyvista as pv
    from export_uniform_snapshot import find_internal_mesh
    reader=pv.OpenFOAMReader(str(case/'case.foam'))
    selected=min(reader.time_values,key=lambda t:abs(t-40*L/U))
    assert abs(selected-40*L/U)<1e-6
    reader.set_active_time_value(selected);mesh=find_internal_mesh(reader.read())
    p=np.asarray(mesh.cell_data['p'],dtype=float)
    v=np.asarray(mesh.compute_cell_sizes(length=False,area=False).cell_data['Volume'])
    assert np.isfinite(p).all() and np.isfinite(v).all() and (v>0).all()
    centered=p-np.average(p,weights=v)
    return {'cells':len(p),'centered_min_over_u2':float(centered.min()/U**2),'centered_max_over_u2':float(centered.max()/U**2),'volume_weighted_centered_rms_over_u2':float(np.sqrt(np.average(centered**2,weights=v))/U**2),'warning':'Raw-cell descriptive screen, not a convergence proof or raster norm.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--resume-job',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((a.project/'test/config/box_fidelity_protocol_20261020.json').read_text())
    proof=a.project/f'results/dataset_a/v2_box_resume/{a.resume_job}'
    meta=dict(l.split('=',1) for l in (proof/'solver_summary.txt').read_text().splitlines() if '=' in l)
    assert meta['status']=='BOX_FINE_RESUMED_SOLVER_COMPLETE' and float(meta['tau_end'])==40
    cases={'coarse':a.project/'dataset_a/v2/box_fidelity/a2_bx_001_coarse_122475','fine':Path(meta['case_dir'])}
    series={k:exports(a.project,v,a.output/k) for k,v in cases.items()}
    force={};flux={};discarded={}
    for k,case in cases.items():
        f,n=histories(case,'obstacleForces',True);f[:,0]*=U/L;force[k]=f;discarded[k]={'forces':n}
        patches={}
        for label in ('inletFlux','outletFlux','frontFlux','backFlux'):
            history,excluded=histories(case,label);patches[label]={float(row[0]):float(row[1]) for row in history};discarded[k][label]=excluded
        common=sorted(set.intersection(*(set(v) for v in patches.values())))
        if not common:raise RuntimeError('No shared flux times')
        io=max(abs(patches['inletFlux'][t]+patches['outletFlux'][t])/(U*.004) for t in common)
        per=max(abs(patches['frontFlux'][t]+patches['backFlux'][t])/(U*.004) for t in common)
        flux[k]={'max_inlet_outlet_relative_imbalance':io,'max_periodic_relative_imbalance':per,'passed':io<1e-4 and per<1e-4,'sample_count':len(common)}
    with (a.project/'test/config/dataset_a_v2_obstacles.csv').open() as f:obs=[r for r in csv.DictReader(f) if r['scenario_id']=='a2_bx_001']
    centers=np.array([[float(r[k]) for k in ('center_x_m','center_y_m')] for r in obs]);radii=np.array([.5*np.sqrt(sum(float(r[k])**2 for k in ('size_x_m','size_y_m','size_z_m'))) for r in obs])
    bounds=[max(-.05,centers[:,0].min()),min(.15,(centers[:,0]+radii).max()+6*L),max(-.05,(centers[:,1]-radii).min()-.02),min(.05,(centers[:,1]+radii).max()+.02)]
    accum={tuple(w):np.zeros(6) for w in protocol['comparison_windows_tau']};snapshots=[]
    for tau in sorted(series['fine']):
        if tau<4:continue
        r=load(series['fine'][tau]);c=load(series['coarse'][tau])
        assert all(np.array_equal(r[k],c[k]) for k in ('x','y','z'))
        mask=r['valid'].astype(bool)&c['valid'].astype(bool)
        x,y,z=np.meshgrid(r['x'],r['y'],r['z'],indexing='ij');wake=mask&(x>=bounds[0])&(x<=bounds[1])&(y>=bounds[2])&(y<=bounds[3])
        assert mask.any() and wake.any()
        ur,uc=r['U'].astype(float),c['U'].astype(float);pr,pc=r['p'].astype(float),c['p'].astype(float)
        pr-=pr[mask].mean();pc-=pc[mask].mean()
        values=np.array([np.sum((uc[mask]-ur[mask])**2),np.sum(ur[mask]**2),np.sum((pc[mask]-pr[mask])**2),np.sum(pr[mask]**2),np.sum((uc[wake]-ur[wake])**2),np.sum(ur[wake]**2)])
        snapshots.append({'tau':tau,'velocity_relative_l2':float(np.sqrt(values[0]/max(values[1],1e-30))),'pressure_relative_l2':float(np.sqrt(values[2]/max(values[3],1e-30))),'mask_disagreement_fraction':float(np.mean(r['valid']!=c['valid']))})
        for w in accum:
            if w[0]-1e-6<=tau<=w[1]+1e-6:accum[w]+=values
    rows=[]
    for w,v in accum.items():
        grid=np.arange(w[0],w[1]+.2,.4);means={}
        for k,history in force.items():
            assert history[0,0]<=grid[0] and history[-1,0]>=grid[-1]-.1
            means[k]=[float(np.interp(grid,history[:,0],history[:,j]).mean()) for j in (1,2)]
        m={'velocity_relative_l2_global':float(np.sqrt(v[0]/max(v[1],1e-30))),'pressure_mean_aligned_relative_l2_global':float(np.sqrt(v[2]/max(v[3],1e-30))),'velocity_relative_l2_wake':float(np.sqrt(v[4]/max(v[5],1e-30))),'mean_drag_coefficient_relative_difference':abs(means['coarse'][0]-means['fine'][0])/max(abs(means['fine'][0]),1e-30),'mean_lift_coefficient_absolute_difference':abs(means['coarse'][1]-means['fine'][1])}
        rows.append({'tau_window':w,'metrics':m,'force_means_cd_cl':means,'passed':all(m[k.removesuffix('_max')]<=limit for k,limit in protocol['screen_limits'].items())})
    raw={k:raw_pressure(v) for k,v in cases.items()}
    report={'status':'BOX_FIDELITY_REVIEW_COMPLETE','resume_job':a.resume_job,'protocol':protocol,'case_paths':{k:str(v) for k,v in cases.items()},'results':rows,'flux_audits':flux,'raw_pressure_tau40':raw,'discarded_overlap_history_rows':discarded,'all_exploratory_screens_passed':all(r['passed'] for r in rows) and all(v['passed'] for v in flux.values()),'production_authorized_automatically':False,'restart_provenance':json.loads((proof/'restart_provenance.json').read_text()),'warning':'Fine run restarted at tau23.2 after timeout. Old post-checkpoint history tails clipped at later restart directory. Configuration discrepancy, not pure h-only/time-step convergence.'}
    report['wake_bounds_xy_m']=list(map(float,bounds))
    report['exported_frames_per_branch']={k:len(v) for k,v in series.items()}
    (a.output/'box_fidelity_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (a.output/'snapshot_metrics.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'results':rows,'all_exploratory_screens_passed':report['all_exploratory_screens_passed']},indent=2),flush=True)


if __name__=='__main__':main()

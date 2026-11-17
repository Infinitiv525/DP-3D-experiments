"""Paired TRAIN family quality screens. Generic U/L, no box-specific constants."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
from native_quality_case_v1 import ROOT,WORK,verify,paths,scenario,save,digest,read,item_for
from evaluate_box_fidelity import histories
from evaluate_fidelity_windows import load


def export_series(case,run_summary,row,out):
    compact=out/'compact';compact.mkdir(parents=True,exist_ok=False)
    expected=read(run_summary.parent/'retained_times.json');result={}
    for entry in expected:
        name=entry['directory'];time=entry['time_s'];tau=entry['tau']
        if tau>0:
            with (out/'export.log').open('a') as log:
                subprocess.run(['reconstructPar','-case',str(case),'-time',name],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/'export.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'test/analysis/export_uniform_snapshot.py'),
                '--case-dir',str(case),'--time',name,'--diameter',row['reference_length_m'],
                '--scenario-id',row['scenario_id'],'--output-dir',str(compact),'--dimensions','128','64','32',
                '--bounds','-.05','.15','-.05','.05','-.02','.02'],stdout=log,stderr=subprocess.STDOUT,check=True)
        p=compact/f"{row['scenario_id']}_t{float(name):g}.npz"
        d=load(p)
        if abs(float(d['time_s'])-time)>1e-6:raise ValueError('Exported time differs')
        result[round(tau,6)]=p
    if len(result)!=101:raise ValueError('Require101states per comparison branch')
    return result


def run(index,coarse_job,fine_job,review_job):
    cfg=verify();item=item_for(index);row=scenario(item['scenario_id'])
    u=float(row['u_inlet_m_s']);length=float(row['reference_length_m'])
    out=WORK/'family_reviews'/f"{item['scenario_id']}_{review_job}"
    out.mkdir(parents=True,exist_ok=False)
    cases={};series={};forces={};flux={};discarded={};source_hashes={}
    domain=read(ROOT/'test/config/dataset_a_v2_protocol.json')['domain_m']
    inlet_area=(domain['y'][1]-domain['y'][0])*(domain['z'][1]-domain['z'][0])
    for branch,job in [('coarse',coarse_job),('fine',fine_job)]:
        case,source=paths(index,branch,job);p=source/'summary.json';s=read(p)
        if s['status']!='NATIVE_FAMILY_LONG_WINDOW_SOLVER_COMPLETE' or s['time_states']!=101:
            raise ValueError('Incomplete solver output')
        cases[branch]=case;source_hashes[str(p)]=digest(p)
        dest=out/branch;dest.mkdir();series[branch]=export_series(case,p,row,dest)
        history,n=histories(case,'obstacleForces',True);history[:,0]*=u/length
        forces[branch]=history;discarded[branch]={'forces':n}
        patches={}
        for function in ['inletFlux','outletFlux','frontFlux','backFlux']:
            h,n=histories(case,function);patches[function]={r[0]:r[1] for r in h};discarded[branch][function]=n
        common=sorted(set.intersection(*(set(v) for v in patches.values())))
        if not common:raise ValueError('No common flux samples')
        io=max(abs(patches['inletFlux'][t]+patches['outletFlux'][t])/(u*inlet_area) for t in common)
        periodic=max(abs(patches['frontFlux'][t]+patches['backFlux'][t])/(u*inlet_area) for t in common)
        flux[branch]=dict(max_inlet_outlet_relative_imbalance=io,max_periodic_relative_imbalance=periodic,
            samples=len(common),passed=io<1e-4 and periodic<1e-4)
    with (ROOT/'test/config/dataset_a_v2_obstacles.csv').open(newline='') as f:
        obstacles=[r for r in csv.DictReader(f) if r['scenario_id']==row['scenario_id']]
    centers=np.array([[float(r[k]) for k in ['center_x_m','center_y_m']] for r in obstacles])
    radii=np.array([.5*np.sqrt(sum(float(r[k])**2 for k in ['size_x_m','size_y_m','size_z_m'])) for r in obstacles])
    bounds=[max(-.05,centers[:,0].min()),min(.15,(centers[:,0]+radii).max()+6*length),
        max(-.05,(centers[:,1]-radii).min()-2*length),min(.05,(centers[:,1]+radii).max()+2*length)]
    accum={tuple(w):np.zeros(6) for w in cfg['fidelity_windows_tau']};snapshots=[]
    for tau in sorted(series['fine']):
        if tau<4:continue
        fine=load(series['fine'][tau]);coarse=load(series['coarse'][tau])
        if not all(np.array_equal(fine[k],coarse[k]) for k in ['x','y','z']):raise ValueError('Different export grids')
        mask=fine['valid'].astype(bool)&coarse['valid'].astype(bool)
        xx,yy,zz=np.meshgrid(fine['x'],fine['y'],fine['z'],indexing='ij')
        wake=mask&(xx>=bounds[0])&(xx<=bounds[1])&(yy>=bounds[2])&(yy<=bounds[3])
        if not mask.any() or not wake.any():raise ValueError('Empty comparison masks')
        uf,uc=fine['U'].astype(float),coarse['U'].astype(float)
        pf,pc=fine['p'].astype(float),coarse['p'].astype(float)
        pf-=pf[mask].mean();pc-=pc[mask].mean()
        v=np.array([np.sum((uc[mask]-uf[mask])**2),np.sum(uf[mask]**2),np.sum((pc[mask]-pf[mask])**2),
            np.sum(pf[mask]**2),np.sum((uc[wake]-uf[wake])**2),np.sum(uf[wake]**2)])
        if not np.isfinite(v).all() or any(v[i]<=1e-30 for i in [1,3,5]):raise ValueError('Degenerate field norm')
        snapshots.append(dict(tau=tau,velocity_relative_l2=float(np.sqrt(v[0]/v[1])),
            pressure_relative_l2=float(np.sqrt(v[2]/v[3])),mask_disagreement_fraction=float(np.mean(fine['valid']!=coarse['valid']))))
        for w in accum:
            if w[0]-1e-6<=tau<=w[1]+1e-6:accum[w]+=v
    results=[]
    for window,v in accum.items():
        grid=np.arange(window[0],window[1]+.2,.4);means={}
        for branch,h in forces.items():
            if h[0,0]>grid[0] or h[-1,0]<grid[-1]-1e-6:raise ValueError('Force history does not cover window')
            means[branch]=[float(np.interp(grid,h[:,0],h[:,j]).mean()) for j in [1,2]]
        if abs(means['fine'][0])<=1e-30:raise ValueError('Degenerate reference drag')
        metrics=dict(velocity_relative_l2_global=float(np.sqrt(v[0]/v[1])),
            pressure_mean_aligned_relative_l2_global=float(np.sqrt(v[2]/v[3])),velocity_relative_l2_wake=float(np.sqrt(v[4]/v[5])),
            mean_drag_coefficient_relative_difference=abs(means['coarse'][0]-means['fine'][0])/abs(means['fine'][0]),
            mean_lift_coefficient_absolute_difference=abs(means['coarse'][1]-means['fine'][1]))
        results.append(dict(window_tau=window,metrics=metrics,force_means_cd_cl=means,
            passed=all(metrics[k]<=limit for k,limit in cfg['exploratory_limits'].items())))
    report=dict(status='NATIVE_FAMILY_FIDELITY_REVIEW_COMPLETE',scenario=row,comparison_windows=results,
        all_exploratory_screens_passed=all(r['passed'] for r in results) and all(r['passed'] for r in flux.values()),
        flux_audits=flux,case_paths={k:str(v) for k,v in cases.items()},source_hashes=source_hashes,
        wake_bounds_xy_m=list(map(float,bounds)),discarded_overlap_history_rows=discarded,
        pressure_gauge_adjusted=True,exported_states_per_branch=101,production_accepted=False,
        model_test_metrics_computed=False,source_freeze_sha256=digest(WORK/'source_freeze.json'),
        limitations=cfg['limitations'])
    save(out/'snapshot_metrics.json',snapshots);save(out/'review.json',report)
    print('NATIVE_FAMILY_FIDELITY_REVIEW_COMPLETE',row['scenario_id'],report['all_exploratory_screens_passed'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--coarse-job',required=True)
    p.add_argument('--fine-job',required=True);p.add_argument('--review-job',required=True)
    a=p.parse_args();run(a.index,a.coarse_job,a.fine_job,a.review_job)

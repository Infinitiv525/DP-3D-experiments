"""Frozen TRAIN late windows, no phase shift or post-result threshold changes."""
import argparse
from pathlib import Path
import numpy as np
from native_cylinder_full_fine_v1 import WORK,ROOT,verify,read,digest
from native_review_serialization_v1 import save
from evaluate_fidelity_windows import load


def field_sums(candidate,reference,mask,wake):
    if not mask.any() or not wake.any():raise ValueError('Empty comparison mask')
    uc=candidate['U'].astype(float);uf=reference['U'].astype(float)
    pc=candidate['p'].astype(float);pf=reference['p'].astype(float)
    if not all(np.isfinite(v).all() for v in [uc,uf,pc,pf]):raise ValueError('Nonfinite field')
    pc-=pc[mask].mean();pf-=pf[mask].mean()
    v=np.array([np.sum((uc[mask]-uf[mask])**2),np.sum(uf[mask]**2),np.sum((pc[mask]-pf[mask])**2),np.sum(pf[mask]**2),np.sum((uc[wake]-uf[wake])**2),np.sum(uf[wake]**2)])
    if not np.isfinite(v).all() or min(v[[1,3,5]])<=1e-30:raise ValueError('Degenerate reference norm')
    return v


def metrics(v,candidate_means,reference_means):
    if abs(reference_means[0])<=1e-30:raise ValueError('Degenerate reference drag')
    return dict(velocity_relative_l2_global=float(np.sqrt(v[0]/v[1])),pressure_mean_aligned_relative_l2_global=float(np.sqrt(v[2]/v[3])),
        velocity_relative_l2_wake=float(np.sqrt(v[4]/v[5])),mean_drag_coefficient_relative_difference=abs(candidate_means[0]-reference_means[0])/abs(reference_means[0]),
        mean_lift_coefficient_absolute_difference=abs(candidate_means[1]-reference_means[1]))


def series(folder):
    summary=read(folder/'summary.json')
    if summary['status']!='FULL_LENGTH_PILOT_NUMERICALLY_COMPLETE' or summary['states']!=461:raise ValueError('Incomplete full-length data')
    if digest(folder/'manifest.json')!=summary['manifest_sha256'] or digest(folder/'full_force_history.json')!=summary['force_history_sha256']:raise ValueError('Full-series integrity mismatch')
    manifest=read(folder/'manifest.json');forces=np.asarray(read(folder/'full_force_history.json')['values'],dtype=float)
    if not np.isfinite(forces).all() or not (np.diff(forces[:,0])>0).all() or forces[-1,0]!=400:raise ValueError('Invalid full force history')
    return {round(e['tau'],6):e for e in manifest},forces,summary


def run(job,fine_export):
    cfg=verify();freeze=read(WORK/'comparison_freeze.json')
    for p,sha in freeze['files'].items():
        if digest(Path(p))!=sha:raise ValueError('Late comparison source changed')
    baseline=ROOT/cfg['baseline_work_root']/'export'/str(cfg['baseline_export_job'])
    fine_folder=WORK/'export'/fine_export
    coarse,coarse_force,cs=series(baseline);fine,fine_force,fs=series(fine_folder)
    if sorted(coarse)!=sorted(fine):raise ValueError('Different full schedules')
    windows=[tuple(w) for w in cfg['late_windows_tau']];accum={w:np.zeros(6) for w in windows};counts={w:0 for w in windows}
    bounds=freeze['wake_bounds_xy_m'];snapshots=[];hashes={}
    for tau in sorted(fine):
        if tau<320:continue
        arrays=[]
        for entry in [coarse[tau],fine[tau]]:
            p=Path(entry['npz'])
            if digest(p)!=entry['npz_sha256']:raise ValueError('Late NPZ SHA differs')
            arrays.append(load(p));hashes[str(p)]=entry['npz_sha256']
        c,f=arrays
        if not all(np.array_equal(c[k],f[k]) for k in ['x','y','z']):raise ValueError('Different grids')
        mask=c['valid'].astype(bool)&f['valid'].astype(bool)
        xx,yy,zz=np.meshgrid(f['x'],f['y'],f['z'],indexing='ij')
        wake=mask&(xx>=bounds[0])&(xx<=bounds[1])&(yy>=bounds[2])&(yy<=bounds[3])
        v=field_sums(c,f,mask,wake)
        snapshots.append(dict(tau=tau,velocity_relative_l2=float(np.sqrt(v[0]/v[1])),pressure_relative_l2=float(np.sqrt(v[2]/v[3])),mask_disagreement_fraction=float(np.mean(c['valid']!=f['valid']))))
        for w in windows:
            if w[0]-1e-6<=tau<=w[1]+1e-6:accum[w]+=v;counts[w]+=1
    results=[]
    for w,v in accum.items():
        grid=np.arange(w[0],w[1]+.2,.4)
        if counts[w]!=len(grid):raise ValueError('Missing late comparison samples')
        means={};stats={}
        for branch,h in [('candidate1.0',coarse_force),('reference0.8',fine_force)]:
            if h[0,0]>grid[0] or h[-1,0]<grid[-1]-1e-6:raise ValueError('Insufficient force coverage')
            samples=np.column_stack([np.interp(grid,h[:,0],h[:,i]) for i in [1,2]])
            means[branch]=samples.mean(axis=0).tolist();stats[branch]=dict(mean_cd_cl=means[branch],std_cd_cl=samples.std(axis=0).tolist())
        m=metrics(v,means['candidate1.0'],means['reference0.8'])
        results.append(dict(window_tau=w,states=counts[w],metrics=m,force_diagnostics=stats,passed=all(m[k]<=limit for k,limit in cfg['exploratory_limits'].items())))
    out=WORK/'late_review'/job;out.mkdir(parents=True,exist_ok=False)
    save(out/'snapshot_metrics.json',snapshots);save(out/'late_npz_sha256.json',hashes)
    save(out/'review.json',dict(status='LATE_FULL_SERIES_COMPARISON_COMPLETE',scenario_id='a2_sc_001',split='train',candidate_mesh_factor=1.0,reference_mesh_factor=.8,
        max_co_both=.1,max_delta_tau_both=.01,comparison_windows=results,all_late_screens_passed=all(r['passed'] for r in results),phase_alignment_used=False,pressure_gauge_adjusted=True,
        overlapping_windows_not_independent_replicates=True,comparison_freeze_sha256=digest(WORK/'comparison_freeze.json'),production_accepted=False,LLM_training_started=False,
        limitations=['One TRAIN geometry; two resolutions do not prove convergence','Relative field differences are not absolute physical errors','No tolerance changes or phase alignment for acceptance','Production decision requires separate review']))
    print('LATE_FULL_SERIES_COMPARISON_COMPLETE',all(r['passed'] for r in results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--fine-export',required=True);a=p.parse_args();run(a.job,a.fine_export)

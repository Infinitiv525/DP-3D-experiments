"""Audit independent mesh/time-step runs against the existing full pilot."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def terminal_metadata(path: Path) -> dict:
    return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)


def force_series(case: Path) -> np.ndarray:
    rows=[]
    for path in (case/'postProcessing/obstacleForces').glob('*/*.dat'):
        data=np.loadtxt(path,comments='#',ndmin=2)
        header=next(line.lstrip('#').split() for line in path.read_text().splitlines()
                    if line.lstrip('#').split()[:1]==['Time'])
        rows.extend(data[:,[0,header.index('Cd'),header.index('Cl')]].tolist())
    if not rows: raise RuntimeError(f'Missing forces: {case}')
    by_time={float(row[0]):row for row in rows}
    result=np.array([by_time[t] for t in sorted(by_time)])
    if not np.isfinite(result).all(): raise RuntimeError('Non-finite forces')
    result[:,0]*=0.02375/0.01
    return result


def export_series(project: Path, case: Path, result: Path, minimum_tau: float=0.0) -> dict[float,Path]:
    compact=result/'compact'
    compact.mkdir(exist_ok=True)
    times=[]
    for p in (case/'processor0').iterdir():
        try: t=float(p.name)
        except ValueError: continue
        if p.is_dir() and t*.02375/.01>=minimum_tau-1e-6: times.append((t,p.name))
    times.sort()
    expected=np.arange(round(minimum_tau/.4),101)*0.4
    actual=np.array([t for t,_ in times])*0.02375/0.01
    if len(times)!=len(expected) or not np.allclose(actual,expected,rtol=0,atol=1e-5):
        raise RuntimeError('Incomplete fidelity time schedule')
    for rank in range(32):
        if any(not (case/f'processor{rank}'/name).is_dir() for _,name in times):
            raise RuntimeError(f'Incomplete processor {rank}')
    paths={}
    for tau,(t,name) in zip(expected,times):
        cached=compact/f'a2_mp_001_t{t:g}.npz'
        if cached.exists() and cached.with_suffix('.json').exists():
            data=load(cached)
            if abs(float(data['time_s'])-t)>1e-6: raise RuntimeError('Cached physical time mismatch')
            paths[round(float(tau),6)]=cached
            continue
        if t!=0:
            with (result/'logs/log.reconstruct_export').open('a') as log:
                subprocess.run(['reconstructPar','-case',str(case),'-time',name],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        with (result/'export_stdout.jsonl').open('a') as log:
            subprocess.run([sys.executable,str(project/'test/analysis/export_uniform_snapshot.py'),
                '--case-dir',str(case),'--time',name,'--diameter','0.01','--scenario-id','a2_mp_001',
                '--output-dir',str(compact),'--dimensions','128','64','32',
                '--bounds','-0.05','0.15','-0.05','0.05','-0.02','0.02'],stdout=log,check=True)
        paths[round(float(tau),6)]=compact/f'a2_mp_001_t{t:g}.npz'
    return paths


def load(path: Path) -> dict:
    with np.load(path,allow_pickle=False) as archive:
        data={key:archive[key] for key in archive.files}
    if data['U'].shape!=(128,64,32,3) or data['p'].shape!=(128,64,32):
        raise RuntimeError('Incorrect field dimensions')
    if not np.isfinite(data['U']).all() or not np.isfinite(data['p']).all():
        raise RuntimeError('Non-finite fields')
    return data


def audit_flux(case: Path) -> dict:
    fluxes={}
    for patch in ('inletFlux','outletFlux','frontFlux','backFlux'):
        history={}
        for path in (case/'postProcessing'/patch).glob('*/*.dat'):
            rows=np.loadtxt(path,comments='#',ndmin=2)
            if not np.isfinite(rows).all(): raise RuntimeError('Non-finite flux history')
            history.update({float(row[0]):float(row[1]) for row in rows})
        if not history: raise RuntimeError(f'Missing flux history: {case} {patch}')
        fluxes[patch]=history
    shared=sorted(set.intersection(*(set(v) for v in fluxes.values())))
    if not shared: raise RuntimeError('Flux histories have no shared times')
    io=max(abs(fluxes['inletFlux'][t]+fluxes['outletFlux'][t])/9.5e-5 for t in shared)
    periodic=max(abs(fluxes['frontFlux'][t]+fluxes['backFlux'][t])/9.5e-5 for t in shared)
    return {'max_sampled_inlet_outlet_relative_imbalance':io,
            'max_sampled_periodic_relative_imbalance':periodic,
            'sample_count':len(shared),'passed':io<1e-4 and periodic<1e-4}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--fine-case',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    project=args.project
    if (args.fine_case is None)!=(args.output is None):
        raise ValueError('Provide both fine-case and output overrides')
    result=args.output or project/f'results/dataset_a/v2_fidelity_review/{args.job}'
    result.mkdir(parents=True,exist_ok=True)
    gate=json.loads((project/f'results/dataset_a/v2_fidelity/a2_mp_001_fidelity_{args.job}_fine_dt02/fidelity_gate.used.json').read_text())
    second_gate=json.loads((project/f'results/dataset_a/v2_fidelity/a2_mp_001_fidelity_{args.job}_coarse_dt01/fidelity_gate.used.json').read_text())
    if gate!=second_gate: raise RuntimeError('Branches used different frozen fidelity gates')
    (result/'fidelity_gate.used.json').write_text(json.dumps(gate,indent=2)+'\n')
    cases={label:project/f'dataset_a/v2/fidelity/a2_mp_001_fidelity_{args.job}_{label}'
           for label in ('fine_dt02','coarse_dt01')}
    if args.fine_case is not None: cases['fine_dt02']=args.fine_case
    sources={}
    for label,case in cases.items():
        branch=project/f'results/dataset_a/v2_fidelity/a2_mp_001_fidelity_{args.job}_{label}'
        if label=='fine_dt02' and args.fine_case is not None:
            branch=result/'smooth_fine_exports'
            (branch/'logs').mkdir(parents=True,exist_ok=True)
        sources[label]=export_series(project,case,branch,4.0 if args.fine_case is not None else 0.0)
    pilot=project/'dataset_a/v2/full_cost_pilot/a2_mp_001_fullcost_109922'
    pilot_result=project/'results/dataset_a/v2_full_cost_pilot/a2_mp_001_fullcost_109922'
    with (pilot_result/'series_index.csv').open() as stream:
        index=list(csv.DictReader(stream))
    sources['coarse_dt02']={round(float(r['tau']),6):pilot_result/'compact'/r['filename']
                            for r in index if float(r['tau'])<=40.000001}
    cases['coarse_dt02']=pilot
    forces={label:force_series(case) for label,case in cases.items()}
    flux_audits={label:audit_flux(case) for label,case in cases.items()}
    with (project/'test/config/dataset_a_v2_obstacles.csv').open() as stream:
        obstacles=[r for r in csv.DictReader(stream) if r['scenario_id']=='a2_mp_001']
    # Conservative bounding box includes the full diagonal of rotated obstacles.
    centers=np.array([[float(r[k]) for k in ('center_x_m','center_y_m')] for r in obstacles])
    radii=np.array([0.5*np.sqrt(sum(float(r[k])**2 for k in ('size_x_m','size_y_m','size_z_m')))
                    for r in obstacles])
    bounds=[max(-.05,float(centers[:,0].min())),min(.15,float((centers[:,0]+radii).max())+.06),
            max(-.05,float((centers[:,1]-radii).min())-.02),min(.05,float((centers[:,1]+radii).max())+.02)]
    results=[]
    snapshots=[]
    for kind,reference,candidate,thresholds in (
        ('mesh','fine_dt02','coarse_dt02',gate['exploratory_mesh_screen']),
        ('time_step','coarse_dt01','coarse_dt02',gate['exploratory_time_step_screen'])):
        accum={tuple(window):np.zeros(6) for window in gate['comparison_windows_tau']}
        for tau in sorted(sources[reference]):
            if tau<4: continue
            ref=load(sources[reference][tau]); cand=load(sources[candidate][tau])
            if abs(float(ref['time_s'])-float(cand['time_s']))>1e-6:
                raise RuntimeError('Physical times differ')
            if not all(np.array_equal(ref[k],cand[k]) for k in ('x','y','z')):
                raise RuntimeError('Coordinates differ')
            common=ref['valid'].astype(bool)&cand['valid'].astype(bool)
            if not common.any(): raise RuntimeError('No common fluid points')
            x,y,z=np.meshgrid(ref['x'],ref['y'],ref['z'],indexing='ij')
            wake=common&(x>=bounds[0])&(x<=bounds[1])&(y>=bounds[2])&(y<=bounds[3])
            if not wake.any(): raise RuntimeError('Empty wake region')
            ur=ref['U'].astype(float); uc=cand['U'].astype(float)
            pr=ref['p'].astype(float); pc=cand['p'].astype(float)
            pr-=pr[common].mean(); pc-=pc[common].mean()
            values=np.array([np.sum((uc[common]-ur[common])**2),np.sum(ur[common]**2),
                             np.sum((pc[common]-pr[common])**2),np.sum(pr[common]**2),
                             np.sum((uc[wake]-ur[wake])**2),np.sum(ur[wake]**2)])
            snapshots.append({'comparison':kind,'tau':tau,
                'velocity_relative_l2':float(np.sqrt(values[0]/max(values[1],1e-30))),
                'pressure_relative_l2':float(np.sqrt(values[2]/max(values[3],1e-30))),
                'wake_velocity_relative_l2':float(np.sqrt(values[4]/max(values[5],1e-30))),
                'mask_disagreement_fraction':float(np.mean(ref['valid']!=cand['valid']))})
            for window in accum:
                if window[0]-1e-6<=tau<=window[1]+1e-6: accum[window]+=values
        for window,values in accum.items():
            grid=np.arange(window[0],window[1]+0.2,0.4)
            force_stats={}
            for label in (reference,candidate):
                a=forces[label]
                if a[0,0]>grid[0] or a[-1,0]<grid[-1]-0.1:
                    raise RuntimeError('Force history does not cover measurement window')
                # Last output can precede the terminal state by <10 solver steps.
                values_force=np.column_stack([np.interp(grid,a[:,0],a[:,j]) for j in (1,2)])
                force_stats[label]={'mean_cd':float(values_force[:,0].mean()),
                                    'mean_cl':float(values_force[:,1].mean()),
                                    'std_cd':float(values_force[:,0].std()),
                                    'std_cl':float(values_force[:,1].std())}
            fr,fc=force_stats[reference],force_stats[candidate]
            metrics={'velocity_relative_l2_global':float(np.sqrt(values[0]/max(values[1],1e-30))),
                     'velocity_relative_l2_wake':float(np.sqrt(values[4]/max(values[5],1e-30))),
                     'pressure_mean_aligned_relative_l2_global':float(np.sqrt(values[2]/max(values[3],1e-30))),
                     'mean_drag_coefficient_relative_difference':abs(fc['mean_cd']-fr['mean_cd'])/max(abs(fr['mean_cd']),1e-30),
                     'mean_lift_coefficient_absolute_difference':abs(fc['mean_cl']-fr['mean_cl'])}
            passed=all(metrics[key]<=limit for key_with_suffix,limit in thresholds.items()
                       for key in [key_with_suffix.removesuffix('_max')])
            results.append({'comparison':kind,'tau_window':window,'reference':reference,
                            'candidate':candidate,'metrics':metrics,'force_stats':force_stats,
                            'exploratory_screen_passed':bool(passed)})
    report={'status':'FIDELITY_WINDOWS_REVIEW_COMPLETE','job':args.job,
            'case_paths':{k:str(v) for k,v in cases.items()},
            'fine_case_override':args.fine_case is not None,
            'comparison_scope':'Smooth fine versus existing hard-transition coarse configuration; not pure h-only convergence' if args.fine_case is not None else 'Original paired mesh/time configurations',
            'wake_bounds_xy_m':bounds,'results':results,
            'flux_audits':flux_audits,
            'all_exploratory_screens_passed':all(r['exploratory_screen_passed'] for r in results) and all(a['passed'] for a in flux_audits.values()),
            'production_authorized_automatically':False,
            'limitations':gate['limitations']}
    (result/'fidelity_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (result/'snapshot_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(snapshots[0])); writer.writeheader(); writer.writerows(snapshots)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()

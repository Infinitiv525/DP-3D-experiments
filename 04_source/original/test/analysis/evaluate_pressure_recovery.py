"""Analyze a short, same-checkpoint pressure-control experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from diagnose_pressure_fields import raw_statistics, snapshot
from evaluate_fidelity_windows import force_series
from export_uniform_snapshot import find_internal_mesh


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args(); project=args.project
    result=project/f'results/dataset_a/v2_pressure_recovery_review/{args.job}'
    result.mkdir(parents=True,exist_ok=True)
    labels=['baseline','pressure_tight','coupling_tight','fine_half_dt']
    cases={label:project/f'dataset_a/v2/pressure_recovery/a2_mp_001_precovery_{args.job}_{label}' for label in labels}
    fields={}
    for label,case in cases.items():
        branch=project/f'results/dataset_a/v2_pressure_recovery/a2_mp_001_precovery_{args.job}_{label}'
        compact=branch/'compact'; compact.mkdir(exist_ok=True)
        available=[]
        for p in (case/'processor0').iterdir():
            try: t=float(p.name)
            except ValueError: continue
            if p.is_dir(): available.append((t,p.name))
        fields[label]={}
        for tau in (24,26,28):
            expected=tau*.01/.02375
            t,name=min(available,key=lambda item:abs(item[0]-expected))
            if abs(t-expected)>1e-6: raise RuntimeError('Missing pressure diagnostic time')
            if tau!=24:
                with (branch/'logs/log.reconstruct_export').open('a') as log:
                    subprocess.run(['reconstructPar','-case',str(case),'-time',name],stdout=log,stderr=subprocess.STDOUT,check=True)
            with (branch/'export_stdout.jsonl').open('a') as log:
                subprocess.run([sys.executable,str(project/'test/analysis/export_uniform_snapshot.py'),
                    '--case-dir',str(case),'--time',name,'--diameter','0.01','--scenario-id','a2_mp_001',
                    '--output-dir',str(compact),'--dimensions','128','64','32',
                    '--bounds','-0.05','0.15','-0.05','0.05','-0.02','0.02'],stdout=log,check=True)
            fields[label][tau]=snapshot(compact,tau)
    original=project/'dataset_a/v2/fidelity/a2_mp_001_fidelity_110147_fine_dt02'
    original_compact=project/'results/dataset_a/v2_fidelity/a2_mp_001_fidelity_110147_fine_dt02/compact'
    cases['original_fine']=original
    fields['original_fine']={tau:snapshot(original_compact,tau) for tau in (24,26,28)}
    original_cells={label:raw_statistics(case,28*.01/.02375) for label,case in cases.items()}
    rows=[]
    for tau in (24,26,28):
        baseline=fields['baseline'][tau]
        for label,by_tau in fields.items():
            data=by_tau[tau]
            if not all(np.array_equal(data[k],baseline[k]) for k in ('x','y','z')):
                raise RuntimeError('Coordinate mismatch')
            common=baseline['valid'].astype(bool)&data['valid'].astype(bool)
            pb=baseline['p'][common].astype(float); pp=data['p'][common].astype(float)
            pb-=pb.mean(); pp-=pp.mean()
            ub=baseline['U'][common].astype(float); uu=data['U'][common].astype(float)
            rows.append({'tau':tau,'label':label,
                'pressure_relative_l2_to_restarted_baseline':float(np.linalg.norm(pp-pb)/max(np.linalg.norm(pb),1e-30)),
                'pressure_rms_over_u2':float(np.sqrt(np.mean(pp**2))/.02375**2),
                'pressure_difference_rms_over_u2':float(np.sqrt(np.mean((pp-pb)**2))/.02375**2),
                'velocity_relative_l2_to_restarted_baseline':float(np.linalg.norm(uu-ub)/max(np.linalg.norm(ub),1e-30))})
    force_stats={}
    for label,case in cases.items():
        a=force_series(case); tau=np.arange(25,28.0001,.1)
        if a[0,0]>25 or a[-1,0]<27.9: raise RuntimeError('Incomplete force history')
        values=np.column_stack([np.interp(tau,a[:,0],a[:,i]) for i in (1,2)])
        force_stats[label]={'mean_cd':float(values[:,0].mean()),'mean_cl':float(values[:,1].mean()),
                            'std_cd':float(values[:,0].std()),'std_cl':float(values[:,1].std())}
    baseline_case=cases['baseline']; marker=baseline_case/'case.foam'
    reader=pv.OpenFOAMReader(str(marker))
    reader.set_active_time_value(min(reader.time_values,key=lambda t:abs(t-24*.01/.02375)))
    internal=find_internal_mesh(reader.read())
    p=np.asarray(internal.cell_data['p']); top=np.argsort(np.abs(p-p.mean()))[-10:][::-1]
    centers=internal.cell_centers().points
    quality_names=[name for name in internal.cell_data.keys()
                   if any(key in name.lower() for key in ('determinant','nonortho','skew','volume','weight','quality'))
                   and np.asarray(internal.cell_data[name]).ndim==1]
    quality=[]
    for i in top:
        values={name:float(internal.cell_data[name][i]) for name in quality_names}
        quality.append({'cell_index':int(i),'xyz':centers[i].tolist(),'p':float(p[i]),'quality':values})
    f0=fields['baseline'][28]; mask=np.logical_and.reduce([fields[k][28]['valid'].astype(bool) for k in cases])
    pressure={label:np.ma.array((fields[label][28]['p'].astype(float)-fields[label][28]['p'][mask].astype(float).mean())/.02375**2,mask=~mask) for label in cases}
    z=int(np.argmin(np.abs(f0['z'])))
    lim=float(np.quantile(np.abs(np.concatenate([a.compressed() for a in pressure.values()])),.99))
    errors={label:p-pressure['baseline'] for label,p in pressure.items()}
    elim=max(1e-12,float(np.quantile(np.abs(np.concatenate([a.compressed() for a in errors.values()])),.99)))
    fig,axes=plt.subplots(2,5,figsize=(18,6),constrained_layout=True)
    for col,label in enumerate(cases):
        for row,(p,vmax) in enumerate(((pressure[label],lim),(errors[label],elim))):
            im=axes[row,col].imshow(p[:,:,z].T,origin='lower',extent=(-.05,.15,-.05,.05),vmin=-vmax,vmax=vmax,cmap='RdBu_r')
            axes[row,col].set_xlabel('x [m]'); axes[row,col].set_ylabel('y [m]')
            fig.colorbar(im,ax=axes[row,col],shrink=.7)
        axes[0,col].set_title(label)
    fig.suptitle('tau=28: mean-aligned pressure / U_inlet^2 (top); difference from restarted baseline (bottom)\nShared row scales clip the largest 1%; local-response diagnostic, not production acceptance')
    fig.savefig(result/'pressure_recovery.png',dpi=150); plt.close(fig)
    report={'status':'PRESSURE_LOCAL_RESPONSE_REVIEW_COMPLETE','job':args.job,'snapshots':rows,
            'raw_statistics_tau28':original_cells,'force_stats_tau25_28':force_stats,
            'initial_pressure_extreme_cell_quality':quality,'available_quality_fields':quality_names,
            'production_accepted':False,'acceptance_thresholds_changed':False,
            'interpretation':'Compare response to same fine checkpoint; baseline versus original also includes restart/history effects. Lower extrema alone do not establish physical correctness.'}
    (result/'pressure_recovery.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()

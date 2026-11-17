"""Localize numerical pressure discrepancies without changing acceptance limits."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from export_uniform_snapshot import find_internal_mesh


def snapshot(root: Path, tau: float) -> dict:
    expected=tau*.01/.02375
    path=min(root.glob('*.npz'),key=lambda p:abs(float(p.stem.split('_t')[1])-expected))
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    if abs(float(data['time_s'])-expected)>1e-6:
        raise RuntimeError('Wrong physical time')
    if not np.isfinite(data['p']).all(): raise RuntimeError('Non-finite pressure')
    data['source']=str(path)
    return data


def raw_statistics(case: Path, physical_time: float) -> dict:
    reader=pv.OpenFOAMReader(str(case/'case.foam'))
    selected=min(reader.time_values,key=lambda t:abs(t-physical_time))
    if abs(selected-physical_time)>1e-6: raise RuntimeError('Raw time unavailable')
    reader.set_active_time_value(selected)
    internal=find_internal_mesh(reader.read())
    if 'p' not in internal.cell_data:
        raise RuntimeError('Raw cell pressure not found')
    pressure=np.asarray(internal.cell_data['p'],dtype=float)
    volumes=np.asarray(internal.compute_cell_sizes(length=False,area=False).cell_data['Volume'])
    centers=internal.cell_centers().points
    if not np.isfinite(pressure).all() or np.any(volumes<=0):
        raise RuntimeError('Invalid raw field or cell volume')
    mean=float(np.average(pressure,weights=volumes))
    centered=pressure-mean
    worst=np.argsort(np.abs(centered))[-5:][::-1]
    result={'cells':len(pressure),'volume_weighted_mean':mean,
            'volume_weighted_centered_rms_over_u2':float(np.sqrt(np.average(centered**2,weights=volumes))/.02375**2),
            'min_over_u2':float(centered.min()/.02375**2),
            'max_over_u2':float(centered.max()/.02375**2),
            'largest_absolute_cell_pressure':[{'xyz':centers[i].tolist(),'p_over_u2':float(centered[i]/.02375**2)} for i in worst]}
    del reader,internal,centers,volumes,pressure,centered
    gc.collect()
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fine-case',type=Path)
    parser.add_argument('--fine-compact',type=Path)
    args=parser.parse_args(); project=args.project; out=args.output
    out.mkdir(parents=True,exist_ok=True)
    cases={
        'fine':project/'dataset_a/v2/fidelity/a2_mp_001_fidelity_110147_fine_dt02',
        'coarse':project/'dataset_a/v2/full_cost_pilot/a2_mp_001_fullcost_109922',
        'coarse_half_dt':project/'dataset_a/v2/fidelity/a2_mp_001_fidelity_110147_coarse_dt01'}
    roots={
        'fine':project/'results/dataset_a/v2_fidelity/a2_mp_001_fidelity_110147_fine_dt02/compact',
        'coarse':project/'results/dataset_a/v2_full_cost_pilot/a2_mp_001_fullcost_109922/compact',
        'coarse_half_dt':project/'results/dataset_a/v2_fidelity/a2_mp_001_fidelity_110147_coarse_dt01/compact'}
    if (args.fine_case is None)!=(args.fine_compact is None):
        raise ValueError('Provide both fine-case and fine-compact overrides')
    if args.fine_case is not None:
        cases['fine']=args.fine_case
        roots['fine']=args.fine_compact
    rows=[]; panels=[]
    for tau in (12,24,40):
        data={k:snapshot(root,tau) for k,root in roots.items()}
        f=data['fine']; common=np.logical_and.reduce([d['valid'].astype(bool) for d in data.values()])
        for d in data.values():
            if not all(np.array_equal(f[a],d[a]) for a in ('x','y','z')):
                raise RuntimeError('Coordinate mismatch')
        x,y,z=np.meshgrid(f['x'],f['y'],f['z'],indexing='ij')
        pressures={k:d['p'].astype(float)-d['p'][common].astype(float).mean() for k,d in data.items()}
        error=pressures['coarse']-pressures['fine']
        squared=error[common]**2; count=max(1,int(np.ceil(len(squared)*.01)))
        regions={
            'whole_domain':common,
            'upstream_x_below_minus_0p02':common&(x<-.02),
            'obstacle_neighborhood':common&(x>=-.015)&(x<=.03)&(y>=-.015)&(y<=.035),
            'downstream_wake':common&(x>.03)&(x<.1)&(y>=-.02)&(y<=.035),
            'outlet_x_above_0p12':common&(x>.12)}
        stats={}
        for name,region in regions.items():
            pr=pressures['fine'][region]; er=error[region]
            if not len(pr): raise RuntimeError(f'Empty region {name}')
            stats[name]={'points':len(pr),'relative_l2':float(np.linalg.norm(er)/max(np.linalg.norm(pr),1e-30)),
                         'reference_rms_over_u2':float(np.sqrt(np.mean(pr**2))/.02375**2),
                         'error_rms_over_u2':float(np.sqrt(np.mean(er**2))/.02375**2),
                         'fraction_of_global_squared_error':float(np.sum(er**2)/max(squared.sum(),1e-30))}
        roughness={}
        interior=common[1:-1,1:-1,1:-1].copy()
        for axis in range(3):
            sl=[slice(1,-1)]*3; sl[axis]=slice(0,-2); interior &= common[tuple(sl)]
            sl[axis]=slice(2,None); interior &= common[tuple(sl)]
        for name,p in pressures.items():
            center=p[1:-1,1:-1,1:-1]
            neighbors=(p[:-2,1:-1,1:-1]+p[2:,1:-1,1:-1]+p[1:-1,:-2,1:-1]+p[1:-1,2:,1:-1]+p[1:-1,1:-1,:-2]+p[1:-1,1:-1,2:])/6
            roughness[name]={'six_neighbor_high_pass_rms_over_u2':float(np.sqrt(np.mean((center[interior]-neighbors[interior])**2))/.02375**2),
                             'centered_rms_over_u2':float(np.sqrt(np.mean(p[common]**2))/.02375**2)}
        worst=np.argsort(np.where(common,error**2,-1).ravel())[-10:][::-1]
        largest=[]
        for flat in worst:
            i,j,k=np.unravel_index(flat,common.shape)
            largest.append({'xyz':[float(f['x'][i]),float(f['y'][j]),float(f['z'][k])],
                            'fine_p_over_u2':float(pressures['fine'][i,j,k]/.02375**2),
                            'coarse_p_over_u2':float(pressures['coarse'][i,j,k]/.02375**2)})
        row={'tau':tau,'regions':stats,'roughness':roughness,
             'top_one_percent_points_fraction_of_squared_error':float(np.partition(squared,-count)[-count:].sum()/max(squared.sum(),1e-30)),
             'largest_error_points':largest,'sources':{k:d['source'] for k,d in data.items()}}
        rows.append(row)
        k=int(np.argmin(np.abs(f['z'])))
        fields=[pressures['fine'],pressures['coarse'],error,pressures['coarse']-pressures['coarse_half_dt']]
        panels.append((tau,float(f['z'][k]),[np.ma.array(a[:,:,k].T/.02375**2,mask=~common[:,:,k].T) for a in fields]))
    fig,axes=plt.subplots(3,4,figsize=(15,9),constrained_layout=True)
    scale_p=float(np.quantile(np.concatenate([p.compressed() for _,_,fields in panels for p in fields[:2]]),.99))
    scale_p=max(abs(scale_p),float(abs(np.quantile(np.concatenate([p.compressed() for _,_,fields in panels for p in fields[:2]]),.01))))
    scale_e=float(np.quantile(np.abs(np.concatenate([fields[2].compressed() for _,_,fields in panels])),.99))
    for i,(tau,plane,fields) in enumerate(panels):
        for j,p in enumerate(fields):
            lim=scale_p if j<2 else scale_e
            image=axes[i,j].imshow(p,origin='lower',extent=(-.05,.15,-.05,.05),vmin=-lim,vmax=lim,cmap='RdBu_r',aspect='equal')
            axes[i,j].set_xlabel('x [m]'); axes[i,j].set_ylabel(f'tau={tau}; y [m]')
            fig.colorbar(image,ax=axes[i,j],shrink=.65)
        if i==0:
            for j,title in enumerate(('Finer p','Coarser p','Coarse minus fine','Coarse Co=.2 minus Co=.1')): axes[i,j].set_title(title)
    fig.suptitle(f'Mean-aligned kinematic pressure / U_inlet^2; z={panels[0][1]:.6g} m\nDifference panels share a scale; color limits clip the largest 1% for visibility')
    fig.savefig(out/'pressure_diagnostic.png',dpi=150); plt.close(fig)
    raw={name:raw_statistics(case,40*.01/.02375) for name,case in cases.items()}
    report={'status':'PRESSURE_SPATIAL_DIAGNOSTIC_COMPLETE','snapshots':rows,'raw_cell_statistics_tau40':raw,
            'case_paths':{k:str(v) for k,v in cases.items()},
            'regions_may_overlap':True,'gauge':'Each raster field independently mean-centered on common fluid points.',
            'roughness_warning':'Six-neighbor raster high-pass response is descriptive, not a proof of numerical instability.',
            'raw_warning':'Raw statistics use volume-weighted mean; raster diagnostics use equal point weights.',
            'acceptance_limits_changed':False}
    (out/'pressure_diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for row in rows: print('tau',row['tau'],'regions',row['regions'],'roughness',row['roughness'],flush=True)
    print('RAW',raw,flush=True)


if __name__=='__main__': main()

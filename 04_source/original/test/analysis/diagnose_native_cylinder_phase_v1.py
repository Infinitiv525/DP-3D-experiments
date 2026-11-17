"""TRAIN-only diagnostic; lag alignment never replaces frozen acceptance metrics."""
import argparse
from pathlib import Path
import re
import numpy as np
from native_quality_case_v1 import ROOT,WORK as OLD,read,digest
from native_cylinder_refinement_v1 import WORK,verify
from evaluate_box_fidelity import histories
from evaluate_fidelity_windows import load
from native_force_endpoint_support_v2 import append_endpoint
from native_review_serialization_v1 import save


def lag_diagnostic(a,b):
    grid=np.arange(24.,36.0001,.02);lags=np.arange(-3.2,3.2001,.02);values=[]
    for lag in lags:
        x=np.interp(grid+lag,a[:,0],a[:,2]);y=np.interp(grid,b[:,0],b[:,2])
        x-=x.mean();y-=y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
        values.append(float(np.dot(x,y)/den) if den>1e-20 else None)
    if any(v is None for v in values):raise ValueError('Degenerate lift signal')
    best=int(np.argmax(values));zero=int(np.argmin(abs(lags)))
    return dict(window_tau=[24,36],max_lag_tau=3.2,lag_step_tau=.02,
        convention='positive lag compares branch(t+lag) to fine(t)',
        zero_lag_correlation=values[zero],best_correlation=values[best],
        best_lag_tau=float(lags[best]),best_at_search_boundary=bool(best in [0,len(lags)-1]),
        diagnostic_only=True,acceptance_metrics_replaced=False)


def run(job):
    verify();freeze=read(WORK/'diagnostic_freeze.json')
    for p,sha in freeze['files'].items():
        if digest(Path(p))!=sha:raise ValueError('Diagnostic source changed')
    out=WORK/'diagnostics'/job;out.mkdir(parents=True,exist_ok=False)
    sid='a2_sc_001';u=.0175;length=.0085
    cases={
        'factor1.2':ROOT/'dataset_a/v2/quality_extension_20261022_v1/a2_sc_001_coarse_126204',
        'factor1.1':ROOT/'dataset_a/v2/cylinder_refinement_20261023_v1/a2_sc_001_coarse_127873',
        'factor1.0':ROOT/'dataset_a/v2/quality_extension_20261022_v1/a2_sc_001_fine_126205'}
    endpoint_old=read(OLD/'force_endpoint_recovery/127805/summary.json')['endpoints']
    endpoints={'factor1.2':next(e for e in endpoint_old if e['scenario_id']==sid and e['branch']=='coarse'),
        'factor1.0':next(e for e in endpoint_old if e['scenario_id']==sid and e['branch']=='fine'),
        'factor1.1':read(WORK/'reviews_v2/a2_sc_001_128119/coarse/offline_endpoint/endpoint.json')}
    cache={'factor1.2':OLD/'family_reviews/a2_sc_001_126206/coarse/compact',
        'factor1.0':OLD/'family_reviews/a2_sc_001_126206/fine/compact',
        'factor1.1':WORK/'reviews/a2_sc_001_128112/coarse/compact'}
    forces={};fingerprints={};initial={};cadence={}
    names=['constant/transportProperties','constant/turbulenceProperties','system/fvSchemes','system/fvSolution','system/setExprFieldsDict']
    for label,case in cases.items():
        h,n=histories(case,'obstacleForces',True);h[:,0]*=u/length
        forces[label],cadence[label]=append_endpoint(h,endpoints[label],u,length)
        fingerprints[label]={name:digest(case/name) for name in names}
        text=(case/'system/controlDict').read_text()
        fingerprints[label]['force_normalization']={k:re.findall(r'\b'+k+r'\s+([^;]+);',text) for k in ['magUInf','lRef','Aref']}
        initial[label]=load(cache[label]/f'{sid}_t0.npz')
    ref=initial['factor1.0'];initial_metrics={}
    for label,arr in initial.items():
        for k in ['x','y','z']:
            if not np.array_equal(arr[k],ref[k]):raise ValueError('Initial export grids differ')
        mask=arr['valid'].astype(bool)&ref['valid'].astype(bool)
        v=arr['U'].astype(float);r=ref['U'].astype(float)
        initial_metrics[label]=dict(mask_disagreement_fraction=float(np.mean(arr['valid']!=ref['valid'])),
            velocity_relative_l2=float(np.linalg.norm(v[mask]-r[mask])/np.linalg.norm(r[mask])))
    grid=np.arange(4.,40.0001,.04)
    save(out/'force_curves.json',dict(tau=grid.tolist(),series={k:{'Cd':np.interp(grid,h[:,0],h[:,1]).tolist(),'Cl':np.interp(grid,h[:,0],h[:,2]).tolist()} for k,h in forces.items()}))
    save(out/'diagnostic.json',dict(status='TRAIN_CYLINDER_DIAGNOSTIC_COMPLETE',case_paths={k:str(v) for k,v in cases.items()},
        source_fingerprints=fingerprints,configuration_file_hashes_identical={n:len({f[n] for f in fingerprints.values()})==1 for n in names},
        initial_metrics=initial_metrics,lift_lag_diagnostics={k:lag_diagnostic(h,forces['factor1.0']) for k,h in forces.items() if k!='factor1.0'},
        endpoint_audits=cadence,limitations=['Correlation after lag shift is diagnostic only, not acceptance or a proof of convergence','Initial interpolation/mask differences do not by themselves prove bad physical geometry','Fine1.0 is a numerical comparator, not ground truth'],
        production_accepted=False,ML_test_fields_used=False,freeze_sha256=digest(WORK/'diagnostic_freeze.json')))
    print('TRAIN_CYLINDER_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);run(p.parse_args().job)

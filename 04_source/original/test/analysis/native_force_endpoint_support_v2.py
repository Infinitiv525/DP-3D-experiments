"""Use actual offline tau40 force, never extrapolate or relax fidelity tolerances."""
import numpy as np
from native_quality_case_v1 import WORK,read,digest


def verify_extension():
    p=WORK/'review_recovery_20261023_v2/freeze.json';f=read(p)
    for name,sha in f['source_files'].items():
        from pathlib import Path
        if digest(Path(name))!=sha:raise ValueError('Review recovery source changed')
    if digest(WORK/'force_endpoint_recovery/127805/summary.json')!=f['endpoint_summary_sha256']:
        raise ValueError('Endpoint provenance changed')
    return f


def append_endpoint(history,endpoint,u,length):
    h=np.asarray(history,dtype=float).copy()
    if h.ndim!=2 or h.shape[1]!=3 or len(h)<2 or not np.isfinite(h).all() or not (np.diff(h[:,0])>0).all():
        raise ValueError('Finite strictly increasing force history required')
    if abs(endpoint['time_s']*u/length-40)>1e-6 or endpoint['tau']!=40 or not endpoint['offline_only']:
        raise ValueError('Verified physical endpoint at tau40 required')
    point=np.array([40.,endpoint['Cd'],endpoint['Cl']],dtype=float)
    if not np.isfinite(point).all() or h[-1,0]>40+1e-6:raise ValueError('Invalid force endpoint')
    # Existing writeInterval10 x maxDeltaTau0.02 imposes an independent cadence bound.
    if (np.diff(h[:,0])>.2+1e-6).any() or 40-h[-1,0]>.2+1e-6:
        raise ValueError('Missing history exceeds configured sampling cadence')
    gap=max(0.,40-h[-1,0])
    if gap<=1e-8:
        if not np.allclose(h[-1,1:],point[1:],rtol=1e-6,atol=1e-8):raise ValueError('Offline endpoint disagrees with existing endpoint')
        h[-1,0]=40.
    else:h=np.vstack([h,point])
    return h,dict(original_end_tau=float(40-gap),appended_actual_endpoint=bool(gap>1e-8),
        endpoint_gap_tau=float(gap),method='OpenFOAM offline forceCoeffs from saved final fields; no solver time stepping')

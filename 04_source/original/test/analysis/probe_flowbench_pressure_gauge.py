"""Check whether solid/fluid interpolation commutes with a pressure-gauge change."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_flowbench_boundaries import selected_samples
from flowbench_resampling import resample_uniform


def center(y,fluid):
    v=np.array(y,dtype=np.float64,copy=True)
    v[3,fluid]-=v[3,fluid].mean()
    v[3,~fluid]=0
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    split=json.loads((a.root/'uniform64_endpoints/split.json').read_text())['indices']
    indices=[split['train'][0],split['validation'][0]]
    xs={i:v for i,v in selected_samples(a.root/'LDC_3d_X.npz',indices)}
    rows=[]
    for i,y in selected_samples(a.root/'LDC_3d_Y.npz',indices):
        x=xs[i];native=x[0]>0;coarse=resample_uniform(x);fluid=coarse[0]>0
        old=center(resample_uniform(y),fluid)
        fixed=center(resample_uniform(center(y,native)),fluid)
        # A pure gauge shift belongs to fluid pressure, not zero-valued solid padding.
        shifted=np.array(y,dtype=np.float64);shifted[3,native]+=37
        old_shift=center(resample_uniform(shifted),fluid)
        fixed_shift=center(resample_uniform(center(shifted,native)),fluid)
        row={'sample':i,'pressure_shift_added':37,'old_order_gauge_change_rmse':float(np.sqrt(np.mean((old_shift[3,fluid]-old[3,fluid])**2))),'native_center_first_gauge_change_rmse':float(np.sqrt(np.mean((fixed_shift[3,fluid]-fixed[3,fluid])**2))),'old_pressure_centered_rms':float(np.sqrt(np.mean(old[3,fluid]**2))),'native_center_first_pressure_centered_rms':float(np.sqrt(np.mean(fixed[3,fluid]**2))),'old_vs_fixed_rmse':float(np.sqrt(np.mean((fixed[3,fluid]-old[3,fluid])**2))),'regions':{}}
        for label,mask in [('near_surface_sdf_0_to_0p05',fluid&(coarse[0]<=.05)),('away_sdf_above_0p05',fluid&(coarse[0]>.05))]:
            row['regions'][label]={'points':int(mask.sum()),'old_vs_fixed_rmse':float(np.sqrt(np.mean((fixed[3,mask]-old[3,mask])**2)))}
        rows.append(row);print(json.dumps(row),flush=True)
    report={'status':'FLOWBENCH_PRESSURE_GAUGE_ORDER_PROBE_COMPLETE','samples':rows,'native_shape':[128]*3,'coarse_shape':[64]*3,'test_evaluated':False,'source_data_modified':False,'scope':'Two train/validation cases only; shift invariance is an algebraic preprocessing test, not a full physical accuracy claim.'}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'gauge_order_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

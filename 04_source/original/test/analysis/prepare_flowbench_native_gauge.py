"""Preserve velocity cache while gauge-centering native fluid pressure before interpolation."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from audit_flowbench_boundaries import selected_samples
from flowbench_resampling import resample_uniform
from prepare_flowbench import digest


def native_center(y,fluid,shift=0.):
    v=np.array(y,copy=True)
    p=y[3].astype(np.float64)
    p[fluid]+=shift
    p[fluid]-=p[fluid].mean()
    p[~fluid]=0
    v[3]=p
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    old=a.root/'uniform64_endpoints';out=a.root/'uniform64_native_gauge'
    audit=json.loads((old/'regrid_audit.json').read_text())
    assert audit['status']=='FLOWBENCH_UNIFORM64_CACHE_COMPLETE'
    assert digest(old/'X_pilot.npy')==audit['channels']['X']['sha256']
    assert digest(old/'Y_pilot.npy')==audit['channels']['Y']['sha256']
    out.mkdir(exist_ok=False);assert shutil.disk_usage(out).free>5_000_000_000
    old_y=np.load(old/'Y_pilot.npy',mmap_mode='r',allow_pickle=False)
    coarse_x=np.load(old/'X_pilot.npy',mmap_mode='r',allow_pickle=False)
    yout=np.lib.format.open_memmap(out/'Y_pilot.npy',mode='w+',dtype='float32',shape=(1000,4,64,64,64))
    max_velocity_delta=0.;max_gauge_delta=0.;rows=[]
    xgen=selected_samples(a.root/'LDC_3d_X.npz',range(1000))
    ygen=selected_samples(a.root/'LDC_3d_Y.npz',range(1000))
    for (i,x),(j,y) in zip(xgen,ygen,strict=True):
        assert i==j
        fluid=x[0]>0;new=resample_uniform(native_center(y,fluid))
        assert np.isfinite(new).all()
        vd=float(np.max(abs(new[:3]-old_y[i,:3])));max_velocity_delta=max(max_velocity_delta,vd)
        assert vd==0,'Velocity channels must stay bit-identical'
        # Check gauge invariance for every sample; this is deterministic preprocessing QC, not a model metric.
        shifted=resample_uniform(native_center(y,fluid,37.))
        gd=float(np.max(abs(shifted[3]-new[3])));max_gauge_delta=max(max_gauge_delta,gd)
        assert gd<1e-5
        mask=coarse_x[i,0]>0
        pp=new[3,mask].astype(float);pp-=pp.mean()
        rows.append({'sample':i,'centered_coarse_pressure_rms':float(np.sqrt(np.mean(pp**2))),'native_fluid_pressure_mean_removed':float(y[3,fluid].mean(dtype=np.float64))})
        yout[i]=new
        if i%100==0:print('corrected',i+1,'max_velocity_delta',max_velocity_delta,'max_gauge_delta',max_gauge_delta,flush=True)
    assert len(rows)==1000
    yout.flush();del yout
    (out/'X_pilot.npy').symlink_to((old/'X_pilot.npy').resolve())
    shutil.copyfile(old/'split.json',out/'split.json')
    report={'status':'FLOWBENCH_NATIVE_GAUGE64_CACHE_COMPLETE','source_revision':audit['source_revision'],'method':'Native SDF>0 fluid pressure mean removed and solid pressure set to zero BEFORE endpoint-aligned interpolation; coarse training gauge-centers again. Velocities and X unchanged.','channels':{'X':audit['channels']['X'],'Y':{'shape':[1000,4,64,64,64],'sha256':digest(out/'Y_pilot.npy')}},'split_sha256':digest(out/'split.json'),'velocity_max_absolute_change':max_velocity_delta,'max_pressure_shift37_invariance_error':max_gauge_delta,'sample_count':1000,'samples':rows,'test_model_metrics_computed':False,'production_accepted':False,'warning':'Pressure labels differ from legacy64. Old joint-model pilot pressure results are not final benchmark evidence; new training must start from scratch with fresh train-only output normalization.'}
    assert report['split_sha256']==audit['split_sha256']
    (out/'pressure_gauge_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['status'],flush=True)


if __name__=='__main__':main()

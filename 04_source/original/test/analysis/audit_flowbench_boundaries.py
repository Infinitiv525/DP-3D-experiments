"""Inspect native-resolution boundary conventions for two train/validation cases."""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
from prepare_flowbench import header, read_exact


def selected_samples(path, indices):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('.npy')]
        if len(names)!=1:raise ValueError('Expected one NPY member')
        with z.open(names[0]) as stream:
            shape,dtype=header(stream)
            count=int(np.prod(shape[1:]))*dtype.itemsize
            for i in range(max(indices)+1):
                payload=read_exact(stream,count)
                if i in indices:
                    yield i,np.frombuffer(payload,dtype=dtype).reshape(shape[1:])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    split=json.loads((a.root/'pilot32/split.json').read_text())['indices']
    chosen=[split['train'][0],split['validation'][0]]
    assert not set(chosen)&set(split['test'])
    a.output.mkdir(parents=True,exist_ok=False)
    inputs={i:v for i,v in selected_samples(a.root/'LDC_3d_X.npz',chosen)}
    rows=[]
    for i,y in selected_samples(a.root/'LDC_3d_Y.npz',chosen):
        x=inputs[i];fluid=x[0]>0;speed=np.linalg.norm(y[:3],axis=0)
        row={'sample':i,'split':'train' if i in split['train'] else 'validation','Re':float(x[1,0,0,0]),'faces':{},'interior_crops':[]}
        for axis in range(3):
            for side,index in [('low',0),('high',-1)]:
                q=np.take(y,index,axis=axis+1);m=np.take(fluid,index,axis=axis)
                row['faces'][f'axis{axis}_{side}']={'points':int(m.sum()),'pressure_exact_zero_fraction_fluid':float(np.mean(q[3][m]==0)),'velocity_rms_components':np.sqrt(np.mean(q[:3]**2,axis=(1,2))).tolist(),'pressure_range':[float(q[3].min()),float(q[3].max())]}
        for crop in (0,1,2,4,8):
            valid=fluid.copy()
            if crop:
                for axis in range(3):
                    sl=[slice(None)]*3;sl[axis]=slice(0,crop);valid[tuple(sl)]=False
                    sl[axis]=slice(-crop,None);valid[tuple(sl)]=False
            pressure=y[3][valid].astype(float);centered=pressure-pressure.mean()
            row['interior_crops'].append({'boundary_layers_removed':crop,'fluid_points':int(valid.sum()),'pressure_mean':float(pressure.mean()),'centered_pressure_rms':float(np.sqrt(np.mean(centered**2))),'pressure_exact_zero_fraction':float(np.mean(pressure==0)),'centered_pressure_abs_quantiles':np.quantile(abs(centered),[.5,.95,.99,1]).tolist(),'velocity_rms':float(np.sqrt(np.mean(speed[valid]**2)))})
        native_mu=float(y[3][fluid].mean(dtype=np.float64));coarse_mask=x[0,::4,::4,::4]>0
        coarse=y[3,::4,::4,::4];coarse_mu=float(coarse[coarse_mask].mean(dtype=np.float64))
        row['native_vs_stride4_pressure_mean_difference']=coarse_mu-native_mu
        rows.append(row);print(json.dumps(row),flush=True)
    report={'status':'FLOWBENCH_NATIVE_BOUNDARY_AUDIT_COMPLETE','selected_indices':chosen,'native_shape':[128,128,128],'samples':rows,'test_targets_evaluated':False,'stream_note':'ZIP decompression traverses earlier records but only selected train/validation records are interpreted; full archives were already SHA256-verified.','warning':'Two-case descriptive audit, not universal mask certification. Zero boundary values are observed, not automatically declared invalid or deleted.','production_accepted':False}
    (a.output/'boundary_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

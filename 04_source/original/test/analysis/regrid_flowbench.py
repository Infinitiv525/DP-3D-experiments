"""Endpoint-preserving uniform 64^3 cache; native archives and first pilot untouched."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
from flowbench_resampling import resample_uniform
from prepare_flowbench import header, read_exact, digest

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
old=a.root/'pilot32';audit=json.loads((old/'audit.json').read_text())
assert audit['status']=='STRUCTURAL_AUDIT_COMPLETE_NOT_PHYSICS_ACCEPTANCE'
out=a.root/'uniform64_endpoints';out.mkdir(exist_ok=False)
assert shutil.disk_usage(out).free>8_000_000_000
report={'status':'UNIFORM64_PREPARATION_IN_PROGRESS','source_revision':audit['revision'],'resolution':[64]*3,'method':'NumPy separable trilinear interpolation, coordinates linspace(0,127,64); endpoints aligned; no anti-aliasing filter','physical_acceptance':False,'test_model_metrics_computed':False,'channels':{}}
for role in ('X','Y'):
    archive=a.root/f'LDC_3d_{role}.npz'
    with zipfile.ZipFile(archive) as z:
        members=[n for n in z.namelist() if n.endswith('.npy')];assert len(members)==1
        with z.open(members[0]) as stream:
            shape,dtype=header(stream);assert list(shape)==audit['arrays'][role]['shape']
            cache=np.lib.format.open_memmap(out/f'{role}_pilot.npy',mode='w+',dtype='float32',shape=(shape[0],shape[1],64,64,64))
            bytes_per_sample=int(np.prod(shape[1:]))*dtype.itemsize
            endpoint_max=0.0
            for i in range(shape[0]):
                sample=np.frombuffer(read_exact(stream,bytes_per_sample),dtype=dtype).reshape(shape[1:])
                v=resample_uniform(sample)
                assert v.shape==(shape[1],64,64,64) and np.isfinite(v).all()
                for ix in (0,-1):
                    for iy in (0,-1):
                        for iz in (0,-1):endpoint_max=max(endpoint_max,float(np.max(abs(v[:,ix,iy,iz]-sample[:,ix,iy,iz]))))
                if role=='X':assert np.allclose(v[1],sample[1,0,0,0],rtol=0,atol=1e-5)
                if role=='Y':assert np.allclose(v[0,:,-1,:],sample[0,:,-1,:][0,0],rtol=0,atol=1e-6) and np.allclose(sample[0,:,-1,:],1,atol=1e-6)
                cache[i]=v
                if i%100==0:print(role,'regridded',i+1,flush=True)
            assert not stream.read(1)
            cache.flush();del cache
    assert endpoint_max<1e-6
    report['channels'][role]={'shape':[shape[0],shape[1],64,64,64],'corner_max_absolute_error':endpoint_max,'sha256':digest(out/f'{role}_pilot.npy')}
    (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copyfile(old/'split.json',out/'split.json')
report.update(status='FLOWBENCH_UNIFORM64_CACHE_COMPLETE',split_sha256=digest(out/'split.json'),warning='Linear resampling preserves domain endpoints and lid but not a proof of spatial fidelity, geometry resolution, or NS residual consistency. No final LLM experiment authorized automatically.')
(out/'regrid_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(report['status'],flush=True)

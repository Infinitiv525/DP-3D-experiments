"""Integrity, group isolation and descriptive train/validation visualization only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    audit=json.loads((a.data/'audit.json').read_text())
    split=json.loads((a.data/'split.json').read_text())
    assert audit['status']=='STRUCTURAL_AUDIT_COMPLETE_NOT_PHYSICS_ACCEPTANCE'
    ids=split['indices']; groups=split['geometry_groups']
    assert sorted(sum(ids.values(),[]))==list(range(1000))
    sets={k:{groups[i] for i in indices} for k,indices in ids.items()}
    assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['test']&sets['validation'])
    for role in ('X','Y'):
        assert sha(a.data/f'{role}_pilot.npy')==audit[f'{role}_pilot_sha256']
    x=np.load(a.data/'X_pilot.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(a.data/'Y_pilot.npy',mmap_mode='r',allow_pickle=False)
    # No test targets read. Select six train/validation geometries for descriptive QC.
    selected=[]
    for label in ('train','validation'):
        seen=set()
        for i in ids[label]:
            if groups[i] not in seen:
                selected.append((label,i));seen.add(groups[i])
            if len(seen)==3:break
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[];fig,axes=plt.subplots(len(selected),3,figsize=(11,2.8*len(selected)),constrained_layout=True)
    for j,(label,i) in enumerate(selected):
        sdf=np.asarray(x[i,0]);inside=sdf<=0;outside=~inside
        speed=np.linalg.norm(y[i,:3],axis=0)
        assert inside.any() and outside.any()
        row={'split':label,'index':i,'Re':float(x[i,1,0,0,0]),'solid_fraction':float(inside.mean()),'speed_rms_sdf_nonpositive':float(np.sqrt(np.mean(speed[inside]**2))),'speed_rms_sdf_positive':float(np.sqrt(np.mean(speed[outside]**2))),'fluid_pressure_mean':float(y[i,3][outside].mean()),'fluid_pressure_range':[float(y[i,3][outside].min()),float(y[i,3][outside].max())]}
        rows.append(row);k=sdf.shape[-1]//2
        fields=[sdf[:,:,k],speed[:,:,k],y[i,3,:,:,k]]
        for c,field in enumerate(fields):
            im=axes[j,c].imshow(field.T,origin='lower',cmap='RdBu_r' if c!=1 else 'viridis')
            axes[j,c].contour(sdf[:,:,k].T,levels=[0],colors='black',linewidths=.7)
            axes[j,c].set_title(f'{label} #{i}, Re={row["Re"]:g}: '+('SDF','speed','pressure')[c])
            axes[j,c].set_xlabel('x index (stride-4)');axes[j,c].set_ylabel('y index')
            fig.colorbar(im,ax=axes[j,c],shrink=.8)
    fig.suptitle('FlowBench 32^3 pilot central-z slices. Black: SDF=0. Not native-resolution physics acceptance.')
    fig.savefig(a.output/'reference_slices.png',dpi=120);plt.close(fig)
    report={'status':'FLOWBENCH_PRE_SMOKE_REVIEW_COMPLETE','cache_sha256_verified':True,'exact_geometry_split_disjoint':True,'groups_per_split':{k:len(v) for k,v in sets.items()},'samples':rows,'test_targets_inspected':False,'physical_acceptance':False,'warning':'SDF<=0 per updated author convention; descriptive coarse-grid checks only. Pressure gauges and near-duplicate families still need final audit.'}
    (a.output/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

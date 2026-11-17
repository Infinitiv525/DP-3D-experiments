"""Render validation-only coarse predictions already exported by GPU evaluator."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
paths=sorted(a.result.glob('validation_*.npz'));assert paths
fig,axes=plt.subplots(len(paths),4,figsize=(13,3*len(paths)),constrained_layout=True,squeeze=False)
for j,path in enumerate(paths):
    with np.load(path,allow_pickle=False) as d:
        sdf=d['sdf'];t=d['target'];v=d['prediction'];k=sdf.shape[-1]//2
        fields=[np.linalg.norm(t[:3],axis=0),np.linalg.norm(v[:3],axis=0),t[3],v[3]]
        speedlim=max(float(fields[0].max()),float(fields[1].max()))
        plim=max(float(np.abs(t[3][sdf>0]).max()),float(np.abs(v[3][sdf>0]).max()))
        for c,field in enumerate(fields):
            masked=np.ma.array(field[:,:,k].T,mask=sdf[:,:,k].T<=0)
            im=axes[j,c].imshow(masked,origin='lower',cmap='viridis' if c<2 else 'RdBu_r',vmin=0 if c<2 else -plim,vmax=speedlim if c<2 else plim)
            axes[j,c].set_title(path.stem+': '+('reference speed','predicted speed','reference centered p','predicted centered p')[c]);fig.colorbar(im,ax=axes[j,c],shrink=.8)
fig.suptitle(f'Validation only, {sdf.shape[0]}^3 pilot, central-z plane. Not a full-resolution acceptance.')
fig.savefig(a.result/'validation_predictions.png',dpi=120);plt.close(fig)

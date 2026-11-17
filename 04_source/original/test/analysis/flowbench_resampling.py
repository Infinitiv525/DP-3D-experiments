"""Dependency-free separable trilinear interpolation with aligned endpoints."""
import numpy as np


def resample_uniform(sample, size=64):
    result=sample
    for axis in (1,2,3):
        coords=np.linspace(0,result.shape[axis]-1,size)
        lo=np.floor(coords).astype(int);hi=np.minimum(lo+1,result.shape[axis]-1)
        shape=[1]*4;shape[axis]=size
        weight=(coords-lo).astype(np.float32).reshape(shape)
        low=np.take(result,lo,axis=axis)
        result=low+(np.take(result,hi,axis=axis)-low)*weight
    return result.astype(np.float32,copy=False)

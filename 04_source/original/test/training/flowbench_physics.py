"""Masked central-FD incompressibility penalty in normalized unit-cube coordinates."""
import torch


def divergence_values(velocity, fluid):
    if velocity.shape[1]!=3 or fluid.shape[1]!=1:raise ValueError('Invalid channels')
    n=velocity.shape[-1]
    if velocity.shape[-3:]!=(n,n,n):raise ValueError('Require cubic uniform grid')
    mask=fluid[:,0,1:-1,1:-1,1:-1].clone()
    for axis in range(3):
        for side in (slice(None,-2),slice(2,None)):
            sl=[slice(None),0,slice(1,-1),slice(1,-1),slice(1,-1)];sl[axis+2]=side
            mask &= fluid[tuple(sl)]
    div=(velocity[:,0,2:,1:-1,1:-1]-velocity[:,0,:-2,1:-1,1:-1]
         +velocity[:,1,1:-1,2:,1:-1]-velocity[:,1,1:-1,:-2,1:-1]
         +velocity[:,2,1:-1,1:-1,2:]-velocity[:,2,1:-1,1:-1,:-2])*(.5*(n-1))
    if not mask.any():raise ValueError('Empty fluid stencil')
    return div[mask]


def relative_divergence_penalty(prediction, reference, fluid, reference_mse_floor=1e-4):
    pd=divergence_values(prediction[:,:3],fluid)
    rd=divergence_values(reference[:,:3],fluid)
    return pd.square().mean()/rd.square().mean().clamp_min(reference_mse_floor)

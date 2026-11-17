"""Frozen-draft metric/loss definitions, shared by trainer and CPU tests."""
import math
from collections import defaultdict

import torch


def weighted_data_loss(prediction, target, pressure_weight):
    if not math.isfinite(pressure_weight) or pressure_weight<=0:
        raise ValueError('Positive finite pressure weight required')
    error=(prediction-target).square()
    return (3*error[:,:3].mean()+pressure_weight*error[:,3:].mean())/(3+pressure_weight)


def validate_hyperparameters(tc):
    for k,lo,hi in [('learning_rate',1e-4,3e-3),('weight_decay',1e-6,1e-2)]:
        v=tc[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError('Hyperparameter outside draft space: '+k)
    for k,choices in [('pressure_weight',[.25,.5,1,2,4]),('divergence_weight',[0,.001,.003,.01,.03])]:
        if isinstance(tc[k],bool) or tc[k] not in choices:raise ValueError('Invalid category: '+k)


def fluid_errors(prediction, target, fluid, epsilon=1e-12):
    """Single sample, physical units, float64 reductions and pressure gauge."""
    if prediction.shape!=target.shape or prediction.shape[0]!=4 or not fluid.any():
        raise ValueError('Invalid metric shapes or empty fluid')
    p=prediction[:,fluid].to(torch.float64)
    t=target[:,fluid].to(torch.float64)
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError('Nonfinite metric input')
    # No mutation of original predicted or reference tensors.
    p=torch.cat((p[:3],p[3:]-p[3:].mean()))
    t=torch.cat((t[:3],t[3:]-t[3:].mean()))
    unorm=torch.linalg.vector_norm(t[:3]);pnorm=torch.linalg.vector_norm(t[3])
    eu=torch.linalg.vector_norm(p[:3]-t[:3])/unorm.clamp_min(epsilon)
    ep=torch.linalg.vector_norm(p[3]-t[3])/pnorm.clamp_min(epsilon)
    return {'velocity_error':float(eu),'pressure_error':float(ep),
            'velocity_norm':float(unorm),'pressure_norm':float(pnorm),
            'velocity_below_epsilon':bool(unorm<epsilon),'pressure_below_epsilon':bool(pnorm<epsilon)}


def aggregate_geometry(rows):
    if not rows:raise ValueError('Empty validation')
    grouped=defaultdict(list)
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ('velocity_error','pressure_error')):
            raise ValueError('Nonfinite score')
        grouped[row['geometry']].append(row)
    eu=sum(sum(v['velocity_error'] for v in group)/len(group) for group in grouped.values())/len(grouped)
    ep=sum(sum(v['pressure_error'] for v in group)/len(group) for group in grouped.values())/len(grouped)
    return {'S':0.5*(eu+ep),'velocity_error':eu,'pressure_error':ep,
            'geometry_count':len(grouped),'sample_count':len(rows),
            'geometry_sample_counts':{k:len(v) for k,v in grouped.items()}}


def validate_split(split):
    ids=split['indices'];groups=split['geometry_groups']
    seen=set();seen_groups=set()
    for role in ('train','validation','test'):
        values=ids[role]
        if not values or len(values)!=len(set(values)) or seen.intersection(values):
            raise ValueError('Duplicate/overlapping/empty split')
        if any(type(i) is not int or not 0<=i<len(groups) for i in values):
            raise ValueError('Invalid sample ID')
        role_groups={groups[i] for i in values}
        if seen_groups.intersection(role_groups):raise ValueError('Geometry leakage')
        seen.update(values);seen_groups.update(role_groups)


class TrainValidationOnly:
    def __init__(self, array, split):
        self.array=array
        self.allowed=set(split['indices']['train'])|set(split['indices']['validation'])
    def __getitem__(self, index):
        if index not in self.allowed:raise PermissionError('Test data access is forbidden')
        return self.array[index]

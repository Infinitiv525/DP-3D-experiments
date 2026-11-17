"""Native geometry/split audit and train-validation label QC, no test Y analysis."""
import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from prepare_flowbench import header,read_exact,digest
from prepare_flowbench_native_gauge import native_center
from flowbench_resampling import resample_uniform


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def stream_samples(path,allowed):
    with zipfile.ZipFile(path) as archive:
        members=[n for n in archive.namelist() if n.endswith('.npy')]
        if len(members)!=1:raise ValueError('Single NPY expected')
        with archive.open(members[0]) as stream:
            shape,dtype=header(stream)
            if shape[0]!=1000 or shape[2:]!=(128,128,128):raise ValueError('Unexpected native shape')
            size=int(np.prod(shape[1:]))*dtype.itemsize
            for i in range(shape[0]):
                raw=read_exact(stream,size)
                if i in allowed:yield i,np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            if stream.read(1):raise ValueError('Trailing native array payload')


def packed(mask):return np.packbits(mask).tobytes()


def boundary_stats(velocity,fluid):
    rows=[]
    for axis in range(3):
        for side,index in (('low',0),('high',-1)):
            face=np.take(velocity,index,axis=axis+1)[:,1:-1,1:-1]
            mask=np.take(fluid,index,axis=axis)[1:-1,1:-1]
            expected=np.zeros(3)
            if axis==1 and side=='high':expected[0]=1
            if not mask.any():
                rows.append({'face':f'{axis}_{side}','fluid_points':0,'max_absolute_error':None});continue
            error=face[:,mask]-expected[:,None]
            rows.append({'face':f'{axis}_{side}','fluid_points':int(mask.sum()),
                         'max_absolute_error':float(np.max(np.abs(error))),
                         'normal_component_max_abs':float(np.max(np.abs(face[axis,mask])))})
    return rows


def div_rms(velocity,fluid):
    stencil=fluid[1:-1,1:-1,1:-1].copy()
    for axis in range(3):
        for side in (slice(None,-2),slice(2,None)):
            sl=[slice(1,-1)]*3;sl[axis]=side;stencil &= fluid[tuple(sl)]
    div=(velocity[0,2:,1:-1,1:-1]-velocity[0,:-2,1:-1,1:-1]
        +velocity[1,1:-1,2:,1:-1]-velocity[1,1:-1,:-2,1:-1]
        +velocity[2,1:-1,1:-1,2:]-velocity[2,1:-1,1:-1,:-2])*(.5*(fluid.shape[0]-1))
    selected=div[stencil].astype(np.float64)
    if not selected.size:raise ValueError('Empty fluid stencil')
    return float(np.sqrt(np.mean(selected**2)))


def main(a):
    started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());source_cfg=json.loads(a.source_config.read_text())
    if source_cfg['revision']!=cfg['revision']:raise ValueError('Pinned revision mismatch')
    coarse=a.data/'uniform64_native_gauge';audit=json.loads((coarse/'pressure_gauge_audit.json').read_text())
    if audit['status']!='FLOWBENCH_NATIVE_GAUGE64_CACHE_COMPLETE':raise ValueError('Corrected cache required')
    split=json.loads((coarse/'split.json').read_text());ids=split['indices'];groups=split['geometry_groups']
    if digest(coarse/'split.json')!=audit['split_sha256']:raise ValueError('Split hash differs')
    memberships={}
    for role,expected in cfg['expected_split_samples'].items():
        if len(ids[role])!=expected:raise ValueError('Split size differs')
        for i in ids[role]:
            if i in memberships:raise ValueError('Sample split overlap')
            memberships[i]=role
    if set(memberships)!=set(range(1000)):raise ValueError('Split coverage differs')
    group_roles={}
    for i,g in enumerate(groups):
        if g in group_roles and group_roles[g]!=memberships[i]:raise ValueError('Exact geometry leakage')
        group_roles[g]=memberships[i]
    counts=Counter(groups)
    if len(counts)!=cfg['expected_geometry_groups'] or set(counts.values())!={10}:raise ValueError('Geometry group count differs')
    integrity={}
    for role in ('X','Y'):
        path=a.data/Path(source_cfg['files'][role]['path']).name
        if cfg['verify_archive_sha256']:
            value=digest(path)
            if value!=source_cfg['files'][role]['sha256']:raise ValueError('Native archive SHA256 mismatch')
            integrity['native_'+role]=value
        value=digest(coarse/f'{role}_pilot.npy')
        if value!=audit['channels'][role]['sha256']:raise ValueError('Coarse cache SHA256 mismatch')
        integrity['coarse_'+role]=value
    save(a.output/'integrity.json',integrity)
    x64=np.load(coarse/'X_pilot.npy',mmap_mode='r',allow_pickle=False)
    y64=np.load(coarse/'Y_pilot.npy',mmap_mode='r',allow_pickle=False)
    reps={};input_rows=[];native_masks={};max_x_delta=0.
    native_x=a.data/Path(source_cfg['files']['X']['path']).name
    for i,x in stream_samples(native_x,set(range(1000))):
        if x.shape!=(2,128,128,128) or not np.isfinite(x).all():raise ValueError('Invalid native X')
        mask=x[0]<=0;bits=packed(mask);ghash=hashlib.sha256(bits).hexdigest()
        if ghash!=groups[i]:raise ValueError('Native mask differs from frozen split hash')
        re=float(x[1,0,0,0])
        if float(np.ptp(x[1]))>1e-5 or not 10<=re<=1000:raise ValueError('Re semantics fail')
        delta=float(np.max(np.abs(resample_uniform(x)-x64[i])));max_x_delta=max(max_x_delta,delta)
        if delta!=0:raise ValueError('Cache input recomputation differs')
        sdfhash=hashlib.sha256(x[0].tobytes()).hexdigest()
        coarsehash=hashlib.sha256(x64[i,0].tobytes()).hexdigest()
        if ghash not in reps:
            occupied=np.nonzero(mask)
            reps[ghash]={'label':f'g{len(reps):03}','native_hash':ghash,'sample':i,'role':memberships[i],
                'solid_voxels_native':int(mask.sum()),'solid_voxels_coarse':int((x64[i,0]<=0).sum()),
                'centroid_unit_cube':[float(v.mean()/127) for v in occupied],
                'native_sdf_hash':sdfhash,'coarse_sdf_hash':coarsehash,'Re':[]}
            native_masks[ghash]=np.frombuffer(bits,dtype=np.uint8)
        elif sdfhash!=reps[ghash]['native_sdf_hash'] or coarsehash!=reps[ghash]['coarse_sdf_hash']:
            raise ValueError('Within-group SDF changes across Re; inspect semantics')
        reps[ghash]['Re'].append(re)
        input_rows.append({'sample':i,'role':memberships[i],'Re':re,'group':reps[ghash]['label']})
        if i%100==0:print('INPUTS_CHECKED',i+1,flush=True)
    representatives=list(reps.values());lut=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    pairs=[];exact_coarse_inputs=[]
    for ia,ra in enumerate(representatives):
        for rb in representatives[ia+1:]:
            ma=native_masks[ra['native_hash']];mb=native_masks[rb['native_hash']]
            inter=int(lut[np.bitwise_and(ma,mb)].sum(dtype=np.uint64))
            union=ra['solid_voxels_native']+rb['solid_voxels_native']-inter
            ca=x64[ra['sample'],0]<=0;cb=x64[rb['sample'],0]<=0
            ci=int(np.logical_and(ca,cb).sum());cu=int(np.logical_or(ca,cb).sum())
            row={'a':ra['label'],'b':rb['label'],'role_a':ra['role'],'role_b':rb['role'],
                 'cross_split':ra['role']!=rb['role'],'native_solid_iou':inter/union,
                 'coarse_solid_iou':ci/cu if cu else 1.,
                 'centroid_distance':float(np.linalg.norm(np.array(ra['centroid_unit_cube'])-rb['centroid_unit_cube']))}
            pairs.append(row)
            if row['cross_split'] and ra['coarse_sdf_hash']==rb['coarse_sdf_hash']:exact_coarse_inputs.append(row)
    near=[p for p in pairs if p['cross_split'] and p['native_solid_iou']>=cfg['native_near_duplicate_solid_iou']]
    coarse_near=[p for p in pairs if p['cross_split'] and p['coarse_solid_iou']>=cfg['coarse_near_duplicate_solid_iou']]
    save(a.output/'geometry_pairs.json',pairs);save(a.output/'geometry_representatives.json',representatives)
    save(a.output/'input_metadata.json',input_rows)
    geometry={'status':'NATIVE_INPUT_GEOMETRY_AUDIT_COMPLETE','pair_count':len(pairs),
        'native_cross_split_near_flags':near,'coarse_cross_split_near_flags':coarse_near,
        'exact_cross_split_coarse_sdf_collisions':exact_coarse_inputs,
        'broad_cross_split_similarity_count':sum(p['cross_split'] and p['native_solid_iou']>=cfg['broad_similarity_solid_iou'] for p in pairs),
        'largest_cross_split_native_iou':max(p['native_solid_iou'] for p in pairs if p['cross_split']),
        'max_input_recompute_delta':max_x_delta,'test_Y_analyzed':False}
    save(a.output/'geometry_audit.json',geometry);print('GEOMETRY_AUDIT_COMPLETE',len(near),len(coarse_near),flush=True)
    # Read native Y only for train/validation; ZIP traversal itself is not field analysis.
    allowed=set(ids['train'])|set(ids['validation']);native_y=a.data/Path(source_cfg['files']['Y']['path']).name
    selected_x={}
    # Avoid retaining 1000 large inputs: store only 100 native masks and recover by group.
    bool_masks={g:np.unpackbits(bits)[:128**3].reshape(128,128,128).astype(bool) for g,bits in native_masks.items()}
    rows=[];wall_failed=[];max_y_delta=0.;degenerate=[]
    for i,y in stream_samples(native_y,allowed):
        if y.shape!=(4,128,128,128) or not np.isfinite(y).all():raise ValueError('Invalid native train/val Y')
        fluid=~bool_masks[groups[i]];fluid64=x64[i,0]>0
        expected=resample_uniform(native_center(y,fluid))
        delta=float(np.max(np.abs(expected-y64[i])));max_y_delta=max(max_y_delta,delta)
        if delta!=0:raise ValueError('Cache target recomputation differs')
        native_faces=boundary_stats(y[:3],fluid);coarse_faces=boundary_stats(y64[i,:3],fluid64)
        native_wall=max((v['max_absolute_error'] or 0) for v in native_faces)
        coarse_wall=max((v['max_absolute_error'] or 0) for v in coarse_faces)
        pp=y[3,fluid].astype(np.float64);pp-=pp.mean()
        cp=y64[i,3,fluid64].astype(np.float64);cp-=cp.mean()
        native_pnorm=float(np.linalg.norm(pp));coarse_pnorm=float(np.linalg.norm(cp))
        if min(native_pnorm,coarse_pnorm)<cfg['relative_error_denominator_epsilon']:degenerate.append(i)
        if max(native_wall,coarse_wall)>cfg['boundary_velocity_absolute_tolerance']:wall_failed.append(i)
        rows.append({'sample':i,'role':memberships[i],'native_faces':native_faces,'coarse_faces':coarse_faces,
            'native_pressure_norm_centered':native_pnorm,'coarse_pressure_norm_centered':coarse_pnorm,
            'native_reference_divergence_rms_unit_cube':div_rms(y[:3],fluid),
            'coarse_reference_divergence_rms_unit_cube':div_rms(y64[i,:3],fluid64),
            'native_solid_velocity_max_abs':float(np.max(np.abs(y[:3,~fluid]))),
            'coarse_solid_velocity_max_abs':float(np.max(np.abs(y64[i,:3,~fluid64]))) if (~fluid64).any() else None})
        if len(rows)%100==0:print('TRAIN_VAL_LABELS_CHECKED',len(rows),flush=True)
    if len(rows)!=850 or {r['sample'] for r in rows}!=allowed:raise ValueError('Field audit membership differs')
    save(a.output/'train_validation_label_audit.json',rows)
    checks={'archive_and_cache_integrity':True,'exact_geometry_split_no_leakage':True,
        'no_exact_coarse_sdf_collisions_across_splits':not exact_coarse_inputs,
        'no_flagged_near_duplicates_across_splits':not near and not coarse_near,
        'reference_lid_and_walls_within_tolerance':not wall_failed,'nondegenerate_pressure_norm':not degenerate,
        'native_to_coarse_recomputation_exact':max_x_delta==0 and max_y_delta==0,'no_test_target_analysis':True}
    report={'status':'FLOWBENCH_DATA_AUDIT_PASSED' if all(checks.values()) else 'FLOWBENCH_DATA_AUDIT_REVIEW_REQUIRED',
        'checks':checks,'geometry':geometry,'label_sample_count':len(rows),'boundary_failed_samples':wall_failed,
        'degenerate_pressure_samples':degenerate,'max_target_recompute_delta':max_y_delta,
        'max_native_wall_velocity_error':max((f['max_absolute_error'] or 0) for r in rows for f in r['native_faces']),
        'max_coarse_wall_velocity_error':max((f['max_absolute_error'] or 0) for r in rows for f in r['coarse_faces']),
        'reference_native_divergence_mean':float(np.mean([r['native_reference_divergence_rms_unit_cube'] for r in rows])),
        'reference_coarse_divergence_mean':float(np.mean([r['coarse_reference_divergence_rms_unit_cube'] for r in rows])),
        'max_native_solid_velocity':max(r['native_solid_velocity_max_abs'] for r in rows),
        'max_coarse_solid_velocity':max(r['coarse_solid_velocity_max_abs'] or 0 for r in rows),
        'config':cfg,'split_sha256':audit['split_sha256'],'elapsed_seconds':time.perf_counter()-started,
        'test_targets_analyzed':False,'model_test_metrics_evaluated':False,
        'warning':'QC/flag thresholds, not mesh convergence or proof of family-level OOD. Solid interpolation and nonzero reference FD divergence remain limitations. No input/split/target files were modified.'}
    save(a.output/'summary.json',report)
    lines=['AUDIT FLOWBENCH DÁT',report['status'],
           'Natívne vstupy všetkých1000vzoriek; cieľové polia iba700train+150validation.',
           'Hodnotenie modelu na testovej sade sa nevykonalo. Pôvodné dáta a rozdelenie sa nemenili.','']
    lines += [f'{k}: {v}' for k,v in checks.items()]
    lines += ['',f'Najvyšší solid IoU medzi rôznymi splitmi: {geometry["largest_cross_split_native_iou"]:.6f}',
              f'Natívne near-duplicate príznaky: {len(near)}, coarse: {len(coarse_near)}.',report['warning']]
    (a.output/'vysledok.txt').write_text('\n'.join(lines)+'\n');print(report['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

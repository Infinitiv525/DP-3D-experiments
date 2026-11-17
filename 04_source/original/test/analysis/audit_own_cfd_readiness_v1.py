"""Read-only audit: no arrays/checkpoints downloaded, source indexes never rewritten."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np


def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def headers(path):
    result={}
    with zipfile.ZipFile(path) as z:
        for k in ['U','p','valid','x','y','z']:
            with z.open(k+'.npy') as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
                else:raise ValueError('Unsupported NPY header version')
                if dtype.hasobject:raise ValueError('Object dtype forbidden')
                result[k]=shape
    if result!={'U':(128,64,32,3),'p':(128,64,32),'valid':(128,64,32),'x':(128,),'y':(64,),'z':(32,)}:
        raise ValueError('Unexpected field/header dimensions')


def main(root,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    cfg=read(root/'test/config/native_expansion_20261022_v1.json')
    table=root/'test/config/dataset_a_pilot_scenarios.csv'
    with table.open(newline='') as f:scenarios=list(csv.DictReader(f))
    if len(scenarios)!=12 or len({r['scenario_id'] for r in scenarios})!=12:raise ValueError('Pilot manifest differs')
    pilot=root/cfg['pilot_dataset_relative'];reports=[];errors=[];hashes={str(table):digest(table)}
    seen_params={};all_frames=0
    for r in scenarios:
        sid=r['scenario_id'];role=r['split'];folder=pilot/sid
        params=tuple(float(r[k]) for k in ['u_m_s','diameter_m','center_x_m','center_y_m'])
        if params in seen_params and seen_params[params]!=role:raise ValueError('Identical scenario parameters across roles')
        seen_params[params]=role
        index=folder/'series_index.csv';summary=read(folder/'series_summary.json');hashes[str(index)]=digest(index)
        with index.open(newline='') as f:rows=list(csv.DictReader(f))
        rows.sort(key=lambda x:float(x['time_s']))
        times=np.array([float(x['time_s']) for x in rows])
        if len(rows)!=201 or not np.allclose(times,160+np.arange(201)*.2,rtol=0,atol=1e-6):raise ValueError('Pilot schedule differs')
        if len({x['filename'] for x in rows})!=201:raise ValueError('Repeated snapshot filenames')
        observed=set(p.name for p in folder.glob('*.npz'))
        if observed!={x['filename'] for x in rows}:raise ValueError('Missing or unindexed frames')
        snapshots=[];axes=None;mask=None;finite_count=0
        for row in rows:
            p=folder/row['filename']
            if p.resolve().parent!=folder.resolve():raise ValueError('Index path escapes scenario')
            if p.stat().st_size!=int(row['bytes']) or digest(p)!=row['sha256']:
                errors.append(str(p)+': hash/size mismatch');continue
            headers(p)
            if role in ['train','validation']:
                with np.load(p,allow_pickle=False) as a:
                    if not np.isfinite(a['U']).all() or not np.isfinite(a['p']).all():raise ValueError('Nonfinite native fields')
                    current=tuple(a[k] for k in ['x','y','z'])
                    if axes is None:
                        axes=tuple(v.copy() for v in current);mask=a['valid'].astype(bool).copy()
                        if any(not np.isfinite(v).all() or not (np.diff(v)>0).all() for v in axes):raise ValueError('Invalid axes')
                    if not all(np.array_equal(x,y) for x,y in zip(axes,current)):raise ValueError('Grid changes in a scenario')
                    if not np.array_equal(mask,a['valid'].astype(bool)):raise ValueError('Static obstacle mask changes')
                    if abs(float(a['time_s'])-float(row['time_s']))>1e-6:raise ValueError('Stored time differs')
                    fraction=float(mask.mean())
                    if not .9<fraction<=1:raise ValueError('Invalid fluid fraction')
                    finite_count+=1
                    if len(snapshots)<1 or finite_count==201:
                        snapshots.append(dict(time_s=float(a['time_s']),fluid_fraction=fraction,
                            max_speed_over_inlet=float(np.linalg.norm(a['U'][mask],axis=-1).max()/float(r['u_m_s'])),
                            centered_pressure_rms_over_u2=float(np.std(a['p'][mask],dtype=np.float64)/float(r['u_m_s'])**2)))
            all_frames+=1
        reports.append(dict(scenario_id=sid,split=role,frames=len(rows),all_sha256_checked=True,
            headers_checked=True,numeric_field_frames_checked=finite_count,
            protected_test_values_loaded=False if role.startswith('test') else None,
            diagnostic_endpoints=snapshots,time_range_s=[160,200],full_startup_available=False,
            preexisting_validation_status=summary['status']))
        save(out/'progress.json',dict(completed=len(reports),total=12,frames=all_frames))
        print('SCENARIO_AUDITED',sid,len(rows),finite_count,flush=True)
    with (root/'test/config/dataset_a_v2_scenarios.csv').open(newline='') as f:v2=list(csv.DictReader(f))
    counts={role:sum(r['split']==role for r in v2) for role in ['train','validation','test_ID','test_OOD']}
    assert len(v2)==72 and counts=={'train':46,'validation':8,'test_ID':8,'test_OOD':10}
    assert all(r['test_locked']=='true' for r in v2 if r['split'].startswith('test'))
    long_reviews={}
    for name,rel in [('mixed_pair','results/dataset_a/v2_smooth_window_review/121700/fidelity_review.json'),
                     ('box','results/dataset_a/v2_box_fidelity_review/122974/box_fidelity_review.json')]:
        p=root/rel;a=read(p);hashes[str(p)]=digest(p)
        long_reviews[name]=dict(status=a['status'],passed=a['all_exploratory_screens_passed'],scope_tau=[0,40])
    for item in cfg['missing_long_window_families']:
        p=root/'results/dataset_a/v2_solver_gate'/item['coarse_source']/'solver_summary.txt'
        s=dict(line.split('=',1) for line in p.read_text().splitlines() if '=' in line)
        assert s['status']=='DATASET_A_V2_SOLVER_GATE_OK' and s['tau_end']=='2'
        assert Path(s['case_dir']).is_dir();hashes[str(p)]=digest(p)
    result=dict(status='OWN_CFD_INTEGRITY_AUDITED_MAIN_DATASET_NOT_READY' if not errors else 'OWN_CFD_INTEGRITY_FAILED',
        pilot_scenarios=reports,pilot_frames_sha_verified=all_frames,hash_errors=errors,
        planned_v2_scenarios=72,v2_split_counts=counts,
        v2_production_directory_exists=(root/'dataset_a/v2/production').exists(),
        v2_accepted_scenarios_from_existing_staged_protocol=read(root/'test/config/dataset_a_v2_staged_decision_20261020.json')['production_scenarios_accepted'],
        existing_long_reviews=long_reviews,missing_long_window_families=cfg['missing_long_window_families'],
        ready_for_main_llm_training=False,all_four_llm_families_required=True,
        reasons=['Pilot has12single-cylinder scenarios and only t160-200, not full transient72scenario target',
                 'No accepted complete Dataset A v2 production series',
                 'Only2of6families have long-window configuration fidelity screens; late-time fidelity remains distinct'],
        numeric_test_fields_loaded=False,model_test_metrics_computed=False,
        source_hashes=hashes,auditor_source_sha256=digest(Path(__file__)),elapsed_seconds=time.perf_counter()-started)
    save(out/'audit.json',result)
    print(result['status'],all_frames,flush=True)
    if errors:raise RuntimeError('Native archive hash integrity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)

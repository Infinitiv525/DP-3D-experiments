"""Post-run integrity audit of every saved test row and raw timing, no CFD arrays."""
import math
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from final_measurement_gate_v3 import DEST,MAIN,MANIFEST,RELEASE,authorized,budget_config,atomic
from main_protocol import read,digest,METHODS
from allocation_budget import PhysicalBudget
from measure_final_models_v4 import geometry_rows
from main_statistics import primary_contrasts
from settle_allocation_uid import verify_user_display


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Recomputed numeric summary differs')


def run():
    manifest,lock=authorized();accuracy=read(DEST/'accuracy/summary.json');latency=read(DEST/'latency/summary.json')
    ids={m['id'] for m in lock['models']}
    for report,stage in [(accuracy,'accuracy'),(latency,'latency')]:
        assert report['status']=='FINAL_MEASUREMENT_COMPLETE' and report['stage']==stage
        assert set(report['models'])==ids and report['model_count']==70
        assert report['logical_reference_count']==95 and report['references']==lock['references']
        assert report['authorization_sha256']==digest(DEST/'authorization.json')
        assert report['catalog_sha256']==lock['catalog_sha256']
        assert report['environment']['gpu']=='NVIDIA H200'
    assert accuracy['test_targets_loaded'] is True and latency['test_targets_loaded'] is False
    paired=read(DEST/'accuracy/paired_scores.json');assert paired['methods']==list(METHODS)
    assert len(paired['references'])==95
    total_rows=0;total_timings=0
    worst={m:[] for m in lock['policy']['modes']};passes={m:0 for m in worst}
    expected_counts=Counter({(k,m):1000 for k in [1,2,3] for m in worst})
    for entry in lock['models']:
        folder=DEST/'accuracy'/entry['id'];s=read(folder/'summary.json');rows=read(folder/'test_rows.json')
        assert s==accuracy['models'][entry['id']] and s['checkpoint_sha256']==entry['checkpoint_sha256']
        assert digest(folder/'test_rows.json')==s['test_rows_sha256']
        assert digest(folder/'geometry_rows.json')==s['geometry_rows_sha256']
        assert [r['sample'] for r in rows]==lock['test_ids']
        for r in rows:
            assert not r['velocity_below_epsilon'] and not r['pressure_below_epsilon']
            assert all(math.isfinite(r[k]) and r[k]>=0 for k in ['velocity_error','pressure_error','velocity_norm',
                'pressure_norm','predicted_divergence_rms','reference_divergence_rms'])
        geos=geometry_rows(rows);assert geos==s['geometry_rows']==read(folder/'geometry_rows.json')
        for metric in ['S','velocity_error','pressure_error']:
            close(sum(g[metric] for g in geos)/15,s['metrics'][metric])
        total_rows+=len(rows)
        lf=DEST/'latency'/entry['id'];ls=read(lf/'summary.json');times=read(lf/'measurements.json')
        assert ls==latency['models'][entry['id']] and ls['checkpoint_sha256']==entry['checkpoint_sha256']
        assert ls['wrappers_bit_identical'] is True and ls['latency_input_ids']==lock['latency_input_ids']
        assert len(times)==ls['measured_count']==9000
        assert digest(lf/'measurements.json')==ls['measurements_sha256']
        assert Counter((r['round'],r['mode']) for r in times)==expected_counts
        assert all(r['sample'] in lock['latency_input_ids'] and math.isfinite(r['wall_ms']) and r['wall_ms']>0
            and math.isfinite(r['cuda_event_ms']) and r['cuda_event_ms']>=0 for r in times)
        for mode in worst:
            for round_id in [None,1,2,3]:
                sample=[r for r in times if r['mode']==mode and (round_id is None or r['round']==round_id)]
                summary=ls['per_mode'][mode] if round_id is None else ls['per_round'][mode][str(round_id)]
                assert summary['n']==len(sample)
                for key in ['wall_ms','cuda_event_ms']:
                    values=np.array([r[key] for r in sample],dtype=np.float64)
                    stats=dict(zip(['p50','p95','p99'],np.quantile(values,[.5,.95,.99],method='linear')))
                    stats.update(min=values.min(),max=values.max(),mean=values.mean())
                    for k,v in stats.items():close(float(v),summary[key][k])
                close(sum(r['wall_ms']<=5 for r in sample)/len(sample),summary['wall_fraction_at_or_below']['5'])
            worst[mode].append(max(ls['per_round'][mode][str(k)]['wall_ms']['p99'] for k in [1,2,3]))
            passes[mode]+=int(worst[mode][-1]<=5)
        assert ls['primary_pass']==(worst['gpu_pipeline'][-1]<=5)
        total_timings+=len(times)
    assert total_rows==10500 and total_timings==630000
    assert latency['primary_pass_count']==passes['gpu_pipeline']
    for metric,values in paired['tensors'].items():
        a=np.asarray(values);assert a.shape==(6,15,3,5) and np.isfinite(a).all()
        for ref in lock['references']:
            geos={r['geometry']:r for r in accuracy['models'][ref['training_id']]['geometry_rows']}
            if ref['method']=='manual':continue
            for gi,g in enumerate(paired['geometries']):
                close(a[METHODS.index(ref['method']),gi,ref['repeat'],ref['seed']-410001],geos[g][metric])
    assert primary_contrasts(paired['tensors']['S'])==read(DEST/'accuracy/predeclared_statistics.json')
    with PhysicalBudget(MAIN/'physical_budget',**budget_config(MANIFEST,RELEASE/'freeze.json')) as b:
        budget=b.report();assert all(r['status']=='settled' for r in b.state['allocations'])
        charges={}
        for stage in ['validation_smoke','accuracy','latency']:
            row=next(r for r in b.state['allocations'] if r['label']=='measurement_v3:'+stage)
            capture=MAIN/('allocation_capture_'+row['job_id']);proof=read(capture/'summary.json')
            assert proof['status']=='MAIN_ALLOCATION_CAPTURED_AND_SETTLED'
            query=read(capture/'scheduler_query.json');raw=query['stdout']
            normalized,uid=verify_user_display(raw,'lutoke823')
            assert 'JobState=COMPLETED' in normalized and 'ExitCode=0:0' in normalized
            assert proof['charged_gpu_seconds']==row['charged_gpu_seconds']
            assert row['scheduler_uid_proof']['original_scheduler_sha256']==uid['original_scheduler_sha256']
            charges[stage]=row['charged_gpu_seconds']
    report=dict(status='FLOWBENCH_MAIN_FINAL_MEASUREMENTS_VERIFIED',physical_models=70,logical_references=95,
        test_rows=total_rows,raw_timing_samples=total_timings,test_geometries=15,reynolds_per_geometry=10,
        primary_5ms_pass_models=passes['gpu_pipeline'],per_mode_pass_models=passes,
        worst_round_p99_wall_ms={m:dict(min=min(v),median=float(np.median(v)),max=max(v)) for m,v in worst.items()},
        accuracy_summary_sha256=digest(DEST/'accuracy/summary.json'),latency_summary_sha256=digest(DEST/'latency/summary.json'),
        authorization_sha256=digest(DEST/'authorization.json'),review_source_sha256=digest(Path(__file__)),
        source_freeze_and_all_model_hashes_verified=True,bootstrap_reproduced=True,
        charged_gpu_seconds_by_final_measurement_stage=charges,
        total_main_charged_gpu_seconds=budget['charged_or_reserved_gpu_seconds'],remaining_gpu_seconds=budget['available_gpu_seconds'],
        accounting_warning='Total includes earlier8 conservative full-reservation shared charges; not every historical charge is precise measured runtime.',
        technical_replay_disclosed=True,prior_exposed_test_ids=[10],
        scope='Locked FlowBench steady3D corrected64 FNO experiment only; not own transient/free-surface CFD.',
        latency_scope='H200 batch1 FP32, GPU-resident pipeline p99<=5ms in EACH round. Not whole-application latency.',
        inference_warning='Prespecified approximate bootstrap with only3 search repeats; repeated timings are not independent scientific replicas.')
    atomic(DEST/'final_verification.json',report)
    print(report,flush=True)


if __name__=='__main__':run()

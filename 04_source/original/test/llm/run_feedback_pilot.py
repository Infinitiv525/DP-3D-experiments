"""Sequential bounded pilot: real train/validation -> LLM -> fresh trial."""
import argparse
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import time
import traceback
from pathlib import Path


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def audit(a,cfg):
    a.output.mkdir(parents=True,exist_ok=False);data=a.root/cfg['dataset_relative']
    meta=json.loads((data/'pressure_gauge_audit.json').read_text())
    if meta['status']!='FLOWBENCH_NATIVE_GAUGE64_CACHE_COMPLETE' or meta['source_revision']!=cfg['revision']:
        raise ValueError('Native gauge audit required')
    actual=sha(data/'split.json')
    if actual!=cfg['split_sha256'] or actual!=meta['split_sha256']:raise ValueError('Split differs')
    hashes={}
    for role in ('X','Y'):
        value=sha(data/f'{role}_pilot.npy')
        if value!=meta['channels'][role]['sha256']:raise ValueError('Cache checksum differs')
        hashes[role]=value
    save(a.output/'audit.json',{'status':'FEEDBACK_PILOT_DATA_CHECKSUMS_OK','channels':hashes,
         'split_sha256':actual,'cache_audit_sha256':sha(data/'pressure_gauge_audit.json'),
         'pilot_config_sha256':sha(a.config),'warning':'Structural full-file hash only, not test metric evaluation'})


def pilot(a,cfg):
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();events=[];trials=[];proposals=[]
    summary={'status':'RUNNING','technical_pilot':True,'test_evaluated':False,'events':events,'trials':trials,'proposals':proposals}
    save(a.output/'summary.json',summary)
    llmcfg=a.root/cfg['llm_config_relative'];models=json.loads(llmcfg.read_text())['models']
    train_python=a.root/'envs/training-torch-2.12-cu130/bin/python'
    llm_python=a.root/'envs/llm-feasibility-20261021/bin/python'
    try:
        verified=json.loads((a.audit/'audit.json').read_text())
        if verified['status']!='FEEDBACK_PILOT_DATA_CHECKSUMS_OK' or verified['pilot_config_sha256']!=sha(a.config):
            raise ValueError('Matching CPU integrity audit required')
        data=a.root/cfg['dataset_relative']
        if verified['cache_audit_sha256']!=sha(data/'pressure_gauge_audit.json') or verified['split_sha256']!=sha(data/'split.json'):
            raise ValueError('Cache provenance changed since audit')
        rng=random.Random(cfg['initial_random_seed']);configs=[cfg['manual']]
        for _ in range(2):
            configs.append({'learning_rate':math.exp(rng.uniform(math.log(1e-4),math.log(3e-3))),
                'weight_decay':math.exp(rng.uniform(math.log(1e-6),math.log(1e-2))),
                'pressure_weight':rng.choice([0.25,0.5,1,2,4]),'divergence_weight':rng.choice([0,0.001,0.003,0.01,0.03])})
        save(a.output/'initial_configs.json',configs)
        def stage(label,command,limit):
            remaining=cfg['overall_process_seconds']-(time.perf_counter()-started)
            if remaining<=10:raise TimeoutError('Overall pilot budget exhausted')
            begin=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
            with (a.output/(label+'.log')).open('w') as log:
                try:rc=subprocess.run([str(v) for v in command],stdout=log,stderr=subprocess.STDOUT,timeout=min(limit,remaining)).returncode
                except subprocess.TimeoutExpired:rc=124
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            event={'stage':label,'exit_code':rc,'wall_seconds':time.perf_counter()-begin,
                   'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime}
            events.append(event);save(a.output/'summary.json',summary)
            print('STAGE_COMPLETE',json.dumps(event),flush=True)
            return rc
        def train(label,parameters,seed):
            path=a.output/(label+'_config.json')
            trialcfg={'revision':cfg['revision'],'seed':seed,'pilot_only':True,'training':dict(cfg['training'],**parameters)}
            save(path,trialcfg);dest=a.output/label
            rc=stage(label,[train_python,'-B',a.root/'test/training/train_flowbench_protocol.py',
                     '--config',path,'--data',data,'--output',dest,'--normalization',a.output/'normalization.json',
                     '--max-wall-seconds',str(cfg['trial_timeout_seconds']-15)],cfg['trial_timeout_seconds'])
            result=json.loads((dest/'summary.json').read_text()) if (dest/'summary.json').exists() else {}
            row={'id':label,'configuration':parameters,'seed':seed,'exit_code':rc,'status':result.get('status','NO_SUMMARY')}
            if rc==0 and result['status']=='FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE':
                row.update(validation=result['best_validation'],best_epoch=result['best_epoch'],
                           predicted_divergence=result['predicted_divergence_rms_mean'],
                           training_process_wall_seconds=events[-1]['wall_seconds'],
                           checkpoint_sha256=result['checkpoint_sha256'])
            trials.append(row);save(a.output/'summary.json',summary);return row
        for i,parameters in enumerate(configs):
            row=train(f's{i+1:02}',parameters,cfg['initial_training_seeds'][i])
            if row['exit_code']!=0:raise RuntimeError('Shared start failed; no fabricated feedback allowed')
        history=[]
        for row in trials:
            history.append(dict(row['configuration'],id=row['id'],velocity_error=row['validation']['velocity_error'],
                  pressure_error=row['validation']['pressure_error'],S=row['validation']['S'],
                  divergence=row['predicted_divergence'],training_gpu_hours=row['training_process_wall_seconds']/3600))
        feedback={'history':history,'remaining_training_gpu_hours':cfg['trial_timeout_seconds']/3600,
                  'technical_pilot':True,'epochs_per_trial':cfg['training']['epochs'],
                  'model_description':'Fixed steady FNO width12/layers2/modes4/padding4,64^3,FP32,batch1,AdamW. All700train/150validation,macro average by geometry. No test or field access.'}
        save(a.output/'feedback.json',feedback)
        for i,spec in enumerate(models):
            dest=a.output/('proposal_'+spec['label'])
            rc=stage('proposal_'+spec['label'],[llm_python,'-B',a.root/'test/llm/propose_local_configuration.py',
                     '--config',llmcfg,'--preparation',a.root/cfg['preparation_relative'],
                     '--feedback',a.output/'feedback.json','--index',str(i),'--output',dest],cfg['proposal_timeout_seconds'])
            proposal=json.loads((dest/'summary.json').read_text()) if (dest/'summary.json').exists() else {}
            row={'model':spec['label'],'exit_code':rc,'status':proposal.get('status','NO_SUMMARY')}
            if rc==0 and proposal['status']=='LOCAL_FEEDBACK_PROPOSAL_COMPLETE':
                chosen=proposal['configuration'];params={k:chosen[k] for k in cfg['manual']}
                row['public_justification']={k:chosen[k] for k in ('reason','evidence_ids','expected_effect','risk')}
                trained=train('trial_'+spec['label'],params,cfg['proposal_training_seed'])
                row['linked_trial_id']=trained['id'];row['configuration']=params
            proposals.append(row);save(a.output/'summary.json',summary)
        success=(len(trials)==7 and len(proposals)==4 and all(t['exit_code']==0 for t in trials) and all(p['exit_code']==0 for p in proposals))
        summary['status']='FEEDBACK_PIPELINE_PILOT_COMPLETE' if success else 'FEEDBACK_PIPELINE_PILOT_INCOMPLETE'
    except Exception:
        summary['status']='FEEDBACK_PIPELINE_PILOT_FAILED';summary['error']=traceback.format_exc();print(summary['error'],flush=True)
    finally:
        summary.update(elapsed_seconds=time.perf_counter()-started,
            one_gpu_allocated_process_hours_proxy=(time.perf_counter()-started)/3600,
            pilot_config_sha256=sha(a.config),warning=cfg['warning'])
        save(a.output/'summary.json',summary)
        lines=['TECHNICKÝ PILOT: SKUTOČNÉ VALIDAČNÉ VÝSLEDKY -> LLM -> TRÉNING',
               'Nie je to hlavné porovnanie ani dôkaz výhody LLM. Iba2epochy/pokus; test nedotknutý.',summary['status'],'']
        for row in trials:
            lines.append(f"{row['id']}: {row['status']}; validačné S={row.get('validation',{}).get('S')}; konfigurácia={row['configuration']}")
        lines.extend(['','Zdôvodnenia a priradenie k výsledku:']+[json.dumps(v,ensure_ascii=False) for v in proposals])
        (a.output/'vysledok.txt').write_text('\n'.join(lines)+'\n')
    return summary['status']=='FEEDBACK_PIPELINE_PILOT_COMPLETE'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',type=Path);p.add_argument('--audit-only',action='store_true')
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    if args.audit_only:audit(args,cfg)
    else:raise SystemExit(0 if pilot(args,cfg) else 1)

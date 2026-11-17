"""Audit small result JSONs and split metadata, never open field arrays."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def read(path):return json.loads(path.read_text())


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    d=read(a.run/'summary.json');checks=[]
    cfg=read(a.root/'test/config/flowbench_feedback_pilot_20261021.json')
    split=read(a.root/cfg['dataset_relative']/'split.json')['indices']
    normalizations=[];rows=[]
    checks.append({'check':'pipeline_complete','passed':d['status']=='FEEDBACK_PIPELINE_PILOT_COMPLETE'})
    for trial in d['trials']:
        path=a.run/trial['id'];s=read(path/'summary.json') if (path/'summary.json').exists() else {'status':'NO_SUMMARY'}
        success=s['status']=='FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE'
        valid=success and set(s['training_indices'])==set(split['train']) and set(s['validation_indices'])==set(split['validation']) and not s['test_evaluated']
        checks.append({'check':trial['id']+'_complete_train_val_only','passed':valid})
        if not success:continue
        normalizations.append(s['normalization_sha256'])
        best=min(s['history'],key=lambda e:e['validation']['S'])
        metrics=read(path/'best_validation_rows.json');groups={}
        for m in metrics:groups.setdefault(m['geometry'],[]).append(m)
        eu=sum(sum(x['velocity_error'] for x in v)/len(v) for v in groups.values())/len(groups)
        ep=sum(sum(x['pressure_error'] for x in v)/len(v) for v in groups.values())/len(groups)
        checks.append({'check':trial['id']+'_score_and_selection','passed':
            best['epoch']==s['best_epoch'] and len(groups)==15 and all(len(v)==10 for v in groups.values())
            and abs(s['best_validation']['S']-.5*(eu+ep))<1e-12
            and {m['sample'] for m in metrics}==set(split['validation'])})
        rows.append({'id':trial['id'],'configuration':trial['configuration'],'S':s['best_validation']['S'],
            'velocity_error':eu,'pressure_error':ep,'best_epoch':s['best_epoch'],
            'process_wall_seconds':trial['training_process_wall_seconds'],
            'trainer_elapsed_seconds':s['elapsed_seconds'],'normalization_seconds':s['normalization_seconds'],
            'train_and_validation_seconds':s['training_and_validation_seconds'],
            'peak_torch_allocated_gib':s['peak_torch_allocated_bytes']/2**30,
            'peak_torch_reserved_gib':s['peak_torch_reserved_bytes']/2**30})
    checks.append({'check':'identical_train_only_normalization','passed':len(normalizations)==7 and len(set(normalizations))==1})
    fhash=hashlib.sha256((a.run/'feedback.json').read_bytes()).hexdigest() if (a.run/'feedback.json').exists() else None
    for proposal in d['proposals']:
        proposal_file=a.run/('proposal_'+proposal['model'])/'summary.json'
        p=read(proposal_file) if proposal_file.exists() else {'status':'NO_SUMMARY'}
        trial=next((v for v in d['trials'] if v['id']==proposal.get('linked_trial_id')),None)
        match=(p['status']=='LOCAL_FEEDBACK_PROPOSAL_COMPLETE' and p['feedback_sha256']==fhash and trial is not None
               and all(p['configuration'][k]==trial['configuration'][k] for k in cfg['manual'])
               and trial['seed']==cfg['proposal_training_seed'])
        checks.append({'check':proposal['model']+'_proposal_feedback_trial_link','passed':match})
    all_ok=all(c['passed'] for c in checks)
    slurm={}
    source_job=a.run.name.removeprefix('cycle_')
    if source_job.isdigit():
        state=subprocess.run(['scontrol','show','job',source_job],capture_output=True,text=True,timeout=20)
        if state.returncode==0:
            slurm={k:v for k,v in re.findall(r'(\w+)=(\S+)',state.stdout)
                   if k in ('JobId','JobState','RunTime','StartTime','EndTime','AllocTRES','NumCPUs')}
    (a.output/'slurm_runtime.json').write_text(json.dumps(slurm,indent=2)+'\n')
    result={'status':'FEEDBACK_PIPELINE_REVIEW_PASSED' if all_ok else 'FEEDBACK_PIPELINE_REVIEW_NOT_PASSED',
            'source_run':str(a.run),'checks':checks,'trials':rows,'proposals':d['proposals'],'stage_costs':d['events'],
            'total_driver_wall_seconds':d.get('elapsed_seconds'),'slurm':slurm,'test_evaluated':False,
            'warning':'Technical two-epoch pilot only. Different initial seeds, one proposal/model, no adaptive TPE/random comparison. No LLM efficacy ranking or 20epoch runtime claim.'}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['PREPOJENIE LLM A REÁLNEHO TRÉNINGU - TECHNICKÝ PILOT',result['status'],
           'Tri spoločné tréningy a po jednom návrhu/tréningu štyroch LLM.',
           'Iba dve epochy na pokus; testovacie dáta neboli hodnotené.','']
    for r in rows:
        lines.append(f"{r['id']}: S={r['S']:.6f}, rýchlosť={100*r['velocity_error']:.2f}%, tlak={100*r['pressure_error']:.2f}%, proces={r['process_wall_seconds']:.1f}s")
    elapsed_text='nedostupný' if d.get('elapsed_seconds') is None else f"{d['elapsed_seconds']:.1f}s"
    lines.extend(['',f"Celý proces: {elapsed_text} na jednom GPU (vrátane CPU práce medzi fázami).",
                 'Kontroly:']+[f"{c['check']}: {c['passed']}" for c in checks]+['',result['warning'],
                 'Zdôvodnenia sú uložené pri jednotlivých návrhoch aj v pôvodnom vysledok.txt.'])
    (a.output/'vysledok.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

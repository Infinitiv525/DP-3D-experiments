"""Small-JSON QC and explicit proposed compute caps; no test targets or fields."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main(a):
    base=a.root/'results/llm_search_prepare';out=base/f'review_{a.runtime}_{a.nonllm}'
    out.mkdir(parents=True,exist_ok=False)
    runtime=read(base/f'runtime20_{a.runtime}/summary.json')
    controls=read(base/f'nonllm_{a.nonllm}/summary.json')
    cpu=read(base/f'controls_{a.cpu}/summary.json')
    job=read(base/f'runtime20_{a.runtime}/job_runtime.json')
    checks={'runtime20_complete':runtime['status']=='FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE' and len(runtime['history'])==20,
            'cpu_tests_passed':cpu['status']=='SEARCH_CONTROLS_CPU_TESTS_PASSED' and cpu['tests_run']==9,
            'nonllm_cycle_complete':controls['status']=='NONLLM_CONTROL_PILOT_COMPLETE',
            'test_not_evaluated':not runtime['test_evaluated'] and not controls['test_evaluated']}
    if checks['runtime20_complete']:
        best=min(runtime['history'],key=lambda v:v['validation']['S'])
        checks['runtime_best_checkpoint_selection']=best['epoch']==runtime['best_epoch']
        records=read(base/f'runtime20_{a.runtime}/best_validation_rows.json');grouped={}
        for r in records:grouped.setdefault(r['geometry'],[]).append(r)
        recomputed=sum(sum((r['velocity_error']+r['pressure_error'])/2 for r in rows)/len(rows) for rows in grouped.values())/len(grouped)
        checks['runtime_geometry_macro_score']=len(grouped)==15 and len(records)==150 and abs(recomputed-runtime['best_validation']['S'])<1e-12
    for m in controls['methods']:
        checks[m['method']+'_ledger_and_ids']=(m.get('success',False) and m.get('next_attempt_blocked',False)
            and m['budget_after_run']['attempts_spent']==4 and m['budget_after_run']['overrun_seconds']==0
            and m.get('normalization_sha256')==runtime.get('normalization_sha256')
            and m.get('train_ids_match',False) and m.get('validation_ids_match',False))
        if m['method']=='tpe' and m.get('success'):
            last=m['tpe_study']['trials'][-1]
            checks['tpe_ask_tell_real_score']=(len(m['tpe_study']['trials'])==4 and last['state']=='COMPLETE'
                and last['parameters']==m['configuration'] and abs(last['S']-m['validation']['S'])<1e-12)
    # Direct process time includes interpreter startup; trainer elapsed alone does not.
    measured=job['wall_seconds_lower_bound']
    budget_seconds=math.ceil((12*measured*1.2)/900)*900
    search_methods=6;repetitions=3;final_trainings=95
    logical_search_cap=search_methods*repetitions*budget_seconds/3600
    finals_cap=final_trainings*900/3600
    llm_proposal_cap=4*3*9*210/3600
    qc_allowance=1.0
    global_cap=math.ceil((logical_search_cap+finals_cap+llm_proposal_cap+qc_allowance)/10)*10
    proposal={'status':'PROPOSED_NOT_APPROVED_NOT_FROZEN','measurement_source_job':a.runtime,
        'measured_one_20epoch_process_wall_seconds':measured,'trainer_elapsed_seconds':runtime.get('elapsed_seconds'),
        'budget_rule':'12 trials * one measured process time *1.2, rounded up to next900seconds. One manual config/seed/node only, not a worst-case runtime guarantee.',
        'training_seconds_per_method_per_search_repeat':budget_seconds,'max_attempts_including_three_shared':12,
        'methods_with_search':search_methods,'search_repetitions':repetitions,
        'logical_training_search_gpu_hours_cap':logical_search_cap,
        'final_training_count_before_deduplication':final_trainings,'final_training_process_seconds_cap':900,
        'final_training_gpu_hours_cap':finals_cap,'llm_decisions_process_gpu_hours_cap':llm_proposal_cap,
        'qc_gpu_hours_allowance':qc_allowance,'proposed_global_physical_gpu_hours_cap':global_cap,
        'scope':'Main benchmark only; earlier pilot costs are recorded separately. Primary training budget does not hide LLM overhead; global physical cap includes overhead and failures.',
        'important':'Not an authorization. Need current allocation balance, final data/boundary/latency audit, uniform frozen prompt, sampler resume/ledger integration and user approval before main search. Shared starts are charged to every method logically, executed once physically.'}
    slurm={}
    for label,number in (('runtime',a.runtime),('nonllm',a.nonllm)):
        completed=subprocess.run(['scontrol','show','job',number],capture_output=True,text=True,timeout=20)
        slurm[label]={k:v for k,v in re.findall(r'(\w+)=(\S+)',completed.stdout)
                      if k in ('JobId','JobState','RunTime','NumCPUs','AllocTRES')}
    result={'status':'SEARCH_PREPARATION_QC_PASSED' if all(checks.values()) else 'SEARCH_PREPARATION_QC_NOT_PASSED',
        'checks':checks,'runtime20':{'job':a.runtime,'best_epoch':runtime.get('best_epoch'),
           'validation':runtime.get('best_validation'),'trainer_elapsed_seconds':runtime.get('elapsed_seconds'),
           'training_and_validation_seconds':runtime.get('training_and_validation_seconds'),
           'process_wall_seconds_lower_bound':measured,
           'epoch_training_seconds':[e['train_seconds'] for e in runtime['history']],
           'epoch_validation_seconds':[e['validation_seconds'] for e in runtime['history']]},
        'nonllm_methods':controls['methods'],'slurm':slurm,'budget_proposal':proposal,
        'test_evaluated':False,'warning':'Technical tests and one runtime calibration only, not final model comparison or efficacy evidence.'}
    save(out/'summary.json',result);save(out/'budget_proposal.json',proposal)
    lines=['PRÍPRAVA HLAVNÉHO EXPERIMENTU - MERANIE A KONTROLY',result['status'],
        f"Plný20epoch proces: približne{measured:.0f}s. Skóre nepochádza z testovej sady.",
        'Random/TPE boli overené iba jedným novým2epoch tréningom po rovnakom3pokusovom štarte.',
        'Deväť CPU testov overilo aj failure účtovanie a skutočný časový timeout procesu.',
        '',f"NÁVRH: {budget_seconds/3600:.2f}GPU-h tréningového rozpočtu/metódu/opakovanie, najviac12pokusov vrátane3shared.",
        f"Šesť vyhľadávacích metód x3opakovania: logický tréningový strop{logical_search_cap:.2f}GPU-h.",
        f"Spolu s finálnymi tréningmi, LLM a rezervou navrhujeme globálny fyzický strop{global_cap}GPU-h.",
        'Strop nie je predpoveď spotreby ani schválenie spustenia. Zdieľané štarty sa fyzicky nevykonávajú zbytočne viackrát.',
        'Potrebujeme aktuálny zostatok alokácie a dokončenie auditov/freeze; hlavné porovnanie ešte nebežalo.','',
        'Kontroly:']+[f"{k}: {v}" for k,v in checks.items()]
    (out/'vysledok.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'checks':checks,'budget':proposal,'slurm':slurm},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runtime',required=True);p.add_argument('--nonllm',required=True);p.add_argument('--cpu',required=True)
    args=p.parse_args()
    if not all(v.isdigit() for v in (args.runtime,args.nonllm,args.cpu)):raise ValueError('Numeric job IDs required')
    main(args)

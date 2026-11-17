"""One real random/TPE ask -> bounded fresh training -> TPE tell; pilot only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from search_controls import RandomConfigurator,TPEConfigurator,TrainingBudget,run_bounded_process,validate_parameters


def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    cfg=read(a.root/'test/config/flowbench_feedback_pilot_20261021.json')
    history=read(a.source/'feedback.json')['history']
    keys=tuple(cfg['manual'])
    imported=[{'parameters':{k:r[k] for k in keys},'S':r['S']} for r in history]
    shared_seconds=sum(row['training_gpu_hours']*3600 for row in history)
    sampler_seed=21261101
    samplers={'random':RandomConfigurator(sampler_seed),'tpe':TPEConfigurator(sampler_seed,imported)}
    result={'status':'RUNNING','methods':[],'technical_pilot':True,'test_evaluated':False,
            'shared_feedback_sha256':hashlib.sha256((a.source/'feedback.json').read_bytes()).hexdigest(),
            'sampler_seed':sampler_seed,'sampler_seed_policy':'Search seed20261101 +1000000, separate from common-start RNG',
            'warning':'Two epochs, one adaptive proposal per method. Wiring/ledger check, not efficacy comparison.'}
    save(a.output/'summary.json',result)
    for label,sampler in samplers.items():
        budget=TrainingBudget(shared_seconds+180,max_attempts=4,shared_seconds=shared_seconds,shared_attempts=3)
        available=budget.begin(label,minimum_seconds=2)
        t=time.perf_counter();params=sampler.ask();validate_parameters(params);decision_seconds=time.perf_counter()-t
        trial_cfg={'revision':cfg['revision'],'seed':310003,'pilot_only':True,
                   'purpose':'Two-epoch non-LLM ask/train/tell wiring pilot; no main-search or test evidence.',
                   'training':dict(cfg['training'],**params)}
        save(a.output/(label+'_config.json'),trial_cfg)
        row={'method':label,'configuration':params,'decision_seconds':decision_seconds,'budget_before_run':budget.export()}
        result['methods'].append(row);save(a.output/'summary.json',result)
        out=a.output/label
        outcome=run_bounded_process([str(a.root/'envs/training-torch-2.12-cu130/bin/python'),'-B',
            str(a.root/'test/training/train_flowbench_protocol.py'),'--config',str(a.output/(label+'_config.json')),
            '--data',str(a.root/cfg['dataset_relative']),'--normalization',str(a.source/'normalization.json'),
            '--output',str(out),'--max-wall-seconds',str(max(1,available-15))],available,a.output/(label+'.log'))
        summary=read(out/'summary.json') if (out/'summary.json').exists() else {}
        success=outcome['exit_code']==0 and summary.get('status')=='FLOWBENCH_PROTOCOL_TECHNICAL_TRIAL_COMPLETE'
        score=summary['best_validation']['S'] if success else None
        budget.finish(label,outcome['wall_seconds'],score,'complete' if success else 'timed_out' if outcome['timed_out'] else 'failed')
        if label=='tpe':sampler.tell(score,failed=not success);row['tpe_study']=sampler.export()
        row.update(outcome=outcome,success=success,budget_after_run=budget.export(),next_attempt_blocked=not budget.can_start())
        if success:
            row.update(validation=summary['best_validation'],normalization_sha256=summary['normalization_sha256'],
                       train_ids_match=summary['training_indices']==read(a.source/'s01/summary.json')['training_indices'],
                       validation_ids_match=summary['validation_indices']==read(a.source/'s01/summary.json')['validation_indices'])
        save(a.output/'summary.json',result)
        print('NONLLM_STAGE_COMPLETE',label,success,outcome['wall_seconds'],flush=True)
    same_norm=len({m.get('normalization_sha256') for m in result['methods']})==1
    checks={'both_trainings_completed':all(m['success'] for m in result['methods']),
            'both_attempt_caps_enforced':all(m['next_attempt_blocked'] for m in result['methods']),
            'no_measured_training_budget_overrun':all(m['budget_after_run']['overrun_seconds']==0 for m in result['methods']),
            'shared_normalization':same_norm,'same_train_val_ids':all(m.get('train_ids_match') and m.get('validation_ids_match') for m in result['methods'])}
    result.update(checks=checks,status='NONLLM_CONTROL_PILOT_COMPLETE' if all(checks.values()) else 'NONLLM_CONTROL_PILOT_INCOMPLETE',
                  elapsed_seconds=time.perf_counter()-started)
    save(a.output/'summary.json',result)
    lines=['NÁHODNÉ HĽADANIE A OPTUNA - TECHNICKÁ SKÚŠKA',result['status'],
           'Rovnaké tri spoločné vstupy, jeden nový návrh/metódu a2epoch tréning. Test sa nehodnotil.',
           'Nejde o hlavné porovnanie presnosti.']
    for row in result['methods']:
        lines.append(f"{row['method']}: úspech={row['success']}, S={row.get('validation',{}).get('S')}, čas tréningového procesu={row['outcome']['wall_seconds']:.2f}s, ďalší pokus blokovaný={row['next_attempt_blocked']}")
    (a.output/'vysledok.txt').write_text('\n'.join(lines)+'\n')
    return all(checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(0 if main(p.parse_args()) else 1)

"""Versioned format-contract repair; never overwrite the failed source pilot."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cfg=read(a.root/'test/config/flowbench_feedback_pilot_20261021.json')
    source=read(a.source/'summary.json');model='granite33_8b';events=[]
    result={'status':'RUNNING','source':str(a.source),'prompt_contract_version':2,'test_evaluated':False,'events':events,
            'warning':'Separate format-debugging repair. Original failed proposal retained. Not uniform-prompt efficacy comparison.'}
    save(a.output/'summary.json',result)
    if source['status']!='FEEDBACK_PIPELINE_PILOT_INCOMPLETE' or len(source['trials'])!=6:
        raise ValueError('Expected preserved six-trial source pilot')
    expected=next(v for v in source['proposals'] if v['model']==model)
    if expected['status']!='LOCAL_FEEDBACK_PROPOSAL_INVALID':raise ValueError('Unexpected failure type')
    def stage(label,cmd,limit):
        t=time.perf_counter()
        with (a.output/(label+'.log')).open('w') as log:
            try:rc=subprocess.run([str(v) for v in cmd],stdout=log,stderr=subprocess.STDOUT,timeout=limit).returncode
            except subprocess.TimeoutExpired:rc=124
        events.append({'stage':label,'exit_code':rc,'wall_seconds':time.perf_counter()-t})
        save(a.output/'summary.json',result);return rc
    dest=a.output/'proposal_granite33_8b'
    rc=stage('proposal',[a.root/'envs/llm-feasibility-20261021/bin/python','-B',a.root/'test/llm/propose_local_configuration.py',
        '--config',a.root/cfg['llm_config_relative'],'--preparation',a.root/cfg['preparation_relative'],
        '--feedback',a.source/'feedback.json','--index','3','--output',dest],210)
    if rc:
        result['status']='FEEDBACK_GRANITE_REPAIR_PROPOSAL_FAILED'
    else:
        proposal=read(dest/'summary.json');chosen=proposal['configuration']
        parameters={k:chosen[k] for k in cfg['manual']}
        tcfg={'revision':cfg['revision'],'seed':cfg['proposal_training_seed'],'pilot_only':True,'training':dict(cfg['training'],**parameters)}
        save(a.output/'trial_config.json',tcfg)
        rc=stage('trial',[a.root/'envs/training-torch-2.12-cu130/bin/python','-B',a.root/'test/training/train_flowbench_protocol.py',
             '--config',a.output/'trial_config.json','--data',a.root/cfg['dataset_relative'],'--output',a.output/'trial_granite33_8b',
             '--normalization',a.source/'normalization.json','--max-wall-seconds','225'],240)
        result['status']='FEEDBACK_GRANITE_REPAIR_TRAINING_FAILED' if rc else 'FEEDBACK_GRANITE_REPAIR_COMPLETE'
        if not rc:
            trial=read(a.output/'trial_granite33_8b/summary.json')
            same_feedback=proposal['feedback_sha256']==hashlib.sha256((a.source/'feedback.json').read_bytes()).hexdigest()
            norms={read(a.source/v['id']/'summary.json')['normalization_sha256'] for v in source['trials']}
            checks={'same_feedback':same_feedback,'all_seven_normalizations_identical':len(norms)==1 and trial['normalization_sha256'] in norms,
                    'proposal_parameters_equal_trained':all(trial['config']['training'][k]==chosen[k] for k in parameters),
                    'paired_seed':trial['config']['seed']==cfg['proposal_training_seed'],
                    'validation_geometry_count':trial['best_validation']['geometry_count']==15,
                    'new_contract_recorded':proposal['prompt_contract_version']==2,'test_not_evaluated':not trial['test_evaluated']}
            result.update(checks=checks,configuration=parameters,validation=trial['best_validation'],
                          public_justification={k:chosen[k] for k in ('reason','evidence_ids','expected_effect','risk')})
            if not all(checks.values()):result['status']='FEEDBACK_GRANITE_REPAIR_QC_FAILED'
    result['elapsed_seconds']=time.perf_counter()-start;save(a.output/'summary.json',result)
    lines=['OPRAVNÝ TECHNICKÝ TEST GRANITE',result['status'],
           'Pôvodný návrh pridal total_words; aj jedna oprava ho ponechala a tréning bol správne odmietnutý.',
           'Nová verzia promptu výslovne zakazuje všetky polia okrem ôsmich povolených.',
           'Pôvodné zlyhanie aj náklady zostávajú zachované. Toto nie je rovnaký prompt ako pôvodný pilot ostatných modelov.',
           json.dumps(result,ensure_ascii=False,indent=2)]
    (a.output/'vysledok.txt').write_text('\n'.join(lines)+'\n')
    return result['status']=='FEEDBACK_GRANITE_REPAIR_COMPLETE'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(0 if main(p.parse_args()) else 1)

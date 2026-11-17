"""One real validation-feedback proposal. No dataset paths/arrays passed to LLM."""
import argparse
import hashlib
import json
import os
import resource
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from smoke_local_configurator import SYSTEM,parse_reply,ResourceSampler,FirstToken,save


def main(a):
    started=time.perf_counter();cpu=time.process_time()
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());spec=cfg['models'][a.index]
    verified=json.loads((a.preparation/(spec['label']+'.json')).read_text())
    feedback=json.loads(a.feedback.read_text())
    allowed={'history','remaining_training_gpu_hours','technical_pilot','epochs_per_trial','model_description'}
    if set(feedback)!=allowed or not feedback['technical_pilot']:raise ValueError('Allowlisted pilot feedback required')
    for row in feedback['history']:
        if set(row)!={'id','learning_rate','weight_decay','pressure_weight','divergence_weight',
                     'velocity_error','pressure_error','S','divergence','training_gpu_hours'}:
            raise ValueError('Unexpected feedback fields')
    if spec!=verified['spec'] or verified['status']!='LOCAL_MODEL_VERIFIED':raise ValueError('Pinned model required')
    if os.environ.get('HF_HUB_OFFLINE')!='1':raise ValueError('Offline required')
    sampler=ResourceSampler(.2);sampler.start()
    record={'status':'RUNNING','model':spec,'attempts':[],'test_evaluated':False,
            'feedback_sha256':hashlib.sha256(a.feedback.read_bytes()).hexdigest(),'technical_pilot':True}
    try:
        torch.set_num_threads(4);torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021)
        t=time.perf_counter();path=Path(verified['path']);extra={}
        if spec['loader']=='mistral3_dequantize_fp8':
            from transformers import MistralCommonBackend,Mistral3ForConditionalGeneration,FineGrainedFP8Config
            tokenizer=MistralCommonBackend.from_pretrained(path,local_files_only=True)
            cls=Mistral3ForConditionalGeneration;extra={'quantization_config':FineGrainedFP8Config(dequantize=True)}
        else:
            tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False);cls=AutoModelForCausalLM
        model=cls.from_pretrained(path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
                                 device_map={'':'cuda:0'},attn_implementation='sdpa',**extra).eval()
        torch.cuda.synchronize();record['tokenizer_and_model_load_seconds']=time.perf_counter()-t
        system=SYSTEM.replace('This is a synthetic software test; every history value is fabricated, not real CFD.',
               'History contains real train/validation results from a TWO-EPOCH technical pilot, NOT final research or test results. '
               'All candidates use the same fixed model and train/validation split; no test information is available. '
               'Do not infer final converged accuracy from two epochs.')
        keys=['learning_rate','weight_decay','pressure_weight','divergence_weight','reason','evidence_ids','expected_effect','risk']
        system+=' The JSON object must contain EXACTLY these eight keys and NO additional keys: '+', '.join(keys)+'. Do not include counters, word counts, metadata or explanations outside the object.'
        record['prompt_contract_version']=2
        record['system_prompt_sha256']=hashlib.sha256(system.encode()).hexdigest()
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(feedback)}]
        t0=time.perf_counter()
        for repair in range(2):
            t=time.perf_counter();kwargs={'enable_thinking':False} if spec.get('disable_thinking') else {}
            batch=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                  return_dict=True,return_tensors='pt',**kwargs)
            if batch['input_ids'].shape[-1]>8192:raise ValueError('Context cap; no truncation allowed')
            batch={k:v.cuda() for k,v in batch.items() if torch.is_tensor(v)}
            streamer=FirstToken();gen=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**batch,max_new_tokens=512,max_time=90,do_sample=False,use_cache=True,streamer=streamer)
            torch.cuda.synchronize();raw=tokenizer.decode(output[0,batch['input_ids'].shape[-1]:],skip_special_tokens=True)
            parsed,errors,strict=parse_reply(raw,{r['id'] for r in feedback['history']})
            attempt={'messages':json.loads(json.dumps(messages)),'raw_output':raw,'parsed':parsed,
                     'errors':errors,'strict_json':strict,'repair':repair,'attempt_seconds':time.perf_counter()-t,
                     'generation_seconds':time.perf_counter()-gen,'input_tokens':int(batch['input_ids'].shape[-1]),
                     'output_tokens':int(output.shape[-1]-batch['input_ids'].shape[-1])}
            record['attempts'].append(attempt);save(a.output/f'attempt_{repair}.json',attempt)
            if not errors:record['configuration']=parsed;break
            messages.extend([{'role':'assistant','content':raw},{'role':'user','content':
                             'Invalid response: '+','.join(errors)+'. Return complete corrected JSON only. '
                             'Keep EXACTLY these eight keys: '+', '.join(keys)+'. Remove ALL other keys, including word counts or metadata.'}])
        record['proposal_seconds_after_loading']=time.perf_counter()-t0
        record['status']='LOCAL_FEEDBACK_PROPOSAL_COMPLETE' if 'configuration' in record else 'LOCAL_FEEDBACK_PROPOSAL_INVALID'
    except Exception:
        record['status']='LOCAL_FEEDBACK_PROPOSAL_FAILED';record['error']=traceback.format_exc();print(record['error'],flush=True)
    finally:
        sampler.stop();record['elapsed_seconds']=time.perf_counter()-started
        record['cpu_seconds']=time.process_time()-cpu
        record['linux_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['peak_sampled_rss_bytes']=max(r['rss_with_children_bytes'] for r in sampler.rows)
        gpu=[r['gpu_process_bytes'] for r in sampler.rows if r['gpu_process_bytes'] is not None]
        record['peak_sampled_nvml_gpu_bytes']=max(gpu) if gpu else None
        record['nvml_warning']=sampler.nvml_error;save(a.output/'resources.json',sampler.rows);save(a.output/'summary.json',record)
    return record['status']=='LOCAL_FEEDBACK_PROPOSAL_COMPLETE'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--feedback',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(0 if main(p.parse_args()) else 1)

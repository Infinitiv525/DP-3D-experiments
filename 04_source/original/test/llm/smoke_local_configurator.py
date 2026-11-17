"""Bounded local-only synthetic configurator check, not a CFD experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import threading
import time
import traceback
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


SYSTEM = ('You select four training hyperparameters for a fixed steady 3D FNO. '
          'This is a synthetic software test; every history value is fabricated, not real CFD. '
          'No tools, code, internet or dataset access. Minimize S=0.5*velocity_error+0.5*pressure_error; '
          'divergence is secondary and lower alone does not prove physical truth. '
          'Return ONE JSON object only, no Markdown or reasoning trace. '
          'Required keys: learning_rate, weight_decay, pressure_weight, divergence_weight, '
          'reason, evidence_ids, expected_effect, risk. '
          'learning_rate: real [0.0001,0.003]; weight_decay: real [0.000001,0.01]; '
          'pressure_weight: one of [0.25,0.5,1,2,4]; divergence_weight: one of [0,0.001,0.003,0.01,0.03]. '
          'reason, expected_effect and risk must be nonempty short strings, total under 100 words. '
          'evidence_ids must be a nonempty list of existing history IDs. '
          'Give a brief public justification, not hidden step-by-step reasoning. '
          'No claims that any proposed improvement is already measured.')


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def fixtures(n):
    rows = []
    for i in range(n):
        # Entirely synthetic, deterministic; not loaded from results/data.
        u = round(0.18-0.003*(i % 6)+0.007*(i//6), 4)
        p = round(0.43-0.009*(i % 5)+0.004*(i//5), 4)
        rows.append({'id':f's{i+1:02}', 'learning_rate':[0.001,0.0003,0.002][i % 3],
                     'weight_decay':[0.01,0.001,0.0001][i % 3],
                     'pressure_weight':[1,2,0.5,4][i % 4],
                     'divergence_weight':[0,0.001,0.003,0.01][i % 4],
                     'velocity_error':u,'pressure_error':p,'S':round((u+p)/2,5),
                     'divergence':round(0.28/(1+i % 4),5),'training_gpu_hours':0.07})
    return rows


def parse_reply(text, ids):
    clean = text.strip()
    strict = True
    if clean.startswith('```') and clean.endswith('```'):
        clean = clean.split('\n',1)[1].rsplit('```',1)[0].strip()
        strict = False
    try:
        obj = json.loads(clean)
    except (ValueError, TypeError) as e:
        return None, ['invalid_json'], False
    required = {'learning_rate','weight_decay','pressure_weight','divergence_weight',
                'reason','evidence_ids','expected_effect','risk'}
    errors = []
    if not isinstance(obj,dict) or set(obj)!=required:
        return None, ['exact_required_keys'], strict
    for name, lo, hi in [('learning_rate',1e-4,3e-3),('weight_decay',1e-6,1e-2)]:
        v = obj[name]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            errors.append(name+'_out_of_range')
    for name, choices in [('pressure_weight',[0.25,0.5,1,2,4]),('divergence_weight',[0,0.001,0.003,0.01,0.03])]:
        if isinstance(obj[name],bool) or obj[name] not in choices:
            errors.append(name+'_not_a_category')
    for name in ('reason','expected_effect','risk'):
        if not isinstance(obj[name],str) or not obj[name].strip():
            errors.append(name+'_required')
    ev = obj['evidence_ids']
    if not isinstance(ev,list) or not ev or any(not isinstance(v,str) or v not in ids for v in ev):
        errors.append('invalid_evidence_ids')
    return obj, errors, strict


class ResourceSampler:
    def __init__(self, interval):
        self.interval = interval
        self.stop_event = threading.Event()
        self.rows = []
        self.nvml = None
        self.handles = []
        self.nvml_error = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as e:
            self.nvml_error = str(e)
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try: rss += child.memory_info().rss
            except psutil.Error: pass
        gpu = None
        if self.nvml:
            try:
                values = [p.usedGpuMemory for h in self.handles
                          for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(h)
                          if p.pid==os.getpid() and p.usedGpuMemory < 2**60]
                if values: gpu = sum(values)
            except Exception as e: self.nvml_error = str(e)
        self.rows.append({'wall_unix':time.time(),'rss_with_children_bytes':rss,'gpu_process_bytes':gpu})

    def loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def start(self): self.thread.start()

    def stop(self):
        self.stop_event.set(); self.thread.join(); self.sample()
        if self.nvml: self.nvml.nvmlShutdown()


class FirstToken(BaseStreamer):
    def __init__(self): self.calls=0; self.first=None
    def put(self, value):
        self.calls += 1
        if self.calls==2:
            torch.cuda.synchronize()
            self.first=time.perf_counter()
    def end(self): pass


def run(args):
    cfg=json.loads(args.config.read_text())
    spec=cfg['models'][args.index]
    verified=json.loads((args.preparation/(spec['label']+'.json')).read_text())
    if verified['status']!='LOCAL_MODEL_VERIFIED' or verified['spec']!=spec:
        raise RuntimeError('Verified pinned download required')
    path=Path(verified['path'])
    args.output.mkdir(parents=True,exist_ok=False)
    # Offline switch must be set by Slurm before Transformers is imported.
    if os.environ.get('HF_HUB_OFFLINE')!='1': raise RuntimeError('Offline mode required')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['generation']['seed'])
    torch.cuda.manual_seed_all(cfg['generation']['seed'])
    sampler=ResourceSampler(cfg['resource_sampling_seconds']); sampler.start()
    started=time.perf_counter(); cpu0=time.process_time()
    result={'status':'RUNNING','spec':spec,'synthetic_only':True,'test_evaluated':False,
            'gpu':torch.cuda.get_device_name(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'job_id':os.environ.get('SLURM_JOB_ID'),'array_task':os.environ.get('SLURM_ARRAY_TASK_ID'),
            'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
            'packages':{n:importlib.metadata.version(n) for n in ('transformers','accelerate','huggingface_hub')},
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'download_verification_record_sha256':hashlib.sha256((args.preparation/(spec['label']+'.json')).read_bytes()).hexdigest(),
            'decode':cfg['generation'],'probes':[],'attention':'sdpa','dtype':'bfloat16',
            'warning':'Synthetic validity/cost check only. Not CFD accuracy, not profitability proven.'}
    save(args.output/'summary.json',result)
    try:
        t=time.perf_counter()
        if spec['loader']=='mistral3_dequantize_fp8':
            from transformers import MistralCommonBackend, Mistral3ForConditionalGeneration, FineGrainedFP8Config
            tokenizer=MistralCommonBackend.from_pretrained(path,local_files_only=True)
            cls=Mistral3ForConditionalGeneration
            extra={'quantization_config':FineGrainedFP8Config(dequantize=True)}
            result['weight_origin']='original FP8 checkpoint dequantized to BF16, not pre-quantization BF16'
        else:
            tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
            cls=AutoModelForCausalLM; extra={}
        result['tokenizer_load_seconds']=time.perf_counter()-t
        t=time.perf_counter()
        model=cls.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                                 dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',**extra)
        model.eval(); torch.cuda.synchronize()
        result['model_process_cold_load_seconds']=time.perf_counter()-t
        result['cache_warning']='Process-cold load; filesystem cache not flushed, not physical cold-disk benchmark'
        result['parameter_count']=sum(p.numel() for p in model.parameters())
        result['chat_template_sha256']=hashlib.sha256(str(getattr(tokenizer,'chat_template',None)).encode()).hexdigest()
        save(args.output/'summary.json',result)

        def encode(messages):
            kwargs={'enable_thinking':False} if spec.get('disable_thinking') else {}
            batch=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                                               return_dict=True,return_tensors='pt',**kwargs)
            if batch['input_ids'].shape[-1]>cfg['generation']['max_input_tokens']:
                raise RuntimeError('Input exceeds non-truncated context cap')
            return {k:v.to('cuda') for k,v in batch.items() if torch.is_tensor(v)}

        warm=encode([{'role':'system','content':SYSTEM},{'role':'user','content':'Return an empty JSON object.'}])
        t=time.perf_counter()
        with torch.inference_mode():
            model.generate(**warm,max_new_tokens=16,do_sample=False,use_cache=True)
        torch.cuda.synchronize(); result['warmup_seconds']=time.perf_counter()-t
        for j,n in enumerate(cfg['probe_histories']):
            history=fixtures(n)
            user=json.dumps({'history_is_synthetic':True,'history':history,'remaining_training_gpu_hours':0.21,
                             'task':'Propose one new configuration with concise justification, citing history IDs.'})
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':user}]
            probe={'id':j,'history_count':n,'attempts':[]}
            allstart=time.perf_counter()
            for repair in range(cfg['generation']['max_repairs']+1):
                torch.cuda.reset_peak_memory_stats()
                t=time.perf_counter(); inputs=encode(messages); torch.cuda.synchronize()
                encode_seconds=time.perf_counter()-t
                streamer=FirstToken(); t=time.perf_counter()
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=cfg['generation']['max_new_tokens'],
                          max_time=cfg['generation']['max_time_seconds'],do_sample=False,use_cache=True,streamer=streamer)
                torch.cuda.synchronize(); generation_seconds=time.perf_counter()-t
                num_input=int(inputs['input_ids'].shape[-1])
                tokens=output[0,num_input:]; raw=tokenizer.decode(tokens,skip_special_tokens=True)
                vstart=time.perf_counter(); parsed,errors,strict=parse_reply(raw,{r['id'] for r in history})
                attempt={'repair':repair,'input_tokens':num_input,'output_tokens':int(tokens.numel()),
                         'encode_transfer_seconds':encode_seconds,'generation_seconds':generation_seconds,
                         'first_token_seconds':None if streamer.first is None else streamer.first-t,
                         'output_tokens_per_second':int(tokens.numel())/generation_seconds,
                         'validation_seconds':time.perf_counter()-vstart,'raw_output':raw,
                         'parsed':parsed,'errors':errors,'strict_json':strict,'valid':not errors,
                         'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(),
                         'generated_to_limit':int(tokens.numel())>=cfg['generation']['max_new_tokens'],
                         'messages':json.loads(json.dumps(messages))}
                probe['attempts'].append(attempt)
                save(args.output/f'probe_{j}_attempt_{repair}.json',attempt)
                if not errors: break
                messages.extend([{'role':'assistant','content':raw},{'role':'user','content':
                    'Invalid response: '+','.join(errors)+'. Return only the complete corrected JSON with the required keys.'}])
            probe['valid']=probe['attempts'][-1]['valid']
            probe['time_to_configuration_seconds']=time.perf_counter()-allstart
            result['probes'].append(probe)
            save(args.output/'summary.json',result)
            print('PROBE_COMPLETE',spec['label'],j,probe['valid'],probe['time_to_configuration_seconds'],flush=True)
        result['all_probes_valid']=all(p['valid'] for p in result['probes'])
        result['all_first_attempts_valid']=all(p['attempts'][0]['valid'] for p in result['probes'])
        fields=('learning_rate','weight_decay','pressure_weight','divergence_weight')
        first=result['probes'][0]['attempts'][-1]['parsed']
        repeated=result['probes'][2]['attempts'][-1]['parsed']
        result['repeated_short_same_configuration']=(
            all(first[k]==repeated[k] for k in fields) if first and repeated else None)
        result['status']='LOCAL_LLM_SYNTHETIC_SMOKE_COMPLETE'
    except Exception:
        result['status']='LOCAL_LLM_SYNTHETIC_SMOKE_FAILED'
        result['error']=traceback.format_exc()
        print(result['error'],flush=True)
    finally:
        sampler.stop()
        result['process_wall_seconds']=time.perf_counter()-started
        result['process_cpu_seconds']=time.process_time()-cpu0
        result['one_gpu_process_active_hours_proxy']=result['process_wall_seconds']/3600
        result['sampling_interval_seconds']=sampler.interval
        result['peak_sampled_rss_with_children_bytes']=max(x['rss_with_children_bytes'] for x in sampler.rows)
        values=[x['gpu_process_bytes'] for x in sampler.rows if x['gpu_process_bytes'] is not None]
        result['peak_sampled_nvml_process_gpu_bytes']=max(values) if values else None
        result['nvml_warning']=sampler.nvml_error
        result['linux_ru_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(args.output/'resources.json',sampler.rows)
        save(args.output/'summary.json',result)
    return result['status']=='LOCAL_LLM_SYNTHETIC_SMOKE_COMPLETE'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raise SystemExit(0 if run(args) else 1)

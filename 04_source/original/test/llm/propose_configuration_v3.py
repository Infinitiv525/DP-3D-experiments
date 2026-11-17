"""Common v3 proposer for four verified local families; no array/tool access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from configurator_contract_v3 import VERSION, SYSTEM_SHA256, messages, parse_reply, repair_message, validate_feedback, canonical
from smoke_local_configurator import ResourceSampler, FirstToken, save


def run(a):
    started = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.config.read_text()); spec = config['models'][a.index]; generation = config['generation']
    verified = json.loads((a.preparation / (spec['label'] + '.json')).read_text())
    feedback = json.loads(a.feedback.read_text()); evidence = validate_feedback(feedback)
    if spec != verified['spec'] or verified['status'] != 'LOCAL_MODEL_VERIFIED': raise ValueError('Verified pinned local model required')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or not torch.cuda.is_available(): raise ValueError('Offline CUDA required')
    sampler = ResourceSampler(config['resource_sampling_seconds']); sampler.start()
    record = {'status': 'RUNNING', 'contract_version': VERSION, 'system_prompt_sha256': SYSTEM_SHA256,
              'feedback_sha256': hashlib.sha256(a.feedback.read_bytes()).hexdigest(),
              'initial_messages_sha256': hashlib.sha256(canonical(messages(feedback)).encode()).hexdigest(),
              'model': spec, 'generation': generation, 'phase': feedback['phase'], 'test_evaluated': False,
              'attempts': [], 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda}
    try:
        torch.set_num_threads(4); torch.manual_seed(generation['seed']); torch.cuda.manual_seed_all(generation['seed'])
        t = time.perf_counter(); path = Path(verified['path']); extra = {}
        if spec['loader'] == 'mistral3_dequantize_fp8':
            from transformers import MistralCommonBackend, Mistral3ForConditionalGeneration, FineGrainedFP8Config
            tokenizer = MistralCommonBackend.from_pretrained(path, local_files_only=True)
            cls = Mistral3ForConditionalGeneration; extra = {'quantization_config': FineGrainedFP8Config(dequantize=True)}
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False); cls = AutoModelForCausalLM
        model = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
                                   device_map={'': 'cuda:0'}, attn_implementation='sdpa', **extra).eval()
        torch.cuda.synchronize(); record['load_seconds'] = time.perf_counter() - t
        conversation = messages(feedback)
        for repair in range(generation['max_repairs'] + 1):
            kwargs = {'enable_thinking': False} if spec.get('disable_thinking') else {}
            batch = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors='pt', **kwargs)
            if batch['input_ids'].shape[-1] > generation['max_input_tokens']: raise ValueError('Input exceeds context cap; no truncation')
            batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
            streamer = FirstToken(); t = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**batch, max_new_tokens=generation['max_new_tokens'], max_time=generation['max_time_seconds'],
                                        do_sample=generation['do_sample'], use_cache=True, streamer=streamer)
            torch.cuda.synchronize(); raw = tokenizer.decode(output[0, batch['input_ids'].shape[-1]:], skip_special_tokens=True)
            parsed, errors = parse_reply(raw, evidence)
            attempt = {'repair': repair, 'messages': json.loads(json.dumps(conversation)), 'raw_output': raw,
                       'parsed': parsed, 'errors': errors, 'generation_seconds': time.perf_counter() - t,
                       'first_token_seconds': None if streamer.first is None else streamer.first - t,
                       'input_tokens': int(batch['input_ids'].shape[-1]), 'output_tokens': int(output.shape[-1] - batch['input_ids'].shape[-1])}
            record['attempts'].append(attempt); save(a.output / f'attempt_{repair}.json', attempt)
            if not errors: record['configuration'] = parsed; break
            conversation.extend([{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': repair_message(errors)}])
        record['status'] = 'LOCAL_CONFIGURATOR_V3_COMPLETE' if 'configuration' in record else 'LOCAL_CONFIGURATOR_V3_INVALID'
    except Exception:
        record['status'] = 'LOCAL_CONFIGURATOR_V3_FAILED'; record['error'] = traceback.format_exc(); print(record['error'], flush=True)
    finally:
        sampler.stop(); record.update(elapsed_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_sampled_rss_bytes=max(r['rss_with_children_bytes'] for r in sampler.rows), nvml_warning=sampler.nvml_error)
        gpu_values = [r['gpu_process_bytes'] for r in sampler.rows if r['gpu_process_bytes'] is not None]
        record['peak_sampled_nvml_gpu_bytes'] = max(gpu_values) if gpu_values else None
        save(a.output / 'resources.json', sampler.rows); save(a.output / 'summary.json', record)
    return record['status'] == 'LOCAL_CONFIGURATOR_V3_COMPLETE'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--feedback', type=Path, required=True); p.add_argument('--index', type=int, required=True); p.add_argument('--output', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)

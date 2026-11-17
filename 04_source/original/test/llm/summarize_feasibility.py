"""Summarize bounded synthetic runs without opening any CFD or test arrays."""
import argparse
import csv
import json
import statistics
import re
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--array',required=True)
    p.add_argument('--preparation',required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    cfg=read(args.config)
    base=args.root/'results/llm_feasibility'
    out=args.output if args.output is not None else base/f'review_{args.array}'
    out.mkdir(parents=True,exist_ok=False)
    baseline=read(args.root/'results/flowbench/native_gauge64_baseline_122480/summary.json')
    baseline_seconds=baseline.get('elapsed_seconds')
    rows=[]
    for i,spec in enumerate(cfg['models']):
        directory=base/f'smoke_{args.array}_{i}'
        d=read(directory/'summary.json'); job=read(directory/'job_runtime.json')
        slurm={}
        if d.get('job_id','').isdigit():
            # Master job ID can expand to multiple array records; select exact task.
            lookup=subprocess.run(['scontrol','show','job',f'{args.array}_{i}'],capture_output=True,text=True,timeout=20)
            if lookup.returncode==0:
                slurm={k:v for k,v in re.findall(r'(\w+)=(\S+)',lookup.stdout)
                       if k in ('JobId','JobState','RunTime','StartTime','EndTime','AllocTRES','NumCPUs')}
            else: slurm={'warning':'scontrol runtime unavailable; job_runtime is only a lower bound'}
        (out/f'slurm_{i}.json').write_text(json.dumps(slurm,indent=2)+'\n')
        probes=d.get('probes',[])
        times=[v['time_to_configuration_seconds'] for v in probes]
        mean=statistics.mean(times) if times else None
        load=d.get('model_process_cold_load_seconds')
        cold=(load+d.get('tokenizer_load_seconds',0)+mean) if load is not None and mean is not None else None
        row={'model':spec['label'],'repo':spec['repo'],'revision':spec['revision'],
             'status':d.get('status','MISSING_OR_INTERRUPTED'),
             'all_probes_valid':d.get('all_probes_valid',False),
             'first_attempt_valid_count':sum(v['attempts'][0]['valid'] for v in probes),
             'completed_probes':len(probes),'mean_warm_proposal_seconds':mean,
             'max_warm_proposal_seconds':max(times) if times else None,
             'model_process_cold_load_seconds':load,'cold_plus_warm_estimate_seconds':cold,
             'peak_nvml_gpu_gib':None if d.get('peak_sampled_nvml_process_gpu_bytes') is None else d['peak_sampled_nvml_process_gpu_bytes']/2**30,
             'peak_rss_gib':None if d.get('peak_sampled_rss_with_children_bytes') is None else d['peak_sampled_rss_with_children_bytes']/2**30,
             'one_gpu_process_wall_hours':d.get('one_gpu_process_active_hours_proxy'),
             'allocation_wall_lower_bound_hours':None if not job else job['allocated_wall_seconds_lower_bound']/3600,
             'exit_code':job.get('exit_code'),
             'slurm_runtime':slurm.get('RunTime'),
             'repeated_short_same_configuration':d.get('repeated_short_same_configuration'),
             'warm_proposal_over_old_20epoch_baseline_percent':None if not baseline_seconds or mean is None else 100*mean/baseline_seconds,
             'cold_plus_warm_over_old_20epoch_baseline_percent':None if not baseline_seconds or cold is None else 100*cold/baseline_seconds,
             'error':d.get('error')}
        rows.append(row)
    result={'status':'LOCAL_LLM_FEASIBILITY_REVIEW_COMPLETE','array':args.array,
            'preparation':args.preparation,'models':rows,'all_four_completed':all(r['status']=='LOCAL_LLM_SYNTHETIC_SMOKE_COMPLETE' for r in rows),
            'all_four_valid':all(r['all_probes_valid'] for r in rows),
            'historical_20epoch_baseline_seconds':baseline_seconds,
            'warning':'Synthetic validity/resources only; does NOT establish LLM search accuracy or benefit. Historical baseline timing includes its own setup/validation/timing and is NOT measured new-protocol trial runtime. Cold-plus-warm projection excludes Python import/startup and warm-up, so is not complete cold job cost.',
            'deployment_caveat':'Keeping an LLM GPU allocated while waiting for FNO training consumes allocation too. Report cold-per-proposal and warm retained-service policies separately. Three synthetic probes cannot establish reliability or p99.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['TECHNICKÁ SKÚŠKA LOKÁLNYCH LLM - SYNTHETICKÉ ÚDAJE',
           'Nie je to dôkaz zlepšenia presnosti CFD ani úspechu hlavného experimentu.',
           'Váhy ostávajú na Perune. Skúška nečítala testové dáta ani CFD polia.',
           'Každý model dostal tri otázky vrátane opakovania a najviac jednu opravu.', '']
    def fmt(v): return 'nedostupné' if v is None else f'{v:.2f}'
    for row in rows:
        lines.extend([f"Model: {row['repo']}",f"Stav: {row['status']}",
                      f"Platné hneď na prvý pokus: {row['first_attempt_valid_count']}/{row['completed_probes']}",
                      f"Priemer / maximum času do návrhu po načítaní: {fmt(row['mean_warm_proposal_seconds'])} / {fmt(row['max_warm_proposal_seconds'])} s",
                      f"Načítanie modelu v novom procese: {fmt(row['model_process_cold_load_seconds'])} s",
                      f"Vzorkované maximum GPU pamäte / RAM: {fmt(row['peak_nvml_gpu_gib'])} / {fmt(row['peak_rss_gib'])} GiB",
                      f"Čas GPU procesu: {fmt(row['one_gpu_process_wall_hours'])} h; Slurm runtime: {row['slurm_runtime']}",
                      f"Opakovanie krátkej otázky dalo rovnaké parametre: {row['repeated_short_same_configuration']}",''])
    lines.extend(['',result['warning'],result['deployment_caveat'],
                  'Pri chybe modelu sa nesmie automaticky tvrdiť úspešný štvor-modelový pilot.'])
    (out/'vysledok.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

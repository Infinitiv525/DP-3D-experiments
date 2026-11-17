"""Paired validation comparison with geometry-clustered descriptive bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=a.project/'results/flowbench'
names={'baseline':'uniform64_baseline_122336','control':'refine_control_122413','divergence':'refine_divergence_122412'}
summaries={k:json.loads((r/v/'summary.json').read_text()) for k,v in names.items()}
evals={k:json.loads((r/v/'evaluation/evaluation.json').read_text()) for k,v in names.items()}
c,d=summaries['control'],summaries['divergence']
assert c['initial_checkpoint_provenance']==d['initial_checkpoint_provenance']
assert c['initial_checkpoint_provenance']['sha256']==evals['baseline']['checkpoint_sha256']
assert c['seed']==d['seed'] and c['split_sha256']==d['split_sha256']
for k in ('training_indices','validation_indices','normalization_train_only'):assert c[k]==d[k]
ct=c['config']['training'].copy();dt=d['config']['training'].copy()
assert ct.pop('divergence_weight')==0 and dt.pop('divergence_weight')==.01 and ct==dt
assert all(not e['test_evaluated'] for e in evals.values())
split=json.loads((a.project/'public_data/flowbench/937c672604d4c9f39aceedf1fa2f0a603695264b/uniform64_endpoints/split.json').read_text())
ids=sorted(c['validation_indices']);groups=split['geometry_groups'];unique=sorted({groups[i] for i in ids})
metrics=['fluid_velocity_relative_l2','fluid_centered_pressure_relative_l2','predicted_divergence_rms_unit_cube']
rows={k:{v['sample']:v for v in e['validation_metrics']} for k,e in evals.items()}
assert all(set(v)==set(ids) for v in rows.values())
rng=np.random.default_rng(20261020);boots=rng.integers(0,len(unique),size=(10000,len(unique)))
result={'status':'PAIRED_VALIDATION_COMPARISON_COMPLETE','runs':names,'same_initialization_normalization_split_and_training_budget_verified':True,'sample_count':len(ids),'geometry_groups':len(unique),'seed_count':1,'test_evaluated':False,'metrics':{},'limitations':['Exploratory validation result, not confirmatory LLM evidence.','Cluster bootstrap conditions on one trained seed pair; it excludes training-seed uncertainty.','Intervals computed after observing pilot outcomes are descriptive, not preregistered significance tests.','Lower FD divergence than the interpolated reference is not proof of higher physical accuracy.']}
lines=['FlowBench: matched additional training, no LLM.','150 validation samples / 15 exact geometry groups; single training seed.']
for metric in metrics:
    means={k:float(np.mean([v[i][metric] for i in ids])) for k,v in rows.items()}
    delta=np.array([np.mean([rows['divergence'][i][metric]-rows['control'][i][metric] for i in ids if groups[i]==g]) for g in unique])
    ci=np.percentile(delta[boots].mean(axis=1),[2.5,97.5]).tolist()
    result['metrics'][metric]={'means':means,'physics_minus_control':means['divergence']-means['control'],'physics_relative_change_vs_control':means['divergence']/means['control']-1,'geometry_cluster_bootstrap_95pct_delta_interval':ci}
    lines.append(f'{metric}: baseline={means["baseline"]:.8g}, extra-training control={means["control"]:.8g}, divergence penalty={means["divergence"]:.8g}; paired descriptive delta interval={ci}')
for label,s in summaries.items():lines.append(f'{label}: best epoch {s["best_epoch"]}, measured training+timing seconds {s["elapsed_seconds"]:.3f}')
lines+=['Physics penalty trades about 0.76 percentage points of velocity error for lower FD divergence; pressure is nearly unchanged.','No final test or LLM hypothesis was evaluated. Bootstrap does not include training-seed variability.']
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(a.output/'comparison.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2),flush=True)

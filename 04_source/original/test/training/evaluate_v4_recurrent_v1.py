"""Exploratory inference-only control: compose the frozen V4 at local dt=0.05.

No retraining or new reference input. Distinct from original direct V4 queries;
the checkpoint was not trained on moving prefix windows. Never relabel old results.
"""
import json
import os
from pathlib import Path
import torch
from fpo_common_v5a import ROOT,WORK,save,digest,verify
from fpo_step_v5a import load_cases,grid
from fpo_state_v4 import StatePINN
from train_fpo_step_v5a import evaluate,timing


def run():
    verify();torch.set_num_threads(4)
    for path,checksum in json.loads((WORK/'v4_rolling_freeze_v1.json').read_text())['files'].items():
        if digest(path)!=checksum:raise ValueError('Inference control changed')
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    records=[r for r in json.loads((WORK/'prepared.json').read_text())['records'] if r['proposed_split']=='validation']
    if len(records)!=6:raise ValueError('Expected the same six development validation cases')
    cases=load_cases(records);qgrid=grid()
    checkpoint=ROOT/'results/fpo_pinn_state_20261030_v4/training/139486/prefix_conditioned/best.pt'
    chosen=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not chosen['config']['state_conditioning']:raise ValueError('Known state required')
    model=StatePINN(chosen['config'],[0,0,0],[1,1,1]).cuda();model.load_state_dict(chosen['state']);model.eval()
    out=WORK/'v4_recurrent_control'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    config=json.loads((WORK/'protocol.json').read_text())['training']
    accuracy=evaluate(model,cases,out/'evaluation',qgrid,config)
    speed=timing(model,cases[0],qgrid)
    save(out/'summary.json',dict(status='FROZEN_V4_RECURRENT_CONTROL_EVALUATED',accuracy=accuracy,latency=speed,
          checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),weights_changed=False,
          validation_future_as_input=False,main_test_opened=False,input_frames=[2,3],future_frames=[4,241],
          scope='Same frozen V4 checkpoint repeatedly queried at LOCAL dt0.05 from own updated states; no refitting',
          limitations=['Inference strategy differs from original V4 direct future queries',
                       'V4 was not trained on moving prefix windows',
                       'Exploratory control, not a preregistered main comparison or sole-cause proof']))


if __name__=='__main__':run()

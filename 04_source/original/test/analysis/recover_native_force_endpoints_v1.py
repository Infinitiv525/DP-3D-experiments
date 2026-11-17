"""Offline functionObjects on COPY of saved final fields; no CFD time stepping."""
import argparse
from pathlib import Path
import shutil
import subprocess
import numpy as np
from native_quality_case_v1 import WORK,verify,paths,scenario,item_for,read,save,digest
from evaluate_box_fidelity import histories


def run(job):
    verify();out=WORK/'force_endpoint_recovery'/job;out.mkdir(parents=True,exist_ok=False)
    reports=[]
    for index in [0,1,2,4]:
        row=scenario(item_for(index)['scenario_id']);u=float(row['u_inlet_m_s']);length=float(row['reference_length_m'])
        for branch,array in [('coarse','126204'),('fine','126205')]:
            source,meta=paths(index,branch,array);entry=read(meta/'retained_times.json')[-1]
            if entry['tau']!=40.:raise ValueError('Saved final state must be tau40')
            folder=out/f"{row['scenario_id']}_{branch}";folder.mkdir();case=folder/'case'
            for name in ['0','constant','system',entry['directory']]:shutil.copytree(source/name,case/name)
            before={str(source/entry['directory']/name):digest(source/entry['directory']/name) for name in ['U','p','phi']}
            control=case/'system/controlDict';s=control.read_text().replace('writeInterval 10;','writeInterval 1;')
            control.write_text(s)
            command=['pimpleFoam','-case',str(case),'-postProcess','-time',entry['directory']]
            with (folder/'postprocess.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if result.returncode:raise RuntimeError('Offline force calculation failed: '+str(folder))
            force,discarded=histories(case,'obstacleForces',True)
            if len(force)!=1 or abs(force[0,0]-entry['time_s'])>1e-7:raise ValueError('Exactly one final-time force expected')
            if not np.isfinite(force).all():raise ValueError('Nonfinite offline force')
            old,_=histories(source,'obstacleForces',True);delta=None
            if abs(old[-1,0]-entry['time_s'])<1e-8:
                delta=(force[0,1:]-old[-1,1:]).tolist()
                if not np.allclose(force[0,1:],old[-1,1:],rtol=1e-6,atol=1e-8):raise ValueError('Offline force differs from saved matching-time force')
            if any(digest(Path(p))!=sha for p,sha in before.items()):raise ValueError('Source fields changed')
            record=dict(scenario_id=row['scenario_id'],branch=branch,tau=40.,time_s=float(force[0,0]),
                Cd=float(force[0,1]),Cl=float(force[0,2]),existing_last_tau=float(old[-1,0]*u/length),
                difference_at_existing_matching_endpoint=delta,source_field_sha256=before,
                command=command,postprocess_log_sha256=digest(folder/'postprocess.log'),
                offline_only=True,source_case_modified=False)
            save(folder/'endpoint.json',record);reports.append(record)
            print('OFFLINE_ENDPOINT_COMPLETE',row['scenario_id'],branch,record['Cd'],record['Cl'],flush=True)
    save(out/'summary.json',dict(status='NATIVE_FORCE_ENDPOINTS_RECOVERED_FROM_SAVED_FIELDS',
        endpoints=reports,flow_solver_rerun=False,source_freeze_sha256=digest(WORK/'source_freeze.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);run(p.parse_args().job)

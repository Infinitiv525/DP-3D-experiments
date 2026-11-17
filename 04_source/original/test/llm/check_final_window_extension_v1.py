"""CPU-only review of operational windows; scientific freeze is not rewritten."""
import argparse
import contextlib
from pathlib import Path
import shutil
import unittest

from main_protocol import read, digest, require_authorization
from main_execution import budget_config, verify_runtime_freeze
from allocation_budget import PhysicalBudget
from main_workflow import atomic
from final_window_controller_v1 import locked_plan
import test_final_windows_v1
import verify_main_workflow


def run(a):
    manifest,freeze,approval=require_authorization(a.manifest,a.freeze,a.authorization)
    verify_runtime_freeze(a.root,manifest,freeze)
    a.review.mkdir(parents=True,exist_ok=False)
    passed_core=verify_main_workflow.run(a.review/'core_regression')
    with (a.review/'window_tests.log').open('w') as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_final_windows_v1))
    if not passed_core or not result.wasSuccessful(): raise RuntimeError('Window/core CPU checks failed')
    provenance=dict(freeze_sha256=digest(a.freeze),authorization_sha256=digest(a.authorization))
    plan,jobs,inputs=locked_plan(a,manifest,provenance)
    with PhysicalBudget(a.output/'physical_budget',**budget_config(a.manifest,a.freeze)) as ledger:
        if any(r['status']!='settled' for r in ledger.state['allocations']): raise RuntimeError('Predecessors not closed')
        reserve=sum(j['seconds']+1 for j in jobs)
        if reserve>ledger.report()['available_gpu_seconds']: raise RuntimeError('Whole final phase exceeds budget')
        budget=dict(current_charged_gpu_seconds=ledger.report()['charged_or_reserved_gpu_seconds'],final_reserve_gpu_seconds=reserve,
                    cap=252000,final_unique_trainings=len(jobs),reference_count=95)
    sources=[a.root/'test'/s for s in ('llm/final_window_engine_v1.py','llm/final_window_controller_v1.py',
        'llm/test_final_windows_v1.py','llm/check_final_window_extension_v1.py','scripts/123_final_window_controller.sbatch',
        'scripts/124_check_final_windows.sbatch','llm/capture_main_allocation_v1.py','llm/test_capture_main_allocation_v1.py',
        'scripts/122_capture_main_allocation.sbatch')]
    snapshot=a.review/'sources';snapshot.mkdir()
    for p in sources: shutil.copyfile(p,snapshot/p.name)
    atomic(a.review/'final_plan.json',plan)
    atomic(a.review/'review.json',dict(status='FINAL_WINDOW_EXTENSION_CHECKS_PASSED',original_freeze_sha256=digest(a.freeze),
        source_files={str(p):digest(p) for p in sources},search_summary_files=inputs,
        core_tests=read(a.review/'core_regression/summary.json')['tests'],window_tests=result.testsRun,
        window_failures=len(result.failures),window_errors=len(result.errors),gpu_jobs_submitted=0,test_evaluated=False,
        user_consent_reference='User2026-10-22: statistical significance required; Pokracujeme. Operational batching only; existing70GPU-h approval unchanged.',
        budget=budget))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','manifest','freeze','authorization','output','review'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())

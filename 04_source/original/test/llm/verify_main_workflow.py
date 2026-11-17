"""Produce a machine-readable CPU verification record; no model/data imports."""
import argparse
import contextlib
from pathlib import Path
import time
import unittest

from main_protocol import digest
from main_workflow import atomic
import test_main_protocol
import test_main_workflow
import test_main_statistics


def run(output):
    import test_main_release
    output.mkdir(parents=True, exist_ok=True); results = []
    for label, module in (('protocol', test_main_protocol), ('workflow', test_main_workflow), ('statistics', test_main_statistics), ('release', test_main_release)):
        started = time.perf_counter()
        with (output / (label + '_tests.log')).open('w') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                suite = unittest.defaultTestLoader.loadTestsFromModule(module)
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        results.append(dict(suite=label, tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                            skipped=len(result.skipped), elapsed_seconds=time.perf_counter() - started, passed=result.wasSuccessful()))
    root = Path(__file__).resolve().parents[1]
    sources = {str(p): digest(p) for p in (root / 'llm').glob('*.py')}
    sources.update({str(p): digest(p) for p in (root / 'training').glob('*.py')})
    for p in (root / 'scripts/117_main_worker.sbatch', root / 'scripts/118_check_main_workflow.sbatch',
              root / 'config/flowbench_main_candidate_20261021.json', root / 'config/flowbench_main_release_20261021.json'):
        sources[str(p)] = digest(p)
    passed = all(r['passed'] for r in results)
    atomic(output / 'summary.json', dict(status='MAIN_WORKFLOW_CPU_CHECKS_PASSED' if passed else 'MAIN_WORKFLOW_CPU_CHECKS_FAILED',
        suites=results, tests=sum(r['tests'] for r in results), synthetic_fixtures_only=True,
        research_result=False, gpu_jobs_submitted=0, dataset_arrays_read=False, test_targets_read=False,
        main_authorized=False, actual_gpu_adapter_integration_exercised=False, source_files=sources,
        limitations=['No real20epoch main GPU trial in this verification.', 'Source release/approval remains gated.',
                     'Final-test execution gate remains separate unfinished work; statistics tests are synthetic only.']))
    (output / 'vysledok.txt').write_text(('CPU_WORKFLOW_TESTS_PASSED' if passed else 'CPU_WORKFLOW_TESTS_FAILED') +
        ': synthetic fixtures only; GPU jobs=0; test data not read.\n')
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args().output) else 1)

"""Synthetic CPU fixtures only. Never loads a real input/target/checkpoint."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import final_measurement_gate_v2 as gate

SOURCE = Path(__file__).resolve().parents[1] / 'training/measure_final_models_v2.py'
spec = importlib.util.spec_from_file_location('measurement_fixture', SOURCE)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


class MeasurementTests(unittest.TestCase):
    def test_testonly_accepts_test(self):
        self.assertEqual(gate.TestOnly(['a','b','c'], [1])[1], 'b')

    def test_testonly_forbids_train_validation(self):
        for index in [0,2,-1,slice(None),True,'1']:
            with self.subTest(index=str(index)), self.assertRaises(PermissionError):
                gate.TestOnly(['a','b','c'], [1])[index]

    def test_claim_once(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'claim.json'; gate.exclusive_claim(path, dict(job='1'))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): gate.exclusive_claim(path, dict(job='2'))
            self.assertEqual(path.read_bytes(), before)

    def test_failed_run_does_not_remove_claim(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'claim.json'; gate.exclusive_claim(path, dict(stage='accuracy'))
            self.assertTrue(path.exists())
            # No success marker is needed to prohibit a second attempt.
            with self.assertRaises(FileExistsError): gate.exclusive_claim(path, {})

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source'; path.write_text('one'); locked={str(path):gate.digest(path)}
            gate.verify_files(locked); path.write_text('two')
            with self.assertRaises(ValueError): gate.verify_files(locked)

    def test_empty_source_lock_rejected(self):
        with self.assertRaises(ValueError): gate.verify_files({})

    def test_unauthorized_worker_cannot_claim(self):
        with patch.object(worker,'authorized',side_effect=PermissionError), patch.object(worker,'exclusive_claim') as claim:
            with self.assertRaises(PermissionError): worker.run('accuracy')
            claim.assert_not_called()

    def test_wrong_allocation_cannot_claim(self):
        with patch.object(worker,'authorized',return_value=({},{})), patch.object(worker,'own_allocation',side_effect=PermissionError), patch.object(worker,'exclusive_claim') as claim:
            with self.assertRaises(PermissionError): worker.run('accuracy')
            claim.assert_not_called()

    def test_bad_authorization_checksum(self):
        with patch.object(gate,'original_gate',return_value={}), patch.object(gate,'read',return_value=dict(payload={},sha256='bad')):
            with self.assertRaises(ValueError): gate.authorized()

    def test_gate_precedes_numerical_import(self):
        fn=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        text=ast.unparse(fn)
        self.assertLess(text.index('authorized()'), text.index('import numpy'))
        self.assertLess(text.index('own_allocation(stage)'), text.index('import numpy'))
        self.assertLess(text.index('exclusive_claim('), text.index('import numpy'))

    def test_no_optimizer_no_training(self):
        source=SOURCE.read_text()
        for forbidden in ['.backward(','.train(', 'torch.optim', '.step(']: self.assertNotIn(forbidden,source)
        self.assertIn('torch.inference_mode()',source)

    def test_y_open_is_accuracy_only(self):
        tree=ast.parse(SOURCE.read_text())
        yloads=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="stage == 'accuracy'"
                and 'Y_pilot.npy' in ast.unparse(n)]
        self.assertEqual(len(yloads),1)
        self.assertIn('TestOnly(rawy',ast.unparse(yloads[0]))

    def test_geometry_macro_and_count(self):
        rows=[dict(geometry=str(g),velocity_error=float(g),pressure_error=float(g+2),
                   predicted_divergence_rms=1.,reference_divergence_rms=2.) for g in range(15) for _ in range(10)]
        groups=worker.geometry_rows(rows)
        self.assertEqual(len(groups),15)
        self.assertAlmostEqual(sum(g['S'] for g in groups)/15,8.)
        with self.assertRaises(ValueError): worker.geometry_rows(rows[:-1])
        with self.assertRaises(ValueError): worker.geometry_rows(rows[:-10])

    def test_latency_each_round_not_pooled(self):
        rows={str(k):dict(n=1000,wall_ms=dict(p99=4.9)) for k in (1,2,3)}
        self.assertTrue(worker.latency_pass(rows))
        rows['3']['wall_ms']['p99']=5.001
        self.assertFalse(worker.latency_pass(rows))

    def test_latency_boundary_inclusive(self):
        self.assertTrue(worker.latency_pass({str(k):dict(n=1000,wall_ms=dict(p99=5.)) for k in (1,2,3)}))

    def test_latency_missing_measurements_rejected(self):
        with self.assertRaises(ValueError): worker.latency_pass({'1':dict(n=1000,wall_ms=dict(p99=1.))})
        with self.assertRaises(ValueError): worker.latency_pass({str(k):dict(n=999,wall_ms=dict(p99=1.)) for k in (1,2,3)})

    def test_original_release_stays_closed(self):
        manifest=gate.read(Path(__file__).resolve().parents[1]/'config/flowbench_main_release_20261021.json')
        self.assertIs(manifest['test_execution_released'],False)
        for key in ['rounds','warmup','measurements','target_ms']:
            self.assertEqual(gate.POLICY[key],manifest['latency'][key])
        self.assertEqual(gate.POLICY['physical_models'],70)
        self.assertFalse(gate.POLICY['automatic_retries'])

    def test_shared_ledger_identity_and_hold_attach_release(self):
        source=(Path(__file__).parent/'final_measurement_submit_v2.py').read_text()
        self.assertIn("MAIN / 'physical_budget'",source)
        self.assertIn("'--hold'",source)
        self.assertIn('ledger.attach(label, job_id)',source)
        self.assertLess(source.index("row['capture_job_id']"),source.index("command(['scontrol','release'"))


if __name__=='__main__': unittest.main(verbosity=2)

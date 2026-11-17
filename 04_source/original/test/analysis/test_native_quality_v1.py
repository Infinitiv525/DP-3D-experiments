"""Synthetic fixtures: source preservation, scenario-specific units and headers."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import audit_own_cfd_readiness_v1 as audit
import native_quality_case_v1 as driver


class QualityTests(unittest.TestCase):
    def test_numeric_job_id_required(self):
        for job in ['../bad','12/3','',None]:
            with self.assertRaises(ValueError):driver.paths(0,'coarse',job)

    def test_branch_required(self):
        with self.assertRaises(ValueError):driver.paths(0,'anything','123')

    def test_expected_npz_headers(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'snapshot.npz'
            values=dict(U=np.zeros((128,64,32,3),dtype=np.float32),p=np.zeros((128,64,32),dtype=np.float32),
                valid=np.ones((128,64,32),dtype=bool),x=np.arange(128),y=np.arange(64),z=np.arange(32))
            np.savez_compressed(p,**values);audit.headers(p)
            values['U']=np.zeros((64,32,16,3),dtype=np.float32);np.savez_compressed(p,**values)
            with self.assertRaises(ValueError):audit.headers(p)

    def test_object_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'snapshot.npz';np.savez(p,U=np.array(['x'],dtype=object))
            with self.assertRaises(ValueError):audit.headers(p)

    def test_prepare_uses_scenario_units_preserves_source_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';work.mkdir();(work/'source_freeze.json').write_text('{}')
            folder=root/'test/cases/dataset_a_v2';folder.mkdir(parents=True)
            template=(Path(__file__).resolve().parents[1]/'cases/dataset_a_v2/controlDict_fidelity.template').read_text()
            (folder/'controlDict_fidelity.template').write_text(template)
            sid='a2_sc_001';stem='source_coarse'
            src=root/'dataset_a/v2/gates/solver'/stem
            for name in ['0','constant','system']:
                (src/name).mkdir(parents=True);(src/name/'sentinel').write_text(name)
            cfg=dict(missing_long_window_families=[dict(index=0,scenario_id=sid,coarse_source=stem)],fine_configuration=dict(mesh_factor=1.0))
            row=dict(scenario_id=sid,u_inlet_m_s='0.0175',reference_length_m='0.0085')
            with patch.object(driver,'ROOT',root),patch.object(driver,'WORK',work),patch.object(driver,'verify',return_value=cfg),patch.object(driver,'scenario',return_value=row):
                driver.prepare(0,'coarse','123',src)
                case,out=driver.paths(0,'coarse','123');text=(case/'system/controlDict').read_text()
                self.assertIn('magUInf 0.0175;',text);self.assertIn('lRef 0.0085;',text)
                self.assertIn('Aref 7.225e-05;',text);self.assertNotIn('__',text)
                self.assertEqual((src/'0/sentinel').read_text(),'0')
                self.assertFalse((src/'system/controlDict').exists())
                with self.assertRaises(RuntimeError):driver.prepare(0,'coarse','123',src)
                driver.segment(0,'coarse','123',40)
                self.assertIn('endTime 19.428571428571;', (case/'system/controlDict').read_text())
                self.assertIn('startFrom latestTime;', (case/'system/controlDict').read_text())

    def test_frozen_manifest_keeps_all_llm_families(self):
        cfg=driver.read(Path(__file__).resolve().parents[1]/'config/native_expansion_20261022_v1.json')
        self.assertEqual(cfg['methods'],['manual','random','tpe','qwen3_8b','ministral3_8b','phi4_mini','granite33_8b'])
        self.assertTrue(cfg['user_authorization']['no_cost_estimation_requested'])
        self.assertEqual(len(cfg['missing_long_window_families']),4)
        self.assertEqual(cfg['exploratory_limits']['pressure_mean_aligned_relative_l2_global'],.10)


if __name__=='__main__':unittest.main(verbosity=2)

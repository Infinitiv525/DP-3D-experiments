"""Temporary synthetic release fixtures; never authorizes real experiment files."""
import copy
import platform
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from main_protocol import read, digest, validate_manifest, require_authorization
from main_workflow import atomic
from freeze_main_release import checked_asset, environment
from main_submit import main

ROOT = Path(__file__).resolve().parents[1]


class MainReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
    def test_released_manifest_train_only_and_statistics_closed(self):
        m = read(ROOT / 'config/flowbench_main_release_20261021.json'); validate_manifest(m)
        self.assertTrue(m['execution_ready']); self.assertFalse(m['test_execution_released'])
        self.assertEqual(m['not_yet_released'], []); self.assertIn('Bonferroni', m['statistics']['multiplicity'])
        self.assertEqual(m['required_gpu_name'], 'NVIDIA H200')

    def test_streaming_hash_and_size_guards(self):
        p = self.root / 'synthetic_asset'; atomic(p, dict(synthetic=True))
        receipt = checked_asset(p, digest(p), p.stat().st_size)
        self.assertEqual(receipt['sha256'], digest(p))
        with self.assertRaises(ValueError): checked_asset(p, 'wrong')
        with self.assertRaises(ValueError): checked_asset(p, digest(p), 0)

    def test_environment_probe_uses_actual_interpreter(self):
        result = environment(Path(sys.executable))
        self.assertEqual(result['prefix'], sys.prefix); self.assertEqual(result['python'], platform.python_version())
        self.assertIn('numpy', result['packages'])

    def asset_fixture(self, packages=None):
        manifest = read(ROOT / 'config/flowbench_main_release_20261021.json')
        m, f, a = [self.root / p for p in ('manifest.json', 'freeze.json', 'temporary_synthetic_approval.json')]
        p = self.root / 'synthetic_asset'; atomic(p, dict(synthetic=True)); row = checked_asset(p, digest(p))
        receipt = self.root / 'asset_receipt.json'; atomic(receipt, dict(status='ALL_MODEL_AND_CACHE_FILES_REHASHED', files=[row]))
        atomic(m, manifest)
        atomic(f, dict(status='MAIN_SOURCE_FREEZE_PASSED', manifest_file_sha256=digest(m),
            source_files={str(receipt): digest(receipt)}, asset_receipt_path=str(receipt),
            environments={sys.prefix: dict(python=platform.python_version(), packages=packages or {})}))
        atomic(a, dict(approved=True, manifest_file_sha256=digest(m), freeze_file_sha256=digest(f), gpu_hours_cap=70,
            user_consent_reference='SYNTHETIC TEMPORARY FIXTURE; not real consent or authorization'))
        return m, f, a, p

    def test_asset_change_blocks_otherwise_matching_gate(self):
        m, f, a, p = self.asset_fixture(); require_authorization(m, f, a)
        atomic(p, dict(synthetic='CHANGED'))
        with self.assertRaises(ValueError): require_authorization(m, f, a)

    def test_package_version_change_blocks_gate(self):
        m, f, a, p = self.asset_fixture(dict(numpy='NOT_A_REAL_INSTALLED_VERSION'))
        with self.assertRaises(ValueError): require_authorization(m, f, a)

    def test_shared_batch_releases_only_predeclared_first_job(self):
        manifest = read(ROOT / 'config/flowbench_main_release_20261021.json')
        a = SimpleNamespace(root=self.root, manifest=self.root / 'm', freeze=self.root / 'f', authorization=self.root / 'a',
                            output=self.root / 'output', action='submit', stage='shared')
        for p in (a.manifest, a.freeze, a.authorization): atomic(p, dict(synthetic_not_real_approval=True))
        jobs = [dict(label=f'r0_s0{i}', stage='shared', seconds=900) for i in range(3)]
        submitted = []; released = []
        def command(argv):
            if argv[0] == 'sbatch': submitted.append(argv); return str(900 + len(submitted))
            if argv[:2] == ['scontrol', 'release']: released.append(argv[2]); return ''
            raise AssertionError(argv)
        with patch('main_submit.require_authorization', return_value=(manifest, {}, {})), \
                patch('main_submit.verify_runtime_freeze'), patch('main_submit.stage_jobs', return_value=jobs), \
                patch('main_submit.command', side_effect=command):
            main(a)
        self.assertEqual(len(submitted), 3); self.assertEqual(released, ['901'])
        self.assertTrue(read(a.output / 'shared_submitted.json')['canary_only'])


if __name__ == '__main__': unittest.main(verbosity=2)

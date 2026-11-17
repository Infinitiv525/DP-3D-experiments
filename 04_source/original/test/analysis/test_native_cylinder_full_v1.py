import tempfile
from pathlib import Path
import unittest
from native_cylinder_full_v1 import schedule,control,locate_times,U,L


class FullPilotTests(unittest.TestCase):
    def test461_unique_states_and_boundaries(self):
        s=schedule();self.assertEqual(len(s),461);self.assertEqual(len(set(s)),461)
        self.assertEqual(s[0],0);self.assertEqual(s[-1],400)
        self.assertEqual(s.count(80),1);self.assertEqual(s.count(320),1)

    def test_stages_cover_remaining360_exactly(self):
        s=schedule();windows=[(40,80),(80,160),(160,240),(240,320),(320,360),(360,400)]
        combined=[t for a,b in windows for t in s if a<t<=b]
        self.assertEqual(len(combined),360);self.assertEqual(s[:101]+combined,s)

    def test_dense_sparse_dense(self):
        s=schedule();self.assertIn(79.6,s);self.assertNotIn(80.4,s)
        self.assertIn(84,s);self.assertIn(316,s);self.assertIn(320.4,s)

    def text(self):
        return '\n'.join(k+' old;' for k in ['startFrom','stopAt','endTime','deltaT','maxCo','maxDeltaT','writeControl','writeInterval','purgeWrite'])+'\nfunctions {\n obstacleForces {\n  writeControl timeStep;\n  writeInterval 10;\n }\n}\n'

    def test_control_only_changes_top_level(self):
        s=control(self.text(),160,4)
        self.assertIn('maxCo 0.1;',s);self.assertIn('  writeInterval 10;',s)
        self.assertIn('  writeControl timeStep;',s)

    def test_duplicate_top_level_rejected(self):
        with self.assertRaises(ValueError):control(self.text()+'maxCo 0.3;\n',80,.4)

    def test_missing_top_level_rejected(self):
        with self.assertRaises(ValueError):control(self.text().replace('maxCo old;',''),80,.4)

    def test_parallel_checkpoint_checks_all_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d);name=f'{40*L/U:.14g}'
            for i in range(2):
                p=case/f'processor{i}'/name;p.mkdir(parents=True)
                (p/'U').touch();(p/'p').touch()
            self.assertEqual(locate_times(case,[40],2)[0]['directory'],name)
            (case/'processor1'/name/'p').unlink()
            with self.assertRaises(ValueError):locate_times(case,[40],2)


if __name__=='__main__':unittest.main(verbosity=2)

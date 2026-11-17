import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from native_review_serialization_v1 import save


class SerializationTests(unittest.TestCase):
    def test_nested_numpy_flags_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json'
            save(p,dict(flux=dict(passed=np.bool_(True),error=np.float64(.01)),n=np.int64(101),nested=[np.bool_(False)]))
            r=json.loads(p.read_text());self.assertIs(r['flux']['passed'],True);self.assertIs(r['nested'][0],False)
            self.assertEqual(r['n'],101);self.assertEqual(r['flux']['error'],.01)

    def test_nonfinite_does_not_create_partial_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json'
            with self.assertRaises(ValueError):save(p,dict(metric=np.float64('nan')))
            self.assertFalse(p.exists())


if __name__=='__main__':unittest.main(verbosity=2)

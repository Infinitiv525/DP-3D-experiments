import json
import unittest
import numpy as np
from native_force_endpoint_support_v2 import append_endpoint


class EndpointTests(unittest.TestCase):
    def setUp(self):
        t=np.arange(1,4000)/100
        self.h=np.column_stack([t,2*t+3,-t])
        self.e=dict(time_s=20.,tau=40.,Cd=83.,Cl=-40.,offline_only=True)

    def test_real_endpoint_added_no_source_mutation(self):
        old=self.h.copy();h,m=append_endpoint(self.h,self.e,2.,1.)
        self.assertEqual(h[-1].tolist(),[40.,83.,-40.]);self.assertTrue(m['appended_actual_endpoint'])
        self.assertTrue(np.array_equal(self.h,old))
        self.assertIsInstance(json.loads(json.dumps(m,allow_nan=False))['appended_actual_endpoint'],bool)
        t=np.arange(24,40.01,.4);self.assertTrue(np.allclose(np.interp(t,h[:,0],h[:,1]),2*t+3))

    def test_existing_endpoint_not_duplicated(self):
        h=np.vstack([self.h,[40.,83.,-40.]])
        result,m=append_endpoint(h,self.e,2.,1.);self.assertEqual(len(result),len(h));self.assertFalse(m['appended_actual_endpoint'])

    def test_mismatching_existing_endpoint_rejected(self):
        h=np.vstack([self.h,[40.,0.,-40.]])
        with self.assertRaises(ValueError):append_endpoint(h,self.e,2.,1.)

    def test_large_gap_rejected(self):
        with self.assertRaises(ValueError):append_endpoint(self.h[:-30],self.e,2.,1.)

    def test_wrong_physical_time_rejected(self):
        with self.assertRaises(ValueError):append_endpoint(self.h,dict(self.e,time_s=40.),2.,1.)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):append_endpoint(self.h,dict(self.e,Cd=float('nan')),2.,1.)


if __name__=='__main__':unittest.main(verbosity=2)

import unittest
import numpy as np
from audit_flowbench_dataset import boundary_stats,div_rms,packed
from flowbench_resampling import resample_uniform
from prepare_flowbench_native_gauge import native_center


class AuditTests(unittest.TestCase):
    def test_lid_axis_and_edge_exclusion(self):
        v=np.zeros((3,6,6,6),dtype=np.float32);v[0,:,-1,:]=1
        faces=boundary_stats(v,np.ones((6,6,6),bool))
        self.assertEqual(max(r['max_absolute_error'] for r in faces),0)
        v[1,2,-1,2]=.1
        self.assertGreater(max(r['max_absolute_error'] for r in boundary_stats(v,np.ones((6,6,6),bool))),.09)
    def test_divergence_and_mask(self):
        grid=np.linspace(0,1,6,dtype=np.float32)
        v=np.zeros((3,6,6,6),dtype=np.float32);v[0]=grid[:,None,None]
        self.assertAlmostEqual(div_rms(v,np.ones((6,6,6),bool)),1,places=6)
    def test_gauge_resampling_and_endpoints(self):
        rng=np.random.default_rng(1);y=rng.normal(size=(4,6,6,6)).astype(np.float32)
        fluid=np.ones((6,6,6),bool);fluid[2:4,2:4,2:4]=False
        a=resample_uniform(native_center(y,fluid),4)
        b=resample_uniform(native_center(y,fluid,37),4)
        self.assertTrue(np.allclose(a,b,atol=1e-6))
        self.assertTrue(np.array_equal(a[:3],resample_uniform(y,4)[:3]))
        self.assertTrue(np.array_equal(a[:,0,0,0],native_center(y,fluid)[:,0,0,0]))
    def test_packed_mask(self):
        m=np.arange(24).reshape(2,3,4)%3==0
        self.assertTrue(np.array_equal(np.unpackbits(np.frombuffer(packed(m),dtype=np.uint8))[:m.size].reshape(m.shape).astype(bool),m))


if __name__=='__main__':unittest.main(verbosity=2)

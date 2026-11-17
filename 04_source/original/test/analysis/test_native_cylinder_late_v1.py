import unittest
import numpy as np
from review_native_cylinder_late_v1 import field_sums,metrics


class LateTests(unittest.TestCase):
    def setUp(self):
        self.p=np.arange(24,dtype=float).reshape(4,3,2)
        self.f={'U':np.ones((4,3,2,3)),'p':self.p}
        self.mask=np.ones((4,3,2),dtype=bool)

    def test_constant_pressure_shift_removed(self):
        c=dict(U=self.f['U'].copy(),p=self.p+99)
        v=field_sums(c,self.f,self.mask,self.mask);self.assertEqual(v[2],0)

    def test_known_velocity_difference(self):
        c=dict(U=self.f['U']*1.1,p=self.p)
        v=field_sums(c,self.f,self.mask,self.mask);m=metrics(v,[2.04,.01],[2.,0.])
        self.assertAlmostEqual(m['velocity_relative_l2_global'],.1)
        self.assertAlmostEqual(m['mean_drag_coefficient_relative_difference'],.02)
        self.assertAlmostEqual(m['mean_lift_coefficient_absolute_difference'],.01)

    def test_empty_wake_rejected(self):
        with self.assertRaises(ValueError):field_sums(self.f,self.f,self.mask,~self.mask)

    def test_nonfinite_rejected(self):
        c=dict(U=self.f['U']*float('nan'),p=self.p)
        with self.assertRaises(ValueError):field_sums(c,self.f,self.mask,self.mask)


if __name__=='__main__':unittest.main(verbosity=2)

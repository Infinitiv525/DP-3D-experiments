"""Synthetic CPU arrays only; no model checkpoint, split or dataset access."""
import unittest
import torch
from measure_final_models_v4 import sample_metrics


class NumericalMeasurementTests(unittest.TestCase):
    def setUp(self):
        grid=torch.stack(torch.meshgrid(*[torch.linspace(0,1,9)]*3,indexing='ij'))
        self.features=torch.ones(1,2,9,9,9)
        self.target=torch.ones(1,4,9,9,9)
        self.target[0,0]=grid[0];self.target[0,1]=1;self.target[0,2]=0
        self.target[0,3]=grid[1]

    def test_same_field_zero_error_reference_nonzero_divergence(self):
        row=sample_metrics(self.target,self.target,self.features)
        self.assertEqual(row['velocity_error'],0)
        self.assertEqual(row['pressure_error'],0)
        self.assertAlmostEqual(row['predicted_divergence_rms'],1)
        self.assertAlmostEqual(row['reference_divergence_rms'],1)

    def test_correct_diagnostic_channel_mask(self):
        prediction=self.target.clone();prediction[:,0]*=2
        row=sample_metrics(prediction,self.target,self.features)
        self.assertGreater(row['velocity_error'],0)
        self.assertEqual(row['pressure_error'],0)
        self.assertAlmostEqual(row['predicted_divergence_rms'],2)
        self.assertAlmostEqual(row['reference_divergence_rms'],1)

    def test_pressure_gauge_shift_invariant(self):
        prediction=self.target.clone();prediction[:,3]+=17
        self.assertEqual(sample_metrics(prediction,self.target,self.features)['pressure_error'],0)

    def test_solids_excluded_and_reference_unmutated(self):
        features=self.features.clone();features[:,:,0]=-1
        prediction=self.target.clone();prediction[:,:,0]=10000
        before=self.target.clone()
        row=sample_metrics(prediction,self.target,features)
        self.assertEqual(row['velocity_error'],0)
        self.assertEqual(row['pressure_error'],0)
        self.assertTrue(torch.equal(self.target,before))

    def test_nonfinite_not_silently_ignored(self):
        prediction=self.target.clone();prediction[:,:,4,4,4]=float('nan')
        with self.assertRaises(ValueError):sample_metrics(prediction,self.target,self.features)

    def test_bad_input_shape_rejected(self):
        with self.assertRaises(ValueError):sample_metrics(self.target,self.target,self.features[:,0])

    def test_degenerate_pressure_rejected(self):
        target=self.target.clone();target[:,3]=0
        with self.assertRaises(ValueError):sample_metrics(target,target,self.features)

    def test_empty_fluid_rejected(self):
        with self.assertRaises(ValueError):sample_metrics(self.target,self.target,-self.features)


if __name__=='__main__':unittest.main(verbosity=2)

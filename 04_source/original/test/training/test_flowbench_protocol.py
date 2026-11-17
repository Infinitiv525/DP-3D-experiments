"""Small CPU unit tests; never open research data."""
import unittest
import torch
from flowbench_protocol_metrics import weighted_data_loss,fluid_errors,aggregate_geometry,validate_split,TrainValidationOnly,validate_hyperparameters


class ProtocolTests(unittest.TestCase):
    def test_equal_weights_reproduce_legacy_loss(self):
        torch.manual_seed(1);p=torch.randn(1,4,3,3,3,requires_grad=True);t=torch.randn_like(p)
        value=weighted_data_loss(p,t,1)
        self.assertTrue(torch.allclose(value,(p-t).square().mean()))
        value.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_pressure_weight_changes_loss(self):
        p=torch.zeros(1,4,3,3,3);p[:,3]=2;t=torch.zeros_like(p)
        self.assertAlmostEqual(float(weighted_data_loss(p,t,1)),1)
        self.assertAlmostEqual(float(weighted_data_loss(p,t,3)),2)

    def test_gauge_and_solid_invariance(self):
        torch.manual_seed(2);t=torch.randn(4,3,3,3,dtype=torch.float64);p=t.clone()
        mask=torch.ones(3,3,3,dtype=torch.bool);mask[0]=False
        p[3]+=37;p[:,~mask]=999
        metrics=fluid_errors(p,t,mask)
        self.assertLess(metrics['velocity_error'],1e-12);self.assertLess(metrics['pressure_error'],1e-12)

    def test_geometry_macro_not_sample_macro(self):
        rows=[{'geometry':'a','velocity_error':0.,'pressure_error':0.}]
        rows += [{'geometry':'b','velocity_error':1.,'pressure_error':1.} for _ in range(3)]
        self.assertEqual(aggregate_geometry(rows)['S'],.5)

    def test_split_and_test_access_guard(self):
        split={'indices':{'train':[0,1],'validation':[2,3],'test':[4,5]},'geometry_groups':['a','a','b','b','c','c']}
        validate_split(split);arr=TrainValidationOnly(list(range(6)),split)
        self.assertEqual(arr[2],2)
        with self.assertRaises(PermissionError):_ = arr[4]
        split['geometry_groups'][4]='a'
        with self.assertRaises(ValueError):validate_split(split)

    def test_invalid_metrics(self):
        with self.assertRaises(ValueError):aggregate_geometry([])
        with self.assertRaises(ValueError):weighted_data_loss(torch.zeros(1,4,2,2,2),torch.zeros(1,4,2,2,2),-1)

    def test_hyperparameter_guard(self):
        tc={'learning_rate':.001,'weight_decay':.01,'pressure_weight':1,'divergence_weight':0}
        validate_hyperparameters(tc)
        tc['learning_rate']=True
        with self.assertRaises(ValueError):validate_hyperparameters(tc)


if __name__=='__main__':unittest.main(verbosity=2)

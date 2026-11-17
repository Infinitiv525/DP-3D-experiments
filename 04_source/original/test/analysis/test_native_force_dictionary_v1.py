import unittest
from native_force_dictionary_v1 import force_every_step


class DictionaryTests(unittest.TestCase):
    def setUp(self):
        self.text='writeInterval .2;\nfunctions {\n vorticity { writeControl timeStep; writeInterval 10; }\n obstacleForces\n { type forceCoeffs; writeControl timeStep; writeInterval 10; nested { a 1; } }\n}\n'

    def test_only_named_force_block_changes(self):
        result=force_every_step(self.text)
        self.assertEqual(result,self.text.replace('type forceCoeffs; writeControl timeStep; writeInterval 10;','type forceCoeffs; writeControl timeStep; writeInterval 1;'))
        self.assertIn('vorticity { writeControl timeStep; writeInterval 10;',result)

    def test_missing_block_rejected(self):
        with self.assertRaises(ValueError):force_every_step(self.text.replace('obstacleForces','other'))

    def test_wrong_cadence_rejected(self):
        with self.assertRaises(ValueError):force_every_step(self.text.replace('type forceCoeffs; writeControl timeStep; writeInterval 10;','type forceCoeffs; writeControl timeStep; writeInterval 20;'))

    def test_duplicate_block_rejected(self):
        with self.assertRaises(ValueError):force_every_step(self.text+self.text)


if __name__=='__main__':unittest.main(verbosity=2)

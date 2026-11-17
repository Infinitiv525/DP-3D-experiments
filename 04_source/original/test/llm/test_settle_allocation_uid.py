import os
import unittest
from settle_allocation_uid import verify_user_display


class UIDTests(unittest.TestCase):
    def test_named_uid_verified(self):
        raw = f'JobId=123 UserId=unit({os.getuid()}) Account=fixture'
        normalized, proof = verify_user_display(raw, 'unit')
        self.assertEqual(normalized, raw)
        self.assertEqual(proof['original_scheduler_text'], raw)

    def test_numeric_display_verified_and_proof_preserved(self):
        raw = f'JobId=123 UserId={os.getuid()}({os.getuid()}) Account=fixture'
        normalized, proof = verify_user_display(raw, 'unit')
        self.assertIn(f'UserId=unit({os.getuid()})', normalized)
        self.assertEqual(proof['original_scheduler_text'], raw)

    def test_wrong_uid_or_name_refused(self):
        for user in (f'unit({os.getuid()+1})', f'other({os.getuid()})', f'unit', 'unit(abc)'):
            with self.assertRaises(ValueError): verify_user_display('JobId=123 UserId=' + user, 'unit')

    def test_multiple_user_fields_refused(self):
        raw = f'UserId=unit({os.getuid()}) UserId=unit({os.getuid()})'
        with self.assertRaises(ValueError): verify_user_display(raw, 'unit')


if __name__ == '__main__': unittest.main(verbosity=2)

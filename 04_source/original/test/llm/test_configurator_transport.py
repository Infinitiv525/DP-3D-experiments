import json
import unittest
from configurator_transport_v1 import parse_reply
from test_contract_v3 import reply


class TransportTests(unittest.TestCase):
    def test_plain_payload_unchanged(self):
        result, errors, info = parse_reply(json.dumps(reply()), {'s01'})
        self.assertEqual(result, reply()); self.assertFalse(errors)
        self.assertTrue(info['raw_strict_json_valid']); self.assertFalse(info['single_fence_removed'])

    def test_one_complete_json_fence(self):
        for language in ('json', 'JSON', ''):
            result, errors, info = parse_reply('```' + language + '\n' + json.dumps(reply()) + '\n```', {'s01'})
            self.assertEqual(result, reply()); self.assertFalse(errors)
            self.assertTrue(info['single_fence_removed']); self.assertFalse(info['raw_strict_json_valid'])

    def test_no_json_extraction_from_prose_or_multiple_fences(self):
        raw = '```json\n' + json.dumps(reply()) + '\n```'
        for bad in ('Here is JSON:\n' + raw, raw + '\nExplanation', raw + '\n' + raw,
                    '```python\n' + json.dumps(reply()) + '\n```', raw[:-3], json.dumps(reply()) + json.dumps(reply())):
            self.assertTrue(parse_reply(bad, {'s01'})[1])

    def test_semantic_errors_not_repaired(self):
        for patch in ({'learning_rate': True}, {'counter': 1}, {'evidence_ids': ['failed']}, {'reason': 'x ' * 101}):
            raw = '```json\n' + json.dumps(dict(reply(), **patch)) + '\n```'
            self.assertTrue(parse_reply(raw, {'s01'})[1])

    def test_nan_and_duplicate_keys_remain_invalid(self):
        raw = json.dumps(reply())
        for bad in (raw.replace('0.001', 'NaN'), raw[:-1] + ',"risk":"duplicate"}'):
            self.assertTrue(parse_reply('```json\n' + bad + '\n```', {'s01'})[1])


if __name__ == '__main__': unittest.main(verbosity=2)

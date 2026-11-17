import copy
import json
import unittest
from configurator_contract_v3 import messages, parse_reply, validate_feedback, SYSTEM_SHA256
from sequential_feedback import build_feedback


def parameters(): return dict(learning_rate=.001, weight_decay=.01, pressure_weight=1, divergence_weight=0)
def fixture():
    return {'contract_version': 3, 'phase': 'technical_pilot', 'epochs_per_trial': 2,
            'remaining_training_seconds': 300., 'remaining_attempts': 2,
            'history': [dict(id='s01', parameters=parameters(), status='complete', S=.3,
                velocity_error=.2, pressure_error=.4, divergence=.1, training_seconds=50, decision_seconds=0)]}
def reply(): return dict(parameters(), reason='Test public reason.', evidence_ids=['s01'], expected_effect='May improve.', risk='May not improve.')


class ContractTests(unittest.TestCase):
    def test_phase_and_epoch_not_hardcoded(self):
        first = messages(fixture()); second_fixture = fixture(); second_fixture.update(phase='main_search', epochs_per_trial=20)
        second = messages(second_fixture)
        self.assertEqual(first[0], second[0]); self.assertEqual(len(SYSTEM_SHA256), 64)
        self.assertNotIn('TWO-EPOCH', first[0]['content'])

    def test_valid_reply(self): self.assertEqual(parse_reply(json.dumps(reply()), {'s01'})[1], [])

    def test_strict_json_duplicate_keys_and_nan(self):
        raw = json.dumps(reply())
        for bad in ('```json\n' + raw + '\n```', raw[:-1] + ',"risk":"duplicate"}', raw.replace('0.001', 'NaN'), '[]'):
            self.assertTrue(parse_reply(bad, {'s01'})[1])

    def test_word_cap_and_extra_fields(self):
        for patch in ({'reason': 'x ' * 101}, {'counter': 4}, {'risk': ''}):
            value = dict(reply(), **patch); self.assertTrue(parse_reply(json.dumps(value), {'s01'})[1])

    def test_unknown_duplicate_or_failed_evidence(self):
        for evidence in (['s02'], ['s01', 's01'], [], [4]):
            self.assertTrue(parse_reply(json.dumps(dict(reply(), evidence_ids=evidence)), {'s01'})[1])

    def test_space_and_bool_guard(self):
        for patch in ({'learning_rate': True}, {'pressure_weight': 3}, {'divergence_weight': .2}):
            self.assertTrue(parse_reply(json.dumps(dict(reply(), **patch)), {'s01'})[1])

    def test_feedback_extra_fields_duplicate_ids_and_false_scores(self):
        for mutation in ('extra', 'duplicate', 'score', 'nan'):
            feedback = fixture()
            if mutation == 'extra': feedback['test_error'] = 0
            if mutation == 'duplicate': feedback['history'] *= 2
            if mutation == 'score': feedback['history'][0]['S'] = .1
            if mutation == 'nan': feedback['history'][0]['pressure_error'] = float('nan')
            with self.assertRaises(ValueError): validate_feedback(feedback)

    def test_failed_history_has_no_score(self):
        value = fixture(); failed = dict(value['history'][0], id='a03', status='timeout', S=None,
                                        velocity_error=None, pressure_error=None, divergence=None)
        value['history'].append(failed); self.assertEqual(validate_feedback(value), {'s01'})
        failed['S'] = .01
        with self.assertRaises(ValueError): validate_feedback(value)

    def test_second_feedback_uses_real_first_result(self):
        shared = [dict(id='s01', **parameters(), S=.3, velocity_error=.2, pressure_error=.4, divergence=.1, training_gpu_hours=50/3600)]
        state = {'remaining_training_seconds': 247., 'state': {'spec': {'attempt_limit': 3}, 'attempts': [
            dict(number=1, phase='finished', status='complete', parameters=parameters(), S=.25, training_seconds=53., decision_seconds=5.)]}}
        measured = {'a01': dict(S=.25, velocity_error=.2, pressure_error=.3, divergence=.2)}
        feedback = build_feedback(shared, state, measured, 2)
        self.assertEqual(feedback['history'][-1]['S'], .25); self.assertEqual(feedback['remaining_attempts'], 1)
        self.assertEqual(feedback['remaining_training_seconds'], 247.)
        with self.assertRaises(ValueError): build_feedback(shared, state, {}, 2)

    def test_failed_attempt_is_visible_and_counted(self):
        shared = [dict(id='s01', **parameters(), S=.3, velocity_error=.2, pressure_error=.4, divergence=.1, training_gpu_hours=50/3600)]
        state = {'remaining_training_seconds': 300., 'state': {'spec': {'attempt_limit': 3}, 'attempts': [
            dict(number=1, phase='finished', status='invalid_reply', parameters=None, S=None, training_seconds=0., decision_seconds=200.)]}}
        feedback = build_feedback(shared, state, {}, 2)
        self.assertEqual(feedback['remaining_attempts'], 1)
        self.assertIsNone(feedback['history'][-1]['S'])
        self.assertEqual(feedback['history'][-1]['decision_seconds'], 200.)


if __name__ == '__main__': unittest.main(verbosity=2)

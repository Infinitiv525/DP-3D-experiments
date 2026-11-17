"""One model-independent prompt/JSON contract; standard library only."""
import hashlib
import json
import math
from search_controls import validate_parameters

VERSION = 3
KEYS = ['learning_rate', 'weight_decay', 'pressure_weight', 'divergence_weight',
        'reason', 'evidence_ids', 'expected_effect', 'risk']
SYSTEM = (
    'Select four training hyperparameters for a fixed steady 3D FNO. No tools, code, internet or dataset access. '
    'The supplied history contains real validation measurements; epochs_per_trial gives the fixed training length. '
    'This is not temporal rollout or free-surface simulation. No test information is available. '
    'Minimize S=0.5*velocity_error+0.5*pressure_error, averaged equally over validation geometries. '
    'Smaller S is better. Divergence is a secondary finite-difference diagnostic; lower alone does not prove physical truth. '
    'All completed trials used the same training length, architecture and train/validation split. '
    'Failed attempts have no usable score, consume an attempt and their recorded resources, and cannot be selected as successful evidence. '
    'You may explore beyond the best observed configuration; do not describe expected improvement as already measured. '
    'Use the actual supplied numbers when describing earlier results; a public justification is not a reasoning trace. '
    'Respect remaining_training_seconds and remaining_attempts. Duplicate configurations are allowed but still cost a full attempt. '
    'Return ONE strict JSON object only: no Markdown, counters, extra keys or text outside JSON. '
    'Use EXACTLY these eight keys: learning_rate, weight_decay, pressure_weight, divergence_weight, reason, evidence_ids, expected_effect, risk. '
    'learning_rate: real [0.0001,0.003]; weight_decay: real [0.000001,0.01]; '
    'pressure_weight: one of [0.25,0.5,1,2,4]; divergence_weight: one of [0,0.001,0.003,0.01,0.03]. '
    'reason, expected_effect and risk must be nonempty short strings, at most 100 whitespace-separated words IN TOTAL across those three strings. '
    'evidence_ids must be a nonempty list of unique IDs of completed trials in the provided history. '
    'Give only a brief public justification, not hidden step-by-step reasoning.'
)
SYSTEM_SHA256 = hashlib.sha256(SYSTEM.encode()).hexdigest()
HISTORY_KEYS = {'id', 'parameters', 'status', 'S', 'velocity_error', 'pressure_error',
                'divergence', 'training_seconds', 'decision_seconds'}
FAILURES = {'failed', 'timeout', 'oom', 'invalid_reply', 'decision_timeout', 'decision_failed', 'interrupted_reserved_cost'}


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative measurement required')


def validate_feedback(value):
    if set(value) != {'contract_version', 'phase', 'epochs_per_trial', 'remaining_training_seconds', 'remaining_attempts', 'history'}:
        raise ValueError('Unexpected feedback fields')
    if value['contract_version'] != VERSION or value['phase'] not in ('technical_pilot', 'main_search'):
        raise ValueError('Invalid feedback phase/version')
    if type(value['epochs_per_trial']) is not int or value['epochs_per_trial'] < 1: raise ValueError('Invalid epochs')
    if type(value['remaining_attempts']) is not int or value['remaining_attempts'] < 1: raise ValueError('No attempt remaining')
    nonnegative(value['remaining_training_seconds'])
    if not isinstance(value['history'], list) or not value['history']: raise ValueError('Nonempty history required')
    seen = set(); complete = set()
    for row in value['history']:
        if set(row) != HISTORY_KEYS: raise ValueError('Unexpected history fields')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in seen: raise ValueError('Unique nonempty history IDs required')
        seen.add(row['id']); nonnegative(row['training_seconds']); nonnegative(row['decision_seconds'])
        if row['parameters'] is not None: validate_parameters(row['parameters'])
        if row['status'] == 'complete':
            if not isinstance(row['parameters'], dict): raise ValueError('Successful trial parameters required')
            validate_parameters(row['parameters'])
            for key in ('S', 'velocity_error', 'pressure_error', 'divergence'): nonnegative(row[key])
            if abs(row['S'] - .5 * (row['velocity_error'] + row['pressure_error'])) > 1e-10: raise ValueError('Inconsistent objective score')
            complete.add(row['id'])
        elif row['status'] not in FAILURES or any(row[k] is not None for k in ('S', 'velocity_error', 'pressure_error', 'divergence')):
            raise ValueError('Invalid failed measurement')
    if not complete: raise ValueError('At least one completed evidence row required')
    canonical(value)
    return complete


def messages(feedback):
    validate_feedback(feedback)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': canonical(feedback)}]


def repair_message(errors):
    return ('Invalid JSON response: ' + ','.join(errors) + '. Return complete corrected strict JSON only, '
            'EXACTLY these eight keys: ' + ', '.join(KEYS) + '. No other keys, Markdown, counters or text. '
            'The three explanation strings must contain at most 100 whitespace-separated words in total.')


def unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj: raise ValueError('Duplicate JSON key')
        obj[key] = value
    return obj


def parse_reply(raw, completed_ids):
    def reject_constant(value): raise ValueError('Nonfinite JSON constant: ' + value)
    try: value = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError): return None, ['strict_json_required']
    if not isinstance(value, dict) or set(value) != set(KEYS): return None, ['exact_eight_keys_required']
    errors = []
    try: validate_parameters({k: value[k] for k in KEYS[:4]})
    except (ValueError, TypeError): errors.append('hyperparameter_space')
    strings = [value[k] for k in ('reason', 'expected_effect', 'risk')]
    if any(not isinstance(s, str) or not s.strip() for s in strings): errors.append('nonempty_explanation_strings')
    elif sum(len(s.split()) for s in strings) > 100: errors.append('explanation_word_cap')
    evidence = value['evidence_ids']
    if (not isinstance(evidence, list) or not evidence or any(not isinstance(i, str) or i not in completed_ids for i in evidence)
            or (all(isinstance(i, str) for i in evidence) and len(set(evidence)) != len(evidence))):
        errors.append('completed_unique_evidence_ids')
    return value, errors

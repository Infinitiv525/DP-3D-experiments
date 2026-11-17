"""UID-aware scheduler adapter; original ledger code/identity remain unchanged.

Compute nodes may display UserId as UID(UID) rather than username(UID).
Verify the authenticated numeric UID before canonicalizing that display field.
Preserve original scheduler text as proof; never infer identity just from a name.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from allocation_budget import PhysicalBudget


def verify_user_display(text, expected_name):
    matches = list(re.finditer(r'(?<!\S)UserId=([^\s]+)', text))
    if len(matches) != 1: raise ValueError('Exactly one scheduler UserId required')
    match = matches[0]; value = match[1]
    parts = re.fullmatch(r'([^()]+)\((\d+)\)', value)
    uid = os.getuid()
    if not parts or int(parts[2]) != uid or parts[1] not in (expected_name, str(uid)):
        raise ValueError('Scheduler identity does not match authenticated UID/name')
    normalized = text[:match.start()] + f'UserId={expected_name}({uid})' + text[match.end():]
    return normalized, {'authenticated_uid': uid, 'original_user_field': value,
                        'original_scheduler_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'original_scheduler_text': text,
                        'normalization': 'Display-name canonicalization ONLY after numeric UID verification'}


def settle(directory, config):
    with PhysicalBudget(directory, **config) as ledger:
        for row in ledger.state['allocations']:
            if row['job_id'] is None: raise RuntimeError('Unattached reservation requires manual reconciliation')
            result = subprocess.run(['scontrol', 'show', 'job', '-o', row['job_id']], capture_output=True, text=True,
                                    timeout=20, check=True)
            normalized, proof = verify_user_display(result.stdout, config['identity']['user'])
            ledger.settle(row['label'], normalized)
            row['scheduler_uid_proof'] = proof
            ledger.save()
        return ledger.report()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--ledger', type=Path, required=True)
    args = p.parse_args(); print(json.dumps(settle(args.ledger, json.loads(args.config.read_text())), allow_nan=False))

"""Pinned inputs/source identity for the v3 preflight; not main authorization."""
import hashlib
import json
from pathlib import Path

from configurator_contract_v3 import SYSTEM_SHA256

SOURCES = ['llm/configurator_contract_v3.py', 'llm/propose_configuration_v3.py', 'llm/sequential_feedback.py',
           'llm/preflight_support.py', 'llm/run_sequential_preflight.py', 'llm/submit_sequential_preflight.py',
           'llm/persistent_search.py', 'llm/search_controls.py', 'llm/allocation_budget.py', 'llm/settle_allocation_uid.py',
           'llm/smoke_local_configurator.py', 'training/train_flowbench_protocol.py', 'training/train_flowbench_smoke.py',
           'training/fno_model.py', 'training/flowbench_physics.py', 'training/flowbench_protocol_metrics.py',
           'config/sequential_preflight_20261021.json', 'config/sequential_allocation_20261021.json',
           'config/llm_configurator_v3_20261021.json', 'scripts/111_sequential_preflight.sbatch']


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_freeze(root, path):
    freeze = read(path)
    if freeze['status'] != 'SEQUENTIAL_TECHNICAL_PREFLIGHT_FROZEN' or freeze['main_authorized']:
        raise ValueError('Only this frozen technical preflight is authorized')
    if freeze['system_prompt_sha256'] != SYSTEM_SHA256: raise ValueError('Prompt differs')
    if set(freeze['sources']) != set(SOURCES): raise ValueError('Source manifest differs')
    for name, expected in freeze['sources'].items():
        if digest(root / 'test' / name) != expected: raise ValueError('Frozen source changed: ' + name)
    for name, expected in freeze['inputs'].items():
        if digest(root / name) != expected: raise ValueError('Frozen input changed: ' + name)
    return freeze

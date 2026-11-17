import argparse
import json
from pathlib import Path
from preflight_support import SOURCES, digest, read
from configurator_contract_v3 import SYSTEM_SHA256


def run(a):
    cfg = read(a.root / 'test/config/sequential_preflight_20261021.json'); base = read(a.root / cfg['pilot_config_relative'])
    inputs = [cfg['source_relative'] + '/feedback.json', cfg['source_relative'] + '/normalization.json',
              cfg['data_audit_relative'] + '/summary.json', base['dataset_relative'] + '/pressure_gauge_audit.json',
              base['dataset_relative'] + '/split.json', cfg['pilot_config_relative']]
    llm = read(a.root / cfg['llm_config_relative'])
    inputs += [cfg['preparation_relative'] + '/' + spec['label'] + '.json' for spec in llm['models']]
    result = {'status': 'SEQUENTIAL_TECHNICAL_PREFLIGHT_FROZEN', 'main_authorized': False,
              'system_prompt_sha256': SYSTEM_SHA256, 'sources': {name: digest(a.root / 'test' / name) for name in SOURCES},
              'inputs': {name: digest(a.root / name) for name in inputs}, 'CPU_tests_required': ['contract_v3', 'persistent', 'allocation', 'uid']}
    with a.output.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], SYSTEM_SHA256)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())

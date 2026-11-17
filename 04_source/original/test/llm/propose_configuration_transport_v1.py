"""Versioned transport adapter around the immutable v3 local-model runner.

Runs in its own process. Only injects the parser; prompts, loaders, generation
and repair limits remain those of v3. Old v3 sources/reports are never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import propose_configuration_v3 as runner
from configurator_transport_v1 import parse_reply, POLICY


def run(args):
    records = []
    def transport_parser(raw, evidence):
        value, errors, transport = parse_reply(raw, evidence)
        records.append(transport)
        return value, errors
    previous = runner.parse_reply
    try:
        runner.parse_reply = transport_parser
        success = runner.run(args)
    finally: runner.parse_reply = previous
    path = args.output / 'summary.json'
    result = json.loads(path.read_text())
    result['transport_policy'] = POLICY
    result['transport_source_sha256'] = hashlib.sha256(Path(__file__).with_name('configurator_transport_v1.py').read_bytes()).hexdigest()
    if len(records) != len(result['attempts']): raise RuntimeError('Parser-call provenance differs')
    for attempt, transport in zip(result['attempts'], records, strict=True):
        attempt['transport'] = transport
        (args.output / f"attempt_{attempt['repair']}.json").write_text(json.dumps(attempt, indent=2, allow_nan=False) + '\n')
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return success


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--feedback', type=Path, required=True); p.add_argument('--index', type=int, required=True); p.add_argument('--output', type=Path, required=True)
    raise SystemExit(0 if run(p.parse_args()) else 1)

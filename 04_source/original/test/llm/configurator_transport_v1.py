"""Uniform transport-only normalization; frozen v3 semantic checks unchanged.

Accept an entire response enclosed in one optional `json` Markdown code fence.
No searching for JSON inside prose, numeric coercion, key repair or truncation.
Keep raw-format compliance distinct from usable structured-payload validity.
"""
import hashlib
import re
from configurator_contract_v3 import parse_reply as strict_parse

POLICY = 'single_complete_json_fence_only_v1'


def parse_reply(raw, completed_ids):
    payload = raw.strip(); removed = False
    fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', payload, flags=re.DOTALL | re.IGNORECASE)
    if fence and '```' not in fence[1]:
        payload = fence[1].strip(); removed = True
    parsed, errors = strict_parse(payload, completed_ids)
    return parsed, errors, {'policy': POLICY, 'single_fence_removed': removed,
                            'raw_strict_json_valid': not strict_parse(raw, completed_ids)[1],
                            'payload_sha256': hashlib.sha256(payload.encode()).hexdigest()}

"""Lossless JSON scalar conversion, validated before any report file write."""
import json
from pathlib import Path
import numpy as np
from native_quality_case_v1 import WORK,read,digest


def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported report object: '+type(value).__name__)


def save(path,value):
    text=json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n'
    path.write_text(text)


def verify_review_extension():
    f=read(WORK/'review_recovery_20261023_v3/freeze.json')
    for name,sha in f['source_files'].items():
        if digest(Path(name))!=sha:raise ValueError('Review source changed')
    if digest(WORK/'force_endpoint_recovery/127805/summary.json')!=f['endpoint_summary_sha256']:
        raise ValueError('Endpoint provenance changed')
    return f

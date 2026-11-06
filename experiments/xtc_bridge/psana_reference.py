"""Frozen prefix identity/calibration contract. No psana import at module scope."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from qrecords import integer, validate_digest, digest_config


def event_timestamp(evt):
    value = evt.timestamp
    if isinstance(value, np.integer):
        value = int(value)
    return integer(value, 'timestamp', maximum=2**64-1)


def calibration_digest(constants):
    """Hash calibration VALUES (not database metadata), including masks and geometry."""
    h = hashlib.sha256()
    for name in sorted(constants):
        value = constants[name][0]
        h.update(name.encode() + b'\0')
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields:
                raise ValueError('unsupported calibration dtype')
            header = json.dumps([value.dtype.str, list(value.shape)]).encode()
            payload = np.ascontiguousarray(value).tobytes()
        else:
            header = b'json'
            payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        h.update(len(header).to_bytes(8, 'little') + header)
        h.update(len(payload).to_bytes(8, 'little') + payload)
    return h.hexdigest()


class Reference:
    def __init__(self, value):
        if set(value) != {'schema_version', 'experiment', 'run', 'calibration_digest', 'timestamps'}:
            raise ValueError('unexpected reference fields')
        if type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError('unsupported reference schema')
        if not isinstance(value['experiment'], str) or not value['experiment'].strip():
            raise ValueError('invalid reference experiment')
        integer(value['run'], 'run')
        validate_digest(value['calibration_digest'], 'calibration_digest')
        stamps = value['timestamps']
        if not isinstance(stamps, list) or not stamps:
            raise ValueError('reference must contain a nonempty event prefix')
        for stamp in stamps:
            integer(stamp, 'timestamp', maximum=2**64-1)
        if any(a >= b for a, b in zip(stamps, stamps[1:])):
            raise ValueError('reference timestamps must be strictly increasing and unique')
        self.value = value
        self.digest = digest_config(value)
        self.ordinals = {stamp: ordinal for ordinal, stamp in enumerate(stamps)}
        self.low, self.high = stamps[0], stamps[-1]

    def ordinal(self, stamp):
        integer(stamp, 'timestamp', maximum=2**64-1)
        if stamp < self.low or stamp > self.high:
            return None
        if stamp not in self.ordinals:
            raise ValueError(f'unexpected in-range timestamp {stamp}')
        return self.ordinals[stamp]

    def check_record(self, record):
        if record['experiment'] != self.value['experiment'] or record['run'] != self.value['run']:
            raise ValueError('record source differs from reference')
        ordinal = self.ordinal(record['timestamp'])
        if ordinal is None or ordinal != record['source_ordinal']:
            raise ValueError('record timestamp/ordinal differs from reference')
        return ordinal

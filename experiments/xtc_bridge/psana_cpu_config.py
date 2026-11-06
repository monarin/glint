"""Strict configuration for the optional native MPI CPU path (stdlib only)."""
import hashlib
import json
import math
from pathlib import Path


def positive(value, name, integer=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def load_config(path):
    path = Path(path)
    c = json.loads(path.read_text())
    required = {'psana_prelude', 'glint_prelude', 'expected_child_prefix', 'source',
                'reference', 'processor', 'devices'}
    defaults = dict(num_groups=2, max_chunk_bytes=8*1024**2, max_group_bytes=64*1024**2,
                    transport_memory_bytes=512*1024**2,
                    activation_timeout=120, call_timeout=300, job_timeout=1200,
                    cell=None, nbest=1)
    allowed = required | set(defaults) | {'geometry_digest'}
    if not isinstance(c, dict) or required-set(c) or set(c)-allowed:
        raise ValueError('missing or unknown CPU configuration fields (integration is unsupported)')
    c = dict(defaults, **c)
    for key in ('psana_prelude', 'glint_prelude', 'expected_child_prefix'):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError(f'{key} must be nonempty')
    for key in ('num_groups', 'max_chunk_bytes', 'max_group_bytes', 'transport_memory_bytes', 'nbest'):
        positive(c[key], key, integer=True)
    for key in ('activation_timeout', 'call_timeout', 'job_timeout'):
        positive(c[key], key)
    if c['max_chunk_bytes'] > c['max_group_bytes']:
        raise ValueError('max_chunk_bytes must not exceed max_group_bytes')
    if 6*c['max_group_bytes']+2*c['max_chunk_bytes'] > c['transport_memory_bytes']:
        raise ValueError('transport_memory_bytes must reserve six group payloads plus two chunks')
    s = c['source']
    if not isinstance(s, dict) or set(s) != {'exp', 'run', 'dir', 'max_events', 'batch_size'}:
        raise ValueError('source requires exp/run/dir/max_events/batch_size only')
    for key in ('run', 'max_events', 'batch_size'):
        positive(s[key], key, integer=True)
    for key in ('exp', 'dir'):
        if not isinstance(s[key], str) or not s[key].strip():
            raise ValueError(f'invalid source {key}')
    s['dir'] = str(Path(s['dir']).resolve())
    if isinstance(c['reference'], str):
        refpath = (path.parent/c['reference']).resolve()
        c['reference'] = json.loads(refpath.read_text())
    # Full Reference validation also occurs inside the MPI application; this
    # launcher stays usable without NumPy/psana in its own environment.
    r = c['reference']
    if (not isinstance(r, dict) or r.get('experiment') != s['exp'] or r.get('run') != s['run']
            or not isinstance(r.get('timestamps'), list) or not r['timestamps']
            or len(r['timestamps']) > s['max_events']):
        raise ValueError('reference source/range does not match configuration')
    p = c['processor']
    defaults = dict(det='jungfrau', wavelength=0.0, energy_det='ebeamh', min_peaks=6,
                    peakfinder='v4', geom=None)
    if not isinstance(p, dict) or 'zdist' not in p or set(p)-set(defaults)-{'zdist','min_pix','son_min','thr_high','thr_low'}:
        raise ValueError('unknown or missing processor options')
    p = dict(defaults, **p)
    if p['peakfinder'] != 'v4':
        raise ValueError('CPU path supports only V4')
    for key in ('det', 'energy_det'):
        if not isinstance(p[key], str) or not p[key].strip():
            raise ValueError(f'invalid processor {key}')
    positive(p['zdist'], 'zdist')
    for key in ('min_peaks', 'min_pix'):
        if key in p:
            positive(p[key], key, integer=True)
    for key in ('wavelength', 'son_min', 'thr_high', 'thr_low'):
        if key in p and (type(p[key]) not in (int,float) or not math.isfinite(p[key]) or p[key]<0):
            raise ValueError(f'invalid {key}')
    c['processor'] = p
    if p['geom'] is not None:
        p['geom'] = str((path.parent/p['geom']).resolve())
        digest = hashlib.sha256(Path(p['geom']).read_bytes()).hexdigest()
        if c.get('geometry_digest', digest) != digest:
            raise ValueError('geometry changed after configuration snapshot')
        c['geometry_digest'] = digest
    else:
        c['geometry_digest'] = None
    if not isinstance(c['devices'], dict) or len(c['devices']) != 2:
        raise ValueError('devices must map the two actual BD world ranks to visible ordinals')
    for key, val in c['devices'].items():
        if not isinstance(key,str) or not key.isdecimal() or str(int(key))!=key or type(val) is not int or val<0:
            raise ValueError('invalid device map')
    if c['cell'] is not None:
        if not isinstance(c['cell'],list) or len(c['cell'])!=6:
            raise ValueError('cell must contain six positive values')
        for value in c['cell']:
            positive(value,'cell')
    return c

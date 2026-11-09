"""Strict CPU-only acceptance comparison. No psana, MPI, CUDA or Torch imports."""
import json
from pathlib import Path
import numpy as np
from psana_reference import Reference
from qspool import iter_records, validate_spools, file_digest


def equivalent_basis(a,b,atol=0.002,rtol=1e-5):
    """Same oriented lattice up to an integer unimodular basis change (Angstrom)."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=(3,3) or b.shape!=(3,3) or not np.isfinite([a,b]).all(): return False
    try: u=np.rint(np.linalg.solve(a,b))
    except np.linalg.LinAlgError: return False
    return bool(abs(round(np.linalg.det(u)))==1 and np.allclose(a@u,b,atol=atol,rtol=rtol))


def read_overlap(reads):
    """Returned bulk byte intervals, not physical-disk reads or unique detector payload."""
    paths={}
    total=0
    for r in reads:
        total+=r['returned']
        paths.setdefault(r['path'],[]).append((r['offset'],r['offset']+r['returned']))
    unique=0
    for intervals in paths.values():
        end=-1
        for lo,hi in sorted(intervals):
            unique+=max(0,hi-max(lo,end));end=max(end,hi)
    return dict(returned_bytes=total,unique_interval_bytes=unique,overlap_bytes=total-unique)


def compare_records(left,right):
    if len(left)!=len(right): raise ValueError('record count differs')
    max_q_error=0.
    for a,b in zip(sorted(left,key=lambda r:r['source_ordinal']),sorted(right,key=lambda r:r['source_ordinal'])):
        for key in a.keys()-{'producer_rank','config_digest','q','peaks'}:
            if a[key]!=b[key]: raise ValueError(f"event {a['timestamp']} differs: {key}")
        if a['q'].shape!=b['q'].shape or not np.allclose(a['q'],b['q'],rtol=1e-10,atol=1e-12):
            raise ValueError(f"event {a['timestamp']} q differs")
        if not np.array_equal(a['peaks'],b['peaks']):
            raise ValueError(f"event {a['timestamp']} peak positions/intensities differ")
        if a['q'].size: max_q_error=max(max_q_error,float(np.max(np.abs(a['q']-b['q']))))
    return max_q_error


def calibration_map(audits,reference):
    result={}
    for a in audits:
        if a['calibrations'] and a['calibration_digests']!=[reference.value['calibration_digest']]:
            raise ValueError('calibration constants changed')
        for row in a['calibrations']:
            stamp=row['timestamp']
            if stamp in result or reference.ordinal(stamp) is None:
                raise ValueError('duplicate or unexpected calibration call')
            if row['frame'] is not None:
                shape=row['frame']['shape']
                if len(shape)!=3 or a['physical_segments']!=[list(range(shape[0]))]:
                    raise ValueError('fixture requires verified dense physical segment order')
            result[stamp]=row['frame']
    if set(result)!=set(reference.ordinals): raise ValueError('missing calibration call')
    return result


def science_config(c):
    return {k:c[k] for k in ('reference','processor','geometry_digest','num_groups','cell','nbest')} | {
        'source':{k:c['source'][k] for k in ('exp','run','dir','max_events')},
        'psana_prelude':c['psana_prelude'],'glint_prelude':c['glint_prelude'],
        'expected_child_prefix':c['expected_child_prefix']}


def read_rows(directory,config):
    summaries=[json.loads(p.read_text()) for p in sorted(directory.glob('producer-*/manifest.json'))]
    if not summaries: raise ValueError('no producer manifests')
    rows=[r for s in summaries for r in iter_records(directory,s,config['max_chunk_bytes'])]
    return summaries,rows


def compare_streams(a,b,expected_events):
    # Exact serialized output is a deliberately stricter gate than equivalent cells.
    # This also checks wavelengths, geometry, peaks/intensities, HKLs and metadata.
    if a.read_bytes()!=b.read_bytes(): raise ValueError(f'stream differs: {a.name}')
    left=json.loads(Path(str(a)+'.index.json').read_text())
    right=json.loads(Path(str(b)+'.index.json').read_text())
    for report in (left,right):
        if ([r['event'] for r in report['frames']]!=expected_events or
                report['mpi_imported'] or report['psana_imported'] or not report['xtc_python_open_guard']):
            raise ValueError('index output identity/isolation differs')
    for key in ('seed','torch','numpy','matmul_precision','tf32','strict_gate'):
        if left[key]!=right[key]: raise ValueError('index configuration differs: '+key)
    indexed=[];strict=[]
    for x,y in zip(left['frames'],right['frames']):
        for key in ('indexed','peak_count','matched_strict','strict_pass'):
            if x[key]!=y[key]: raise ValueError('indexed event diagnostics differ: '+key)
        gate=left['strict_gate']
        expected=bool(x['indexed'] and x['matched_strict']>=gate['minimum'] and
                      x['matched_strict']>=gate['fraction']*x['peak_count'])
        if x['strict_pass']!=expected: raise ValueError('inconsistent strict index gate')
        if expected: strict.append(x['event'])
        if x['indexed']:
            if not equivalent_basis(x['M'],y['M']): raise ValueError('oriented lattice differs')
            indexed.append(x['event'])
    return dict(raw=indexed,strict=strict)


def compare(baseline,native):
    baseline,native=Path(baseline),Path(native)
    config=json.loads((baseline/'config.json').read_text())
    current=json.loads((native/'config.json').read_text())
    if science_config(config)!=science_config(current): raise ValueError('frozen science configuration differs')
    reference=Reference(config['reference'])
    wanted=[];ba=[]
    for g in range(config['num_groups']):
        directory=baseline/f'group-{g}'
        report=json.loads((directory/'baseline-complete.json').read_text())
        if not report['child_reaped'] or report['group']!=g: raise ValueError('baseline incomplete')
        _,rows=read_rows(directory,config);wanted.extend(rows)
        ba.append(json.loads((directory/f'audit-{g}.json').read_text()))
        if ba[-1]['seen']!=reference.value['timestamps']: raise ValueError('baseline traversal differs')
    summaries,actual=read_rows(native,current)
    gate=validate_spools(native,summaries,reference,current)
    complete=json.loads((native/'mpi-complete.json').read_text())
    if complete['production']!=gate or complete['status']!='complete' or complete['harness'] is not False:
        raise ValueError('native completion differs')
    audits=[json.loads((native/f'audit-{rank}.json').read_text()) for rank in range(4)]
    delivered=[t for a in audits for t in a['seen'] if reference.ordinal(t) is not None]
    if sorted(delivered)!=reference.value['timestamps']:
        raise ValueError('native delivery ownership differs')
    for r in complete['ranks']:
        if r['role']!='bd' and (audits[r['rank']]['seen'] or audits[r['rank']]['calibrations'] or r['child'] is not None):
            raise ValueError('control role processed data')
        if r['role']=='bd' and (not audits[r['rank']]['calibrations'] or not r['child_reaped']):
            raise ValueError('required BD participation incomplete')
        for result in r['groups']:
            path=native/f"group-{result['group']}.stream"
            if file_digest(path)!=result['sha256'] or path.stat().st_size!=result['output_bytes']:
                raise ValueError('group stream changed after completion')
    merged=native/'merged.stream'
    if (file_digest(merged)!=complete['merged']['sha256'] or
            merged.stat().st_size!=complete['merged']['output_bytes']):
        raise ValueError('merged stream changed after completion')
    if calibration_map(ba,reference)!=calibration_map(audits,reference):
        raise ValueError('calibrated image diagnostics differ')
    error=compare_records(wanted,actual)
    indexed=[];raw_indexed=[]
    for g in range(config['num_groups']):
        events=sorted(r['source_ordinal'] for r in actual if r['group_id']==g and r['status']=='hit')
        result=compare_streams(baseline/f'group-{g}'/f'group-{g}.stream',native/f'group-{g}.stream',events)
        indexed+=result['strict'];raw_indexed+=result['raw']
    return dict(parity_pass=True,events=len(actual),hits=gate['statuses']['hit'],
        raw_indexed_events=sorted(raw_indexed),indexed_events=sorted(indexed),indexed_coverage=bool(indexed),max_q_abs_error=error,
        batch_size=current['source']['batch_size'],devices=current['devices'],
        owners={str(r['timestamp']):r['producer_rank'] for r in actual},
        calibration_attempts=dict(baseline=sum(len(a['calibrations']) for a in ba),
                                  native=sum(len(a['calibrations']) for a in audits)),
        reads=dict(baseline=read_overlap([r for a in ba for r in a['reads']]),
                   native=read_overlap([r for a in audits for r in a['reads']])),
        rss_scope='psana-side processes only; excludes GLINT children',
        audits=dict(baseline=[{k:a[k] for k in ('seconds','max_rss_kib')} for a in ba],
                    native=[{k:a[k] for k in ('seconds','max_rss_kib')} for a in audits]))

"""Byte-bounded compact chunks and reference-checked group assembly; no MPI/CUDA."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import qrecords as qr
from glint_psana_common import atomic_json


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def records_digest(records):
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: r['source_ordinal']):
        qr.validate_record(record)
        metadata = qr._metadata(record)
        h.update(len(metadata).to_bytes(8, 'little') + metadata)
        h.update(record['q'].tobytes())
        h.update(record['peaks'].tobytes())
    return h.hexdigest()


class RecordWriter:
    def __init__(self, directory, rank, config_digest, max_bytes):
        qr.integer(max_bytes, 'max_chunk_bytes', 1)
        qr.integer(rank, 'producer rank')
        qr.validate_digest(config_digest, 'config_digest')
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.rank, self.config_digest, self.max_bytes = rank, config_digest, max_bytes
        self.pending, self.chunks, self.used, self.count = [], [], 0, 0
        self.finished = False

    def append(self, record):
        if self.finished:
            raise RuntimeError('writer already finished')
        size = qr.record_nbytes(record)
        if size > self.max_bytes:
            raise ValueError('one event exceeds max_chunk_bytes')
        if record['producer_rank'] != self.rank or record['config_digest'] != self.config_digest:
            raise ValueError('record differs from writer owner/configuration')
        if self.used + size > self.max_bytes:
            self.flush()
        self.pending.append(record)
        self.used += size
        self.count += 1

    def flush(self):
        if not self.pending:
            return
        records = self.pending
        header = json.dumps([json.loads(qr._metadata(r)) for r in records],
                            sort_keys=True, separators=(',', ':')).encode()
        offsets = np.array([0] + [len(r['q']) for r in records], dtype=np.int64).cumsum()
        name = f'chunk-{len(self.chunks):06d}.npz'
        path = self.directory/name
        fd, temporary = tempfile.mkstemp(dir=self.directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, metadata=np.frombuffer(header, np.uint8), offsets=offsets,
                         q=np.concatenate([r['q'] for r in records]),
                         peaks=np.concatenate([r['peaks'] for r in records]))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.chunks.append(dict(name=name, sha256=file_digest(path), records=len(records),
                                payload_bytes=self.used, file_bytes=path.stat().st_size))
        self.pending, self.used = [], 0

    def finish(self, audit):
        if self.finished:
            raise RuntimeError('writer already finished')
        self.flush()
        summary = dict(rank=self.rank, config_digest=self.config_digest, count=self.count,
                       chunks=self.chunks, out_of_range=audit)
        atomic_json(self.directory/'manifest.json', summary)
        self.finished = True
        return summary


def read_chunk(path, entry, max_bytes):
    path = Path(path)
    if path.stat().st_size != entry['file_bytes'] or file_digest(path) != entry['sha256']:
        raise ValueError('chunk size/digest mismatch')
    if entry['payload_bytes'] > max_bytes:
        raise ValueError('chunk exceeds configured payload limit')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos)!=4 or {x.filename for x in infos}!={'metadata.npy','offsets.npy','q.npy','peaks.npy'}:
            raise ValueError('unexpected chunk members')
        # Per-record offsets/JSON delimiters and fixed NPY headers are storage overhead.
        if sum(x.file_size for x in infos) > max_bytes + entry['records']*16 + 4096:
            raise ValueError('chunk archive exceeds size limit')
    with np.load(path, allow_pickle=False) as arrays:
        metadata, offsets, q, peaks = (arrays[k] for k in ('metadata','offsets','q','peaks'))
    if metadata.dtype != np.uint8 or metadata.ndim!=1:
        raise ValueError('invalid chunk metadata')
    headers = json.loads(metadata.tobytes().decode())
    if (not isinstance(headers,list) or len(headers)!=entry['records'] or
            offsets.dtype!=np.int64 or offsets.shape!=(len(headers)+1,) or offsets[0]!=0 or
            np.any(offsets[1:]<offsets[:-1]) or offsets[-1]!=len(q) or peaks.shape!=q.shape):
        raise ValueError('invalid chunk offsets/counts')
    records=[]
    for i, header in enumerate(headers):
        if not isinstance(header,dict) or 'q' in header or 'peaks' in header:
            raise ValueError('invalid record header')
        lo, hi = int(offsets[i]), int(offsets[i+1])
        records.append(qr.validate_record(dict(header, q=q[lo:hi], peaks=peaks[lo:hi])))
    if sum(qr.record_nbytes(r) for r in records)!=entry['payload_bytes']:
        raise ValueError('chunk payload size mismatch')
    return records


def iter_records(output, summary, max_bytes):
    directory = Path(output)/f'producer-{summary["rank"]}'
    count=0
    names=set()
    for entry in summary['chunks']:
        name=entry['name']
        if not isinstance(name,str) or Path(name).name!=name or name in names:
            raise ValueError('invalid/duplicate chunk name')
        names.add(name)
        for record in read_chunk(directory/name, entry, max_bytes):
            if record['producer_rank']!=summary['rank'] or record['config_digest']!=summary['config_digest']:
                raise ValueError('spool record owner/config mismatch')
            count+=1
            yield record
    if count!=summary['count']:
        raise ValueError('manifest record count mismatch')


def validate_spools(output, summaries, reference, config):
    if len({s['rank'] for s in summaries})!=len(summaries):
        raise ValueError('duplicate producer manifest')
    digests={s['config_digest'] for s in summaries}
    if len(digests)!=1:
        raise ValueError('mixed producer configurations')
    seen=set(); sizes=[0]*config['num_groups']; statuses={s:0 for s in qr.STATUSES}; audit=[]
    for summary in summaries:
        for stamp in summary['out_of_range']:
            if reference.ordinal(stamp) is not None:
                raise ValueError('in-range timestamp was incorrectly excluded')
            audit.append(dict(rank=summary['rank'],timestamp=stamp))
        for record in iter_records(output, summary, config['max_chunk_bytes']):
            ordinal=reference.check_record(record)
            if ordinal in seen:
                raise ValueError('duplicate event ownership across producers')
            seen.add(ordinal)
            if record['num_groups']!=config['num_groups']:
                raise ValueError('record group count mismatch')
            sizes[record['group_id']]+=qr.record_nbytes(record)
            if sizes[record['group_id']]>config['max_group_bytes']:
                raise ValueError('complete group exceeds max_group_bytes; cannot split consensus')
            statuses[record['status']]+=1
    if seen!=set(reference.ordinals.values()):
        raise ValueError('missing expected events (including non-hits)')
    return dict(config_digest=next(iter(digests)), count=len(seen), statuses=statuses,
                group_bytes=sizes, out_of_range=audit)


def load_group(output, summaries, reference, config, group):
    records=[]; used=0
    for summary in summaries:
        for record in iter_records(output, summary, config['max_chunk_bytes']):
            if record['group_id']==group:
                reference.check_record(record)
                used+=qr.record_nbytes(record)
                if used>config['max_group_bytes']:
                    raise ValueError('complete group exceeds max_group_bytes')
                # Detach from chunk backing arrays so one surviving row cannot retain a whole chunk.
                records.append(dict(record,q=record['q'].copy(),peaks=record['peaks'].copy()))
    records.sort(key=lambda r:r['source_ordinal'])
    return records

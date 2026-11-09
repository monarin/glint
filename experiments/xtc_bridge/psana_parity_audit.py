"""Validation-only observers. Never installed by the production launcher."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch
import numpy as np
from glint_probe_common import identity
from psana_reference import calibration_digest, event_timestamp


def frame_digest(frame):
    if frame is None:
        return None
    a=np.asarray(frame)
    if a.dtype.hasobject:
        raise ValueError('object calibration array')
    h=hashlib.sha256(str((a.dtype.str,a.shape)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    finite=np.isfinite(a)
    return dict(sha256=h.hexdigest(),shape=list(a.shape),dtype=a.dtype.str,
                finite=int(finite.sum()),size=int(a.size))


class Proxy:
    def __init__(self, target): self.target=target
    def __getattr__(self, name): return getattr(self.target,name)


class Audit:
    def __init__(self, detector):
        self.detector=detector
        self.seen=[]; self.calibrations=[]; self.finding={}; self.reads=[]
        self.constants=[]; self.segments=[]; self.seconds={}
        self.started=time.monotonic()

    @contextmanager
    def timed(self, name):
        start=time.monotonic()
        try: yield
        finally: self.seconds[name]=self.seconds.get(name,0.)+time.monotonic()-start

    def wrap_run(self, run):
        audit=self
        class Run(Proxy):
            def Detector(self,name,*args,**kwargs):
                det=self.target.Detector(name,*args,**kwargs)
                if name!=audit.detector: return det
                digest=calibration_digest(det.calibconst)
                if digest not in audit.constants: audit.constants.append(digest)
                ids=[int(x) for x in det.raw._segment_numbers]
                if ids not in audit.segments: audit.segments.append(ids)
                class Raw(Proxy):
                    def calib(self,evt,*args,**kwargs):
                        with audit.timed('calibration'):
                            frame=self.target.calib(evt,*args,**kwargs)
                        with audit.timed('diagnostic_hash'):
                            diagnostic=frame_digest(frame)
                        audit.calibrations.append(dict(timestamp=event_timestamp(evt),frame=diagnostic))
                        return frame
                wrapped=Proxy(det);wrapped.raw=Raw(det.raw)
                return wrapped
            def events(self):
                for evt in self.target.events():
                    audit.seen.append(event_timestamp(evt))
                    yield evt
        return Run(run)

    def factory(self, factory):
        audit=self
        def create(**kwargs):
            ds=factory(**kwargs)
            class DS(Proxy):
                def runs(self):
                    for run in self.target.runs(): yield audit.wrap_run(run)
            return DS(ds)
        return create

    @contextmanager
    def bulk_reads(self):
        # This observes Python EventManager bulk reads, not Cython SMD reads,
        # other file access, or physical storage traffic. Retries are inside _read.
        from psana.psexp.event_manager import EventManager
        original=EventManager._read
        audit=self
        def read(manager,fd,size,offset):
            path=os.readlink(f'/proc/self/fd/{fd}')
            with audit.timed('bulk_read'):
                data=original(manager,fd,size,offset)
            audit.reads.append(dict(path=path,offset=int(offset),requested=int(size),returned=len(data)))
            return data
        with patch.object(EventManager,'_read',read): yield

    @contextmanager
    def legacy_finding(self):
        import xtc_core
        original=xtc_core.frame_q
        audit=self
        def find(*args,**kwargs):
            stamp=audit.calibrations[-1]['timestamp']
            if stamp in audit.finding: raise ValueError('duplicate legacy finding')
            with audit.timed('peak_q'):
                result=original(*args,**kwargs)
            audit.finding[stamp]=float(args[6])  # wavelength in the unchanged frame_q API
            return result
        with patch.object(xtc_core,'frame_q',find): yield

    def report(self):
        import psana
        return dict(identity(),psana_file=psana.__file__,seen=self.seen,
                    calibrations=self.calibrations,calibration_digests=self.constants,
                    physical_segments=self.segments,reads=self.reads,
                    bulk_returned_bytes=sum(x['returned'] for x in self.reads),
                    seconds=dict(self.seconds,wall=time.monotonic()-self.started),
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    read_scope='EventManager._read returned bigdata bytes; excludes SMD/metadata and OS cache effects')

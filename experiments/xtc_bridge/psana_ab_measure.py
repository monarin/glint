"""A/B-only aggregate counters and timing; no calibrated image hashing or read logs."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch
from glint_probe_common import atomic_json


class Proxy:
    def __init__(self,target): self.target=target
    def __getattr__(self,name): return getattr(self.target,name)


class Metrics:
    def __init__(self,detector):
        self.detector=detector;self.seconds={};self.read_calls=0;self.read_bytes=0;self.calib_calls=0
    @contextmanager
    def timed(self,name):
        start=time.perf_counter()
        try: yield
        finally: self.seconds[name]=self.seconds.get(name,0)+time.perf_counter()-start
    def factory(self,original):
        metrics=self
        def detector(original,name):
            det=original(name)
            if name!=metrics.detector: return det
            raw=Proxy(det.raw)
            def calib(*a,**kw):
                metrics.calib_calls+=1
                with metrics.timed('calibration'): return det.raw.calib(*a,**kw)
            raw.calib=calib;wrapped=Proxy(det);wrapped.raw=raw;return wrapped
        def factory(*a,**kw):
            ds=original(*a,**kw);wrapped=Proxy(ds)
            def runs():
                for run in ds.runs():
                    r=Proxy(run);r.Detector=lambda name,run=run:detector(run.Detector,name)
                    yield r
            wrapped.runs=runs;return wrapped
        return factory
    @contextmanager
    def processing(self):
        from psana.psexp.event_manager import EventManager
        import xtc_core
        read=EventManager._read;frame_q=xtc_core.frame_q
        def observed_read(manager,fd,size,offset):
            with self.timed('bulk_read'): value=read(manager,fd,size,offset)
            self.read_calls+=1;self.read_bytes+=len(value);return value
        def observed_frame(*a,**kw):
            # Existing frame_q returns host arrays, completing its GPU work.
            with self.timed('peak_q'): return frame_q(*a,**kw)
        with patch.object(EventManager,'_read',observed_read),patch.object(xtc_core,'frame_q',observed_frame):
            yield
    def report(self):
        return dict(seconds=self.seconds,read_calls=self.read_calls,read_bytes=self.read_bytes,
                    calibration_calls=self.calib_calls,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),executable=sys.executable,
                    python=sys.version,numpy=getattr(sys.modules.get('numpy'),'__version__',None),
                    psana_path=getattr(sys.modules.get('psana'),'__file__',None))


def read_legacy(*args,_parent_pid=None,**kwargs):
    """Actual serial reader with matched source guard and scalar observation only."""
    import ctypes
    import signal
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise RuntimeError('parent death guard failed')
    if os.getppid()!=_parent_pid:raise RuntimeError('reader parent changed')
    os.environ['PS_PARALLEL']='none' 
    import inspect
    import psana
    import xtc_qreader
    from psana_cpu_config import load_config
    config=load_config(os.environ['GLINT_AB_CONFIG'])
    bound=inspect.signature(xtc_qreader.run_to_qframes).bind(*args,**kwargs);bound.apply_defaults()
    rank=bound.arguments['rank'];metrics=Metrics(config['processor']['det']);original=psana.DataSource
    def factory(**source):
        if source!={'exp':config['source']['exp'],'run':config['source']['run']}:
            raise ValueError('unexpected legacy source')
        return metrics.factory(original)(**config['source'])
    with patch.object(psana,'DataSource',factory),metrics.processing(),metrics.timed('reader_total'):
        out=xtc_qreader.run_to_qframes(*args,**kwargs)
    report=metrics.report();report.update(events=out['n_events'],hits=out['n_sent'])
    atomic_json(Path(os.environ['GLINT_AB_OUTPUT'])/f'reader-{rank}.json',report)
    return out


def fingerprint(out):
    """Small output q/peaks only; never reads a calibrated image."""
    import numpy as np
    digest=hashlib.sha256()
    digest.update(json.dumps(dict(events=[int(e) for e in out['events']],lams=out['lams']),sort_keys=True).encode())
    for key in ('qframes','peaks'):
        for value in out[key]:
            a=np.ascontiguousarray(value,dtype=np.float64)
            digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def install_index():
    import glint_xtc
    original=glint_xtc.index_and_write
    def index(out,args,output,report=True):
        # Preserve lazy CUDA/Torch loading on the legacy path until reading is done.
        import random
        import numpy as np
        import torch
        from glint.stream import has_cell
        from glint.glint_fast import matched_strict,GATE_MIN,GATE_FRAC,GATE_TOL
        from psana_parity_worker import index_rows
        random.seed(0);np.random.seed(0)
        torch.random.default_generator.manual_seed(0);torch.cuda.manual_seed(0)
        torch.cuda.synchronize();started=time.perf_counter()
        results,stats,count=original(out,args,output,report=report)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        gate=dict(minimum=GATE_MIN,fraction=GATE_FRAC,tolerance=GATE_TOL)
        rows=index_rows(results,out['events'],out['qframes'],has_cell,matched_strict,gate)
        atomic_json(Path(str(output)+'.bench.json'),dict(index_seconds=seconds,
            q_peaks_sha256=fingerprint(out),events=out['events'],raw_indexed=int(count),
            strict_indexed=[r['event'] for r in rows if r['strict_pass']],frames=rows,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            torch=torch.__version__,numpy=np.__version__,seed=0,strict_gate=gate,
            matmul_precision=torch.get_float32_matmul_precision(),tf32=bool(torch.backends.cuda.matmul.allow_tf32)))
        return results,stats,count
    glint_xtc.index_and_write=index


def initialize(parent,config):
    from glint_qconsumer import initialize as original
    result=original(parent,config);install_index();return result

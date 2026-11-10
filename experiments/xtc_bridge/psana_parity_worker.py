"""Acceptance-only GLINT instrumentation; production consumers do not import this."""
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
from glint_psana_common import atomic_json

_ready=None


def index_rows(results,events,qframes,has_cell,matched_strict,gate):
    """Build diagnostics from complete input frames; dependencies are explicit for CPU tests."""
    if len(events)!=len(qframes) or len(set(events))!=len(events):
        raise ValueError('invalid input frame mapping')
    full_q=dict(zip(events,qframes))
    rows=[]
    for r in results or []:
        event=int(r['event']);q=full_q[event];indexed=bool(has_cell(r.get('M')))
        matched=int(matched_strict(r['M'],q)) if indexed else 0
        rows.append(dict(event=event,indexed=indexed,
            M=np.asarray(r['M']).tolist() if indexed else None,
            peak_count=len(q),matched_strict=matched,
            strict_pass=bool(indexed and matched>=gate['minimum'] and matched>=gate['fraction']*len(q))))
    return rows


def initialize(parent,config):
    global _ready
    from glint_qconsumer import initialize as initialize_consumer
    _ready=initialize_consumer(parent,config)
    # Fail a Python-level attempt to open raw XTC in the validation child.
    # C-library arbitrary IO is not intercepted; production isolation also forbids psana.
    def deny_xtc(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0])
            if name.endswith(('.xtc','.xtc2')):
                raise RuntimeError('acceptance child attempted XTC access')
    sys.addaudithook(deny_xtc)
    guard_observed=False
    try:
        open('/nonexistent/glint-parity-guard-probe.xtc2', 'rb')
    except RuntimeError as exc:
        guard_observed=str(exc)=='acceptance child attempted XTC access'
    except OSError:
        pass
    if not guard_observed: raise RuntimeError('XTC audit guard self-test failed')
    import torch
    import glint_xtc
    from glint.glint_fast import matched_strict, GATE_MIN, GATE_FRAC, GATE_TOL
    from glint.stream import has_cell
    original=glint_xtc.index_and_write
    def index(out,args,output,report=True):
        random.seed(0);np.random.seed(0)
        torch.random.default_generator.manual_seed(0);torch.cuda.manual_seed(0)
        results,stats,count=original(out,args,output,report=report)
        gate=dict(minimum=GATE_MIN,fraction=GATE_FRAC,tolerance=GATE_TOL)
        rows=index_rows(results,out['events'],out['qframes'],has_cell,matched_strict,gate)
        atomic_json(Path(str(output)+'.index.json'),dict(
            frames=rows,strict_gate=gate,seed=0,torch=torch.__version__,numpy=np.__version__,
            matmul_precision=torch.get_float32_matmul_precision(),
            tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            xtc_python_open_guard=guard_observed,mpi_imported='mpi4py' in sys.modules,
            psana_imported='psana' in sys.modules))
        return results,stats,count
    glint_xtc.index_and_write=index
    return _ready


def index_legacy(out,config,output):
    if _ready is None: raise RuntimeError('initialize first')
    from glint_xtc import index_and_write
    from glint_qconsumer import isolated
    isolated()
    p=config['processor'];s=config['source']
    args=SimpleNamespace(exp=s['exp'],run=s['run'],geom=p['geom'],zdist=p['zdist'],
        wavelength=p['wavelength'],cell=[str(x) for x in config['cell']] if config['cell'] else None,
        nbest=config['nbest'],integrate=False)
    _,_,count=index_and_write(out,args,output,report=False)
    isolated()
    return dict(n_frames=len(out['events']),n_indexed=int(count))

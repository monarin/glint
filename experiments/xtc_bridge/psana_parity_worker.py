"""Acceptance-only GLINT instrumentation; production consumers do not import this."""
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
from glint_probe_common import atomic_json

_ready=None


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
    import torch
    import glint_xtc
    from glint.stream import has_cell
    original=glint_xtc.index_and_write
    def index(out,args,output,report=True):
        random.seed(0);np.random.seed(0)
        torch.random.default_generator.manual_seed(0);torch.cuda.manual_seed(0)
        results,stats,count=original(out,args,output,report=report)
        rows=[dict(event=int(r['event']),indexed=bool(has_cell(r.get('M'))),
                   M=np.asarray(r['M']).tolist() if has_cell(r.get('M')) else None)
              for r in (results or [])]
        atomic_json(Path(str(output)+'.index.json'),dict(
            frames=rows,seed=0,torch=torch.__version__,numpy=np.__version__,
            matmul_precision=torch.get_float32_matmul_precision(),
            tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            xtc_python_open_guard=True,mpi_imported='mpi4py' in sys.modules,
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

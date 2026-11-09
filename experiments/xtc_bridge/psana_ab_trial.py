"""Benchmark entry points around unchanged GLINT MPI and native psana producers."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch
from glint_probe_common import atomic_json
from psana_cpu_config import load_config
from psana_ab_measure import Metrics


def legacy_mask(mask,rank):
    devices=mask.split(',')
    if len(devices)!=2 or any(not x.strip() for x in devices) or rank not in (0,1):
        raise ValueError('legacy launcher requires exactly two allocated GPU tokens')
    return devices[rank].strip()


def legacy(config,output):
    # Launcher-owned selection BEFORE importing MPI/CUDA, within the allocated mask.
    rank=int(os.environ['OMPI_COMM_WORLD_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES']=legacy_mask(os.environ['CUDA_VISIBLE_DEVICES'],rank)
    from mpi4py import MPI
    import glint_xtc
    import glint_xtc_mpi
    import envbridge
    from psana_ab_measure import install_index
    metrics=Metrics(config['processor']['det'])
    original_read=glint_xtc.read_qframes;original_call=envbridge.Worker.call
    def read(*a,**kw):
        with metrics.timed('reader_bridge_total'):return original_read(*a,**kw)
    def call(worker,fn,*a,**kw):
        if fn=='xtc_qreader:run_to_qframes':
            fn='psana_ab_measure:read_legacy';kw['_parent_pid']=os.getpid()
        return original_call(worker,fn,*a,**kw)
    root=Path(__file__).resolve().parent
    def conda(*a,**kw):
        return envbridge.Env(config['psana_prelude'],pythonpath=(str(root),str(root.parents[1])),inherit_pythonpath=False)
    install_index()
    source=config['source'];p=config['processor']
    argv=['--psana','2','--exp',source['exp'],'--run',str(source['run']),
          '--max-events',str(source['max_events']),'--keep-parts','--out',str(output/'merged.stream'),
          '--nbest',str(config['nbest']),'--bridge-timeout',str(config['call_timeout'])]
    for key in ('det','zdist','wavelength','energy_det','min_peaks','peakfinder','geom','min_pix','son_min','thr_high','thr_low'):
        if key in p and p[key] is not None:argv += ['--'+key.replace('_','-'),str(p[key])]
    if config['cell']:argv+=['--cell']+[str(x) for x in config['cell']]
    with patch.object(glint_xtc,'read_qframes',read),patch.object(envbridge.Env,'conda',conda),patch.object(envbridge.Worker,'call',call),metrics.timed('application'):
        code=glint_xtc_mpi.main(argv)
    if code:MPI.COMM_WORLD.Abort(code)
    atomic_json(output/f'glint-{rank}.json',metrics.report())


def native(config,path,output):
    os.environ.update(PS_PARALLEL='mpi',PS_EB_NODES='1',PS_EB_NODE_LOCAL='0',PS_SRV_NODES='0')
    import psana_mpi_producer as pipeline
    from envbridge import Worker
    metrics=Metrics(config['processor']['det'])
    produce,consume,finalize=pipeline.produce,pipeline.consume,pipeline.finalize
    class ObservedWorker(Worker):
        def start(self):
            with metrics.timed('child_start'):return super().start()
        def call(self,fn,*a,**kw):
            if fn=='glint_qconsumer:initialize':fn='psana_ab_measure:initialize'
            with metrics.timed('child_initialize' if fn.endswith(':initialize') else 'child_index_rpc'):
                return super().call(fn,*a,**kw)
    def observed_produce(c,o,comm,dsf,pf):
        with metrics.processing(),metrics.timed('production'):
            return produce(c,o,comm,metrics.factory(dsf),pf)
    def observed_consume(*a):
        with metrics.timed('consumption'):return consume(*a,worker_factory=ObservedWorker)
    def observed_finalize(*a):
        with metrics.timed('merge'):return finalize(*a)
    with patch.object(pipeline,'produce',observed_produce),patch.object(pipeline,'consume',observed_consume),patch.object(pipeline,'finalize',observed_finalize),metrics.timed('application'):
        code=pipeline.main(['--config',str(path),'--output',str(output)])
    if code:raise RuntimeError('native benchmark failed')
    from mpi4py import MPI
    atomic_json(output/f'native-{MPI.COMM_WORLD.Get_rank()}.json',metrics.report())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('A','B'));parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=load_config(args.config)
    if config['num_groups']!=2:raise ValueError('A/B requires two matching consensus groups')
    os.environ.update(GLINT_AB_CONFIG=str(args.config.resolve()),GLINT_AB_OUTPUT=str(args.output.resolve()),
                      GLINT_PSANA_PROBE_ID=str(args.output.resolve()))
    if args.mode=='A':legacy(config,args.output)
    else:native(config,args.config,args.output)


if __name__=='__main__':main()

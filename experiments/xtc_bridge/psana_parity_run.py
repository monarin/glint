"""Validation-only entry point: unchanged serial reader or instrumented native MPI path."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from envbridge import Env, Worker
from glint_psana_common import atomic_json, identity, gpu_identity
from psana_cpu_config import load_config
from psana_reference import Reference
from psana_parity_audit import Audit, Proxy
from qspool import RecordWriter
import qrecords


def worker_type(audit):
    class ObservedWorker(Worker):
        def start(self):
            with audit.timed('child_start'): return super().start()
        def call(self,fn,*args,**kwargs):
            if fn=='glint_qconsumer:initialize': fn='psana_parity_worker:initialize'
            with audit.timed('child_initialize' if fn.endswith(':initialize') else 'child_index'):
                return super().call(fn,*args,**kwargs)
    return ObservedWorker


def baseline(config,output,group,audit):
    import psana
    import xtc_qreader
    ref=Reference(config['reference']);source=config['source'];p=config['processor']
    if group not in range(config['num_groups']): raise ValueError('invalid baseline group')
    parent=dict(identity(),rank=group,role='bd',gpu=gpu_identity(group%2))
    original=psana.DataSource
    # The legacy function has no dir/max_events DataSource kwargs. Inject only
    # source selection and a matched positive delivery guard in this test process.
    def factory(**kwargs):
        if kwargs!={'exp':source['exp'],'run':source['run']}:
            raise ValueError('legacy source changed')
        return audit.factory(original)(**source)
    with patch.object(psana,'DataSource',factory),audit.legacy_finding(),audit.timed('production'):
        out=xtc_qreader.run_to_qframes(source['exp'],source['run'],rank=group,
            nranks=config['num_groups'],max_events=len(ref.ordinals),**p)
    if audit.seen!=ref.value['timestamps']:
        raise ValueError('serial reader did not traverse exactly the frozen prefix')
    if audit.constants!=[ref.value['calibration_digest']]:
        raise ValueError('serial calibration provenance changed')
    expected=list(range(group,len(ref.ordinals),config['num_groups']))
    calls=[ref.ordinal(x['timestamp']) for x in audit.calibrations]
    if calls!=expected: raise ValueError('serial calibration ownership differs')
    hits={ordinal:i for i,ordinal in enumerate(out['events'])}
    if len(hits)!=len(out['events']): raise ValueError('duplicate legacy hit')
    digest=qrecords.digest_config(config)
    writer=RecordWriter(output/f'producer-{group}',group,digest,config['max_chunk_bytes'])
    for entry,ordinal in zip(audit.calibrations,expected):
        stamp=entry['timestamp'];i=hits.get(ordinal)
        status=('hit' if i is not None else 'missing_calibration' if entry['frame'] is None
                else 'below_min_peaks' if stamp in audit.finding else 'invalid_wavelength')
        row=dict(schema_version=1,config_digest=digest,experiment=source['exp'],run=source['run'],
            timestamp=stamp,source_ordinal=ordinal,group_id=group,num_groups=config['num_groups'],
            producer_rank=group,status=status,calibration_attempts=1,
            wavelength=out['lams'][i] if i is not None else audit.finding.get(stamp),
            peak_count=len(out['qframes'][i]) if i is not None else 0,
            q=np.ascontiguousarray(out['qframes'][i],dtype=np.float64) if i is not None else np.empty((0,3)),
            peaks=np.ascontiguousarray(out['peaks'][i],dtype=np.float64) if i is not None else np.empty((0,3)))
        writer.append(row)
    summary=writer.finish([])
    # Confirm reconstruction retained the UNCHANGED legacy dictionary/counters.
    from qspool import iter_records
    rebuilt=qrecords.to_legacy(list(iter_records(output,summary,config['max_chunk_bytes'])),
        experiment=source['exp'],run=source['run'],config_digest=digest,group_id=group,
        num_groups=config['num_groups'],expected_ordinals=expected,max_bytes=config['max_group_bytes'])
    for key in ('events','lams','n_events','n_sent','n_skipped_wl'):
        if rebuilt[key]!=out[key]: raise ValueError('baseline record reconstruction differs: '+key)
    for key in ('qframes','peaks'):
        if len(rebuilt[key])!=len(out[key]) or any(
                not np.array_equal(a,b) for a,b in zip(rebuilt[key],out[key])):
            raise ValueError('baseline array reconstruction differs: '+key)
    root=Path(__file__).resolve().parent
    env=Env(config['glint_prelude'],pythonpath=(str(root),str(root.parents[1])),inherit_pythonpath=False)
    worker=worker_type(audit)(env,timeout=config['activation_timeout'],call_timeout=config['call_timeout'])
    try:
        worker.start();child=worker.call('psana_parity_worker:initialize',parent,config)
        indexed=worker.call('psana_parity_worker:index_legacy',out,config,str(output/f'group-{group}.stream'))
        worker.close()
        if worker._proc.returncode!=0: raise RuntimeError('baseline child exit failed')
        atomic_json(output/'baseline-complete.json',dict(group=group,parent=parent,child=child,
            child_reaped=True,indexed=indexed,counters={k:out[k] for k in ('n_events','n_sent','n_skipped_wl')}))
    finally:
        worker.close(kill=worker._proc is not None and worker._proc.poll() is None)


def native(config,config_path,output,audit):
    import psana_mpi_producer as pipeline
    original_produce,original_consume,original_finalize=pipeline.produce,pipeline.consume,pipeline.finalize
    def produce(c,o,comm,dsf,pf):
        def factory(*args,**kwargs):
            class Processor(Proxy):
                def process_event(self,*a,**kw):
                    with audit.timed('event_processing'): return self.target.process_event(*a,**kw)
            return Processor(pf(*args,**kwargs))
        with audit.timed('production'):
            return original_produce(c,o,comm,audit.factory(dsf),factory)
    def consume(*args):
        with audit.timed('consumption'):
            return original_consume(*args,worker_factory=worker_type(audit))
    def finalize(*args):
        with audit.timed('merge'): return original_finalize(*args)
    with patch.object(pipeline,'produce',produce),patch.object(pipeline,'consume',consume),patch.object(pipeline,'finalize',finalize):
        code=pipeline.main(['--config',str(config_path),'--output',str(output)])
    if code: raise RuntimeError('native validation failed')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('baseline','native'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group',type=int,default=0)
    args=parser.parse_args(argv)
    config=load_config(args.config);output=args.output.resolve()
    # The coordinator creates native output once, before MPI. Baselines own one new directory.
    if args.mode=='baseline': output.mkdir(parents=True,exist_ok=False)
    os.environ['PS_PARALLEL']='none' if args.mode=='baseline' else 'mpi'
    os.environ.update(PS_EB_NODES='1',PS_EB_NODE_LOCAL='0',PS_SRV_NODES='0')
    os.environ['GLINT_PSANA_PROBE_ID']=str(output)
    audit=Audit(config['processor']['det'])
    original_flush=RecordWriter.flush
    def flush(writer):
        with audit.timed('spool_flush'): return original_flush(writer)
    with audit.bulk_reads(),patch.object(RecordWriter,'flush',flush):
        if args.mode=='baseline':
            baseline(config,output,args.group,audit);rank=args.group
        else:
            native(config,args.config,output,audit)
            from mpi4py import MPI
            rank=MPI.COMM_WORLD.Get_rank()
    atomic_json(output/f'audit-{rank}.json',audit.report())


if __name__=='__main__': main()

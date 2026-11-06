"""One native psana MPI application: produce records, then index complete groups."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys

from envbridge import Env, Worker
from glint_probe_common import atomic_json, identity, gpu_identity
from psana_cpu_config import load_config
from psana_reference import Reference, calibration_digest, event_timestamp
from qspool import RecordWriter, validate_spools, load_group, records_digest, file_digest


def placement(records, devices):
    if len(records)!=4 or sorted(r['rank'] for r in records)!=list(range(4)):
        raise ValueError('CPU path requires four MPI ranks')
    if len({r['host'] for r in records})!=1:
        raise ValueError('initial CPU path requires one node')
    if sorted(r['role'] for r in records)!=['bd','bd','eb','smd0']:
        raise ValueError('expected actual psana roles: smd0, eb, bd, bd')
    bds=sorted(r['rank'] for r in records if r['role']=='bd')
    if set(devices)!={str(r) for r in bds}:
        raise ValueError('device map does not match actual psana BD roles')
    return bds


def release_gpu_memory():
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def produce(config, output, comm, datasource_factory, processor_factory, gpu_fn=gpu_identity,
            release_gpu=release_gpu_memory):
    """Exhaust the native lifecycle; return only after run resources are released.

    Injectable factories are for tests, never accepted from a user config.
    Fatal exceptions go directly to main's MPI Abort, not collective cleanup.
    """
    reference=Reference(config['reference'])
    ds=datasource_factory(**config['source'])
    if not ds.is_mpi():
        raise ValueError('DataSource must be native MPI')
    parent=dict(identity(),rank=comm.Get_rank(),role=ds.comms.node_type(),gpu=None)
    parents=comm.allgather(parent)
    bds=placement(parents,config['devices'])
    if parent['role']=='bd':
        parent['gpu']=gpu_fn(config['devices'][str(parent['rank'])])
    parents=comm.allgather(parent)
    if len({p['gpu']['pci'] for p in parents if p['role']=='bd'})!=2:
        raise ValueError('BDs must use distinct physical GPUs')
    atomic_json(output/f'parent-{parent["rank"]}.json',parent)
    writer=None; audit=[]; seen=set(); runs=0
    for run in ds.runs():
        runs+=1
        if runs!=1 or run.expt!=config['source']['exp'] or run.runnum!=config['source']['run']:
            raise ValueError('expected exactly the configured offline run')
        processor=None
        if parent['role']=='bd':
            actual=calibration_digest(run.Detector(config['processor']['det']).calibconst)
            if actual!=reference.value['calibration_digest']:
                raise ValueError('calibration values differ from frozen reference')
            processor=processor_factory(run,experiment=run.expt,run_number=run.runnum,
                producer_rank=parent['rank'],num_groups=config['num_groups'],
                reference_digest=reference.digest,calibration_digest=actual,**config['processor'])
            if config['processor']['geom'] and processor.geometry_digest!=config['geometry_digest']:
                raise ValueError('geometry changed before processing')
            writer=RecordWriter(output/f'producer-{parent["rank"]}',parent['rank'],
                                processor.config_digest,config['max_chunk_bytes'])
        for evt in run.events():
            if processor is None:
                raise RuntimeError('control role received a user event')
            stamp=event_timestamp(evt)
            ordinal=reference.ordinal(stamp)
            if ordinal is None:
                audit.append(stamp)
                if len(audit)>4*config['source']['max_events']:
                    raise ValueError('out-of-range audit exceeds bounded delivery budget')
                continue
            if ordinal in seen:
                raise ValueError(f'duplicate local event {stamp}')
            seen.add(ordinal)
            writer.append(processor.process_event(evt,source_ordinal=ordinal))
        evt=None  # do not retain the last event's run context/buffers across teardown
        processor=None  # release geometry/finder scratch before run shared-memory cleanup
        # Current psana has collective shared-memory teardown; older installed
        # releases have no shared windows and expose no cleanup hook.
        close=getattr(run,'close_shared_memory',None)
        if close is not None:
            close()
        close=None
    if runs!=1:
        raise ValueError('DataSource yielded no configured run')
    del run, ds
    gc.collect()
    if parent['role']=='bd':
        release_gpu()
    if parent['role']!='bd' and ('cupy' in sys.modules or 'torch' in sys.modules):
        raise RuntimeError('control role imported an analysis runtime')
    summary=writer.finish(audit) if writer else None
    return parent,bds,summary


def consume(config, output, parent, bds, summaries, gate, worker_factory=Worker):
    if parent['role']!='bd':
        return dict(parent,child=None,child_reaped=True,groups=[])
    reference=Reference(config['reference'])
    root=Path(__file__).resolve().parent
    os.environ['GLINT_PSANA_PROBE_ID']=str(output)
    env=Env(config['glint_prelude'],label='glint-psana-cpu',
            pythonpath=(str(root),str(root.parents[1])),inherit_pythonpath=False)
    worker=worker_factory(env,timeout=config['activation_timeout'],call_timeout=config['call_timeout'])
    answers=[]
    try:
        worker.start()
        child=worker.call('glint_qconsumer:initialize',parent,config)
        atomic_json(output/f'child-{parent["rank"]}.json',child)
        for group in range(config['num_groups']):
            if bds[group % len(bds)]!=parent['rank']:
                continue
            records=load_group(output,summaries,reference,config,group)
            digest=records_digest(records)
            answer=worker.call('glint_qconsumer:index_group',records,config,group,
                               gate['config_digest'],str(output/f'group-{group}.stream'))
            if (answer['digest']!=digest or answer['pid']!=child['pid'] or
                    answer['group']!=group or answer['config_digest']!=gate['config_digest'] or
                    answer['n_frames']!=sum(r['status']=='hit' for r in records)):
                raise ValueError('child acknowledgement differs from assigned group')
            answers.append(answer)
            del records
        worker.close()
        if worker._proc.returncode!=0:
            raise RuntimeError('GLINT child did not exit cleanly')
        return dict(parent,child=child,child_reaped=True,groups=answers)
    finally:
        worker.close(kill=worker._proc is not None and worker._proc.poll() is None)


def finalize(output, reports, gate, num_groups):
    groups={}
    bds=sorted(r['rank'] for r in reports if r['role']=='bd')
    for report in reports:
        if not report['child_reaped'] or (report['role']=='bd')!=(report['child'] is not None):
            raise ValueError('incomplete child lifecycle report')
        for result in report['groups']:
            group=result['group']
            if group in groups or group not in range(num_groups) or bds[group % len(bds)]!=report['rank']:
                raise ValueError('duplicate/unexpected output group')
            path=output/f'group-{group}.stream'
            if (result['output']!=str(path) or result['config_digest']!=gate['config_digest'] or
                    not path.is_file() or path.stat().st_size!=result['output_bytes'] or
                    file_digest(path)!=result['sha256']):
                raise ValueError('missing or changed group output')
            groups[group]=path
    if set(groups)!=set(range(num_groups)):
        raise ValueError('missing group output')
    # This module imports only stdlib at top level; no legacy reader/placement runs.
    from glint_xtc_mpi import concat_streams
    temporary=output/'merged.stream.tmp'
    count=concat_streams([groups[g] for g in range(num_groups)],temporary)
    if not temporary.stat().st_size:
        raise ValueError('empty merged stream')
    destination=output/'merged.stream'
    os.replace(temporary,destination)
    return dict(output=str(destination),sha256=file_digest(destination),chunks=count)


def interrupted(signum, _frame):
    raise InterruptedError(f'rank interrupted by signal {signum}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    # Set before psana import. There is one MPI world, with no service ranks.
    os.environ.update(PS_PARALLEL='mpi',PS_EB_NODES='1',PS_EB_NODE_LOCAL='0',PS_SRV_NODES='0')
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    output=args.output.resolve()
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,interrupted)
    try:
        config=load_config(args.config)
        Reference(config['reference'])
        if comm.Get_size()!=4:
            raise ValueError('native CPU path requires four MPI ranks')
        from psana import DataSource
        from xtc_cpu_processor import CpuEventProcessor
        parent,bds,summary=produce(config,output,comm,DataSource,CpuEventProcessor)
        summaries=[s for s in comm.allgather(summary) if s is not None]
        gate=None
        if comm.Get_rank()==0:
            if sorted(s['rank'] for s in summaries)!=bds:
                raise ValueError('missing producer manifest')
            gate=validate_spools(output,summaries,Reference(config['reference']),config)
            atomic_json(output/'production-complete.json',gate)
        gate=comm.bcast(gate,root=0)
        report=consume(config,output,parent,bds,summaries,gate)
        reports=comm.gather(report,root=0)
        if comm.Get_rank()==0:
            final=finalize(output,reports,gate,config['num_groups'])
            atomic_json(output/'mpi-complete.json',dict(status='complete',mode='cpu',harness=False,
                        ranks=reports,production=gate,merged=final,
                        mpi_library=MPI.Get_library_version().rstrip('\x00')))
        comm.Barrier()
        return 0
    except BaseException as exc:
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,signal.SIG_IGN)
        try:
            atomic_json(output/f'rank-{comm.Get_rank()}-failed.json',
                        dict(rank=comm.Get_rank(),error=f'{type(exc).__name__}: {exc}'))
        finally:
            print(f'glint-psana CPU rank {comm.Get_rank()}: {exc}',file=sys.stderr,flush=True)
            comm.Abort(1)
        return 1


if __name__=='__main__':
    raise SystemExit(main())

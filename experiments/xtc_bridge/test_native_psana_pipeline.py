"""CPU tests for native-path orchestration; real MPI/GPU smoke is separate."""
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np
import qrecords as qr
import qspool
import psana_mpi_producer as pipeline
import glint_qconsumer as consumer
from psana_reference import Reference, calibration_digest, event_timestamp
from psana_cpu_config import load_config
from glint_psana import launch_command, main as launcher_main

CONSTANTS={'geometry':('geometry',{}),'pixel_status':(np.zeros((1,2,2),np.uint16),{})}


def reference():
    return dict(schema_version=1,experiment='test',run=1,
                calibration_digest=calibration_digest(CONSTANTS),timestamps=[100,110,120,130])


def config():
    return dict(psana_prelude='true',glint_prelude='true',expected_child_prefix='/other',
                reference=reference(),source=dict(exp='test',run=1,dir='/tmp',max_events=4,batch_size=1),
                processor=dict(zdist=.1,wavelength=1.3,det='jungfrau',geom=None),
                devices={'2':0,'3':1},num_groups=3,max_chunk_bytes=1024,max_group_bytes=10000,
                activation_timeout=1,call_timeout=1,job_timeout=30,cell=None,nbest=1,
                geometry_digest=None)


def record(i,rank=2,status='hit',timestamp=None):
    n=1 if status=='hit' else 0
    return dict(schema_version=1,config_digest='a'*64,experiment='test',run=1,
                timestamp=100+i*10 if timestamp is None else timestamp,source_ordinal=i,
                group_id=i%3,num_groups=3,producer_rank=rank,status=status,
                q=np.ones((n,3),np.float64),peaks=np.ones((n,3),np.float64),
                wavelength=1.3 if status in ('hit','below_min_peaks') else None,
                calibration_attempts=1,peak_count=n)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.c=config()

    def spool(self, rows, rank=2, audit=None):
        w=qspool.RecordWriter(self.root/f'producer-{rank}',rank,'a'*64,self.c['max_chunk_bytes'])
        for r in rows:
            w.append(r)
        return w.finish(audit or [])

    def test_reference_exact_mapping_and_rejection(self):
        r=Reference(reference())
        self.assertEqual(r.ordinal(120),2)
        with self.assertRaisesRegex(ValueError,'precedes frozen prefix'): r.ordinal(99)
        self.assertIsNone(r.ordinal(131))
        with self.assertRaises(ValueError): r.ordinal(115)
        for stamps in ([100,100],[110,100],[],[100,1.5]):
            with self.assertRaises(ValueError): Reference(dict(reference(),timestamps=stamps))
        with self.assertRaises(ValueError): r.check_record(record(0,timestamp=110))
        for v in (True,1.5,-1,2**64):
            with self.assertRaises(ValueError): event_timestamp(SimpleNamespace(timestamp=v))
        self.assertEqual(event_timestamp(SimpleNamespace(timestamp=np.uint64(2**64-1))),2**64-1)

    def test_calibration_digest_values_and_metadata(self):
        old=calibration_digest(CONSTANTS)
        altered=copy.deepcopy(CONSTANTS)
        altered['geometry'][1]['database']='irrelevant'
        self.assertEqual(calibration_digest(altered),old)
        altered['pixel_status'][0][0,0,0]=1
        self.assertNotEqual(calibration_digest(altered),old)

    def test_config_snapshot_and_guards(self):
        p=self.root/'config.json'
        p.write_text(json.dumps(self.c))
        loaded=load_config(p)
        self.assertEqual(loaded['reference'],reference())
        self.assertNotIn('son_min',loaded['processor'])  # use shared adapter default
        for field,value in [('integrate',False),('job_timeout',0),('num_groups',True),('devices',{'2':0})]:
            p.write_text(json.dumps(dict(self.c,**{field:value})))
            with self.assertRaises(ValueError): load_config(p)
        for processor in (dict(zdist=.1,peakfinder='pf8'),dict(zdist=.1,integrate=True),dict(zdist=-1)):
            p.write_text(json.dumps(dict(self.c,processor=processor)))
            with self.assertRaises(ValueError): load_config(p)
        command=launch_command(self.c,p,self.root,['mpirun','-n','4'],mode='cpu')
        self.assertIn('psana_mpi_producer.py',command[-1])
        self.assertNotIn('probe_psana_glint.py',command[-1])

    def test_chunk_roundtrip_rollover_and_empty_producer(self):
        self.c['max_chunk_bytes']=qr.record_nbytes(record(0))+1
        s=self.spool([record(0),record(1,status='missing_calibration'),record(2)])
        self.assertGreater(len(s['chunks']),1)
        rows=list(qspool.iter_records(self.root,s,self.c['max_chunk_bytes']))
        self.assertEqual([r['timestamp'] for r in rows],[100,110,120])
        self.assertEqual(rows[1]['q'].shape,(0,3))
        self.assertEqual(qspool.records_digest(rows),qspool.records_digest(list(reversed(rows))))
        idle=self.spool([],rank=3)
        self.assertEqual(list(qspool.iter_records(self.root,idle,self.c['max_chunk_bytes'])),[])

    def test_chunk_corruption_and_oversize(self):
        s=self.spool([record(0)])
        path=self.root/'producer-2'/s['chunks'][0]['name']
        with path.open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'digest'): list(qspool.iter_records(self.root,s,1024))
        w=qspool.RecordWriter(self.root/'tiny',2,'a'*64,1)
        with self.assertRaisesRegex(ValueError,'one event'): w.append(record(0))

    def test_completion_gate_and_groups_independent_of_bd_count(self):
        a=self.spool([record(0),record(2,status='invalid_wavelength')],audit=[141])
        b=self.spool([record(3,3),record(1,3,status='missing_calibration')],rank=3,audit=[140])
        gate=qspool.validate_spools(self.root,[a,b],Reference(reference()),self.c)
        self.assertEqual(gate['count'],4)
        self.assertEqual(gate['statuses']['hit'],2)
        self.assertEqual(len(gate['out_of_range']),2)
        rows=qspool.load_group(self.root,[a,b],Reference(reference()),self.c,0)
        self.assertEqual([r['source_ordinal'] for r in rows],[0,3])
        out=qr.to_legacy(rows,experiment='test',run=1,config_digest='a'*64,group_id=0,
                         num_groups=3,expected_ordinals=[0,3],max_bytes=10000)
        self.assertEqual(out['events'],[0,3])
        with self.assertRaisesRegex(ValueError,'max_group_bytes'):
            qspool.validate_spools(self.root,[a,b],Reference(reference()),dict(self.c,max_group_bytes=1))
        bad=copy.deepcopy(a);bad['out_of_range']=[99]
        with self.assertRaisesRegex(ValueError,'precedes frozen prefix'):
            qspool.validate_spools(self.root,[bad,b],Reference(reference()),self.c)
        bad=copy.deepcopy(a);bad['out_of_range']=[110]
        with self.assertRaisesRegex(ValueError,'incorrectly excluded'):
            qspool.validate_spools(self.root,[bad,b],Reference(reference()),self.c)

    def test_completion_rejects_duplicates_missing_and_wrong_mapping(self):
        a=self.spool([record(0),record(1,status='missing_calibration')])
        b=self.spool([record(0,3),record(2,3)],rank=3)
        with self.assertRaisesRegex(ValueError,'duplicate event ownership'):
            qspool.validate_spools(self.root,[a,b],Reference(reference()),self.c)
        with self.assertRaisesRegex(ValueError,'missing expected'):
            qspool.validate_spools(self.root,[a],Reference(reference()),self.c)
        with patch('qspool.iter_records',return_value=iter([record(0,timestamp=110)])):
            with self.assertRaisesRegex(ValueError,'timestamp/ordinal'):
                qspool.validate_spools(self.root,[a],Reference(reference()),self.c)

    def test_actual_role_placement(self):
        records=[dict(rank=i,host='node',role=r) for i,r in enumerate(['bd','smd0','bd','eb'])]
        self.assertEqual(pipeline.placement(records,{'0':0,'2':1}),[0,2])
        with self.assertRaisesRegex(ValueError,'actual psana BD'):
            pipeline.placement(records,{'2':0,'3':1})
        records[0]['host']='other'
        with self.assertRaisesRegex(ValueError,'one node'):
            pipeline.placement(records,{'0':0,'2':1})

    def fake_produce(self, rank, stamps, failing=False):
        order=[]
        class Run:
            expt='test';runnum=1
            def Detector(self,name): return SimpleNamespace(calibconst=CONSTANTS)
            def events(self):
                order.append('events-start')
                for stamp in stamps: yield SimpleNamespace(timestamp=stamp)
                order.append('events-exhausted')
            def close_shared_memory(self): order.append('run-closed')
        class DS:
            comms=SimpleNamespace(node_type=lambda: ['smd0','eb','bd','bd'][rank])
            def __init__(self,**kw): self.options=kw
            def is_mpi(self): return True
            def runs(self):
                yield Run()
                order.append('runs-exhausted')
        class Processor:
            config_digest='a'*64
            def __init__(self,*args,**kw): order.append('processor-created')
            def process_event(self,evt,source_ordinal):
                order.append(('process',evt.timestamp,source_ordinal))
                if failing: raise ValueError('nonfinite output')
                return record(source_ordinal,rank)
        class Comm:
            calls=0
            def Get_rank(self): return rank
            def allgather(self,value):
                self.calls+=1
                return [dict(value,rank=i,role=r,gpu=({'pci':str(i),'ordinal':i-2}
                         if r=='bd' and self.calls==2 else None))
                         for i,r in enumerate(['smd0','eb','bd','bd'])]
        def finish(writer,audit):
            self.assertEqual(order[-2:],['run-closed','runs-exhausted'])
            return original(writer,audit)
        original=qspool.RecordWriter.finish
        with patch.object(qspool.RecordWriter,'finish',finish):
            result=pipeline.produce(self.c,self.root,Comm(),DS,Processor,
                                    gpu_fn=lambda ordinal:dict(pci=str(ordinal),ordinal=ordinal),
                                    release_gpu=lambda:None)
        return result,order

    def test_native_lifecycle_ownership_and_out_of_range_without_calibration(self):
        (parent,bds,summary),order=self.fake_produce(2,[100,120,140])
        self.assertEqual([x for x in order if isinstance(x,tuple)],[('process',100,0),('process',120,2)])
        self.assertEqual(summary['out_of_range'],[140])
        self.assertEqual(summary['count'],2)
        self.assertEqual(bds,[2,3])

    def test_controls_never_create_processor_or_spool(self):
        (parent,bds,summary),order=self.fake_produce(0,[])
        self.assertIsNone(summary)
        self.assertNotIn('processor-created',order)
        self.assertEqual(order,['events-start','events-exhausted','run-closed','runs-exhausted'])

    def test_fatal_processing_and_duplicate_stop_without_success_manifest(self):
        with self.assertRaisesRegex(ValueError,'nonfinite output'):
            self.fake_produce(2,[100],failing=True)
        self.assertFalse((self.root/'producer-2/manifest.json').exists())
        with self.assertRaisesRegex(ValueError,'duplicate local'):
            self.fake_produce(3,[100,100])
        self.assertFalse((self.root/'producer-3/manifest.json').exists())

    def test_consumer_uses_existing_indexer_and_refuses_integration(self):
        self.c['num_groups']=3
        def index(out,args,path,report):
            self.assertFalse(args.integrate)
            self.assertEqual(out['events'],[0,3])
            Path(path).write_text('CrystFEL stream format 2.3\n')
            return None,None,0
        import glint_xtc
        with patch.object(consumer,'_ready',dict(gpu={'ordinal':0})),patch.object(glint_xtc,'index_and_write',side_effect=index):
            result=consumer.index_group([record(3,3),record(0)],self.c,0,'a'*64,str(self.root/'group-0.stream'))
            self.assertEqual(result['n_frames'],2)
            self.assertEqual(result['digest'],qspool.records_digest([record(0),record(3,3)]))
            with self.assertRaisesRegex(ValueError,'integration'):
                consumer.index_group([],dict(self.c,integrate=True),0,'a'*64,str(self.root/'bad'))
            with self.assertRaisesRegex(ValueError,'incomplete group'):
                consumer.index_group([record(0)],self.c,0,'a'*64,str(self.root/'bad'))

    def test_child_reaping_on_success_failure_and_idle_assignment(self):
        summaries=[self.spool([record(0),record(2)]),
                   self.spool([record(1,3),record(3,3)],rank=3)]
        calls=[]
        class Worker:
            def __init__(self,*args,**kwargs):
                self._proc=SimpleNamespace(returncode=None)
                self._proc.poll=lambda:self._proc.returncode
            def start(self): calls.append('start')
            def call(self,fn,*args):
                calls.append(fn)
                if fn.endswith(':initialize'): return dict(pid=55)
                records,config,group,digest,output=args
                return dict(pid=55,group=group,config_digest=digest,
                            digest=qspool.records_digest(records),
                            n_frames=sum(r['status']=='hit' for r in records))
            def close(self,kill=False):
                calls.append(('close',kill))
                self._proc.returncode=0
        parent=dict(rank=3,role='bd')
        report=pipeline.consume(self.c,self.root,parent,[2,3],summaries,
                                {'config_digest':'a'*64},worker_factory=Worker)
        self.assertTrue(report['child_reaped'])
        self.assertEqual([g['group'] for g in report['groups']],[1])
        calls.clear()
        idle=pipeline.consume(dict(self.c,num_groups=1),self.root,parent,[2,3],summaries,
                              {'config_digest':'a'*64},worker_factory=Worker)
        self.assertEqual(idle['groups'],[])
        self.assertEqual(calls.count('glint_qconsumer:initialize'),1)
        self.assertNotIn('glint_qconsumer:index_group',calls)
        class Failed(Worker):
            def call(self,fn,*args):
                if fn.endswith(':index_group'): raise RuntimeError('child failed')
                return super().call(fn,*args)
        calls.clear()
        with self.assertRaisesRegex(RuntimeError,'child failed'):
            pipeline.consume(self.c,self.root,parent,[2,3],summaries,
                             {'config_digest':'a'*64},worker_factory=Failed)
        self.assertIn(('close',True),calls)
        with patch('psana_mpi_producer.Worker',side_effect=AssertionError('control spawned')):
            report=pipeline.consume(self.c,self.root,dict(rank=0,role='smd0'),[2,3],summaries,{})
            self.assertIsNone(report['child'])

    def test_consumer_empty_group_and_nonfinite_failure(self):
        import glint_xtc
        c=dict(self.c,num_groups=5)
        def empty(out,args,path,report):
            self.assertEqual(out['events'],[])
            Path(path).write_text('CrystFEL stream format 2.3\n')
            return None,None,0
        with patch.object(consumer,'_ready',dict(gpu={'ordinal':0})),patch.object(glint_xtc,'index_and_write',side_effect=empty):
            result=consumer.index_group([],c,4,'a'*64,str(self.root/'empty.stream'))
            self.assertEqual(result['n_frames'],0)
            bad=record(0);bad['peaks'][0,2]=np.inf
            with self.assertRaisesRegex(ValueError,'finite'):
                consumer.index_group([bad,record(3)],self.c,0,'a'*64,str(self.root/'bad.stream'))

    def test_output_merge_requires_every_file_and_hash(self):
        reports=[dict(rank=i,role=r,child=None if i<2 else {},child_reaped=True,groups=[])
                 for i,r in enumerate(['smd0','eb','bd','bd'])]
        for g in range(3):
            path=self.root/f'group-{g}.stream'
            path.write_text('CrystFEL stream format 2.3\n'+('----- Begin chunk -----\nx\n----- End chunk -----\n' if g==1 else ''))
            reports[2+g%2]['groups'].append(dict(group=g,output=str(path),config_digest='a'*64,
                output_bytes=path.stat().st_size,sha256=qspool.file_digest(path),n_frames=int(g==1)))
        merged=pipeline.finalize(self.root,reports,{'config_digest':'a'*64},3)
        self.assertEqual(merged['chunks'],1)
        self.assertEqual((self.root/'merged.stream').read_text().count('CrystFEL stream'),1)
        with patch('glint_xtc_mpi.concat_streams',return_value=0):
            with self.assertRaisesRegex(ValueError,'chunk count'):
                pipeline.finalize(self.root,reports,{'config_digest':'a'*64},3)
        (self.root/'group-2.stream').unlink()
        with self.assertRaisesRegex(ValueError,'missing or changed'):
            pipeline.finalize(self.root,reports,{'config_digest':'a'*64},3)


    def test_outer_cpu_completion_checks_merged_output(self):
        config_path=self.root/'config.json'
        config_path.write_text(json.dumps(self.c))
        for fault in (None,'size','digest','mode','missing'):
            output=self.root/str(fault)
            def started(*args,**kwargs):
                stream=output/'merged.stream'
                stream.write_text('CrystFEL stream format 2.3\n')
                merged=dict(output=str(stream),output_bytes=stream.stat().st_size,
                            sha256=qspool.file_digest(stream))
                if fault=='size': merged['output_bytes']+=1
                if fault=='digest': merged['sha256']='0'*64
                if fault=='missing': stream.unlink()
                (output/'mpi-complete.json').write_text(json.dumps(dict(
                    status='complete',ranks=[{}]*4,mode='probe' if fault=='mode' else 'cpu',
                    harness=False,merged=merged)))
                return Mock(wait=Mock(return_value=0))
            with patch('glint_psana.subprocess.Popen',side_effect=started), \
                 patch('glint_psana.stop_launcher'),patch.dict(os.environ,{},clear=True):
                code=launcher_main(['--mode','cpu','--config',str(config_path),
                                    '--output',str(output),'--','true'])
            self.assertEqual(code,0 if fault is None else 1)
            self.assertEqual((output/'complete.json').exists(),fault is None)
            self.assertEqual((output/'failed.json').exists(),fault is not None)

    def test_main_completion_and_abort_artifacts(self):
        config_path=self.root/'config.json'
        config_path.write_text(json.dumps(self.c))
        for fail in (False,True):
            output=self.root/str(fail);output.mkdir()
            comm=Mock()
            comm.Get_rank.return_value=0;comm.Get_size.return_value=4
            comm.allgather.return_value=[None,None,{'rank':2},{'rank':3}]
            comm.gather.return_value=[{'rank':i} for i in range(4)]
            comm.bcast.side_effect=lambda value,root:value
            mpi=SimpleNamespace(COMM_WORLD=comm,Get_library_version=lambda:'test MPI')
            gate=dict(count=4,config_digest='a'*64)
            with patch.dict(sys.modules,{'mpi4py':SimpleNamespace(MPI=mpi),
                                         'psana':SimpleNamespace(DataSource=Mock())}), \
                 patch.object(pipeline.signal,'signal'), \
                 patch.object(pipeline,'produce',return_value=({'rank':0},[2,3],None)), \
                 patch.object(pipeline,'validate_spools',side_effect=ValueError('missing expected events') if fail else None,
                              return_value=gate), \
                 patch.object(pipeline,'consume') as consume, \
                 patch.object(pipeline,'finalize',return_value={'chunks':2}) as finalize, \
                 patch.dict(os.environ,{},clear=True):
                code=pipeline.main(['--config',str(config_path),'--output',str(output)])
            self.assertEqual(code,int(fail))
            self.assertEqual((output/'mpi-complete.json').exists(),not fail)
            if fail:
                comm.Abort.assert_called_once_with(1)
                consume.assert_not_called();finalize.assert_not_called()
                error=json.loads((output/'rank-0-failed.json').read_text())['error']
                self.assertIn('missing expected events',error)
                comm.bcast.assert_not_called();comm.Barrier.assert_not_called()
            else:
                comm.Abort.assert_not_called();comm.Barrier.assert_called_once()
                report=json.loads((output/'mpi-complete.json').read_text())
                self.assertEqual(report['production'],gate)
                self.assertFalse(report['harness'])


if __name__=='__main__':
    unittest.main(verbosity=2)

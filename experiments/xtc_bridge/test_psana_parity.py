"""CPU acceptance-checker tests, including corruption and non-vacuous orientation gates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from psana_parity_audit import Audit, frame_digest
from psana_parity_compare import (calibration_map, compare_records, compare_streams,
                                  equivalent_basis, read_overlap, science_config, compare)
from psana_reference import Reference, calibration_digest
from qrecords import validate_record


def record(i=0):
    return validate_record(dict(schema_version=1,config_digest='a'*64,experiment='exp',run=1,
        timestamp=2**63+i,source_ordinal=i,group_id=i%3,num_groups=3,producer_rank=2,
        status='hit',q=np.ones((20,3)),peaks=np.ones((20,3)),wavelength=1.3,
        calibration_attempts=1,peak_count=20))


class ParityTests(unittest.TestCase):
    def test_record_comparison_ignores_only_execution_owner_and_digest(self):
        a=[record(),record(1)]
        b=[dict(r,producer_rank=3,config_digest='b'*64) for r in reversed(a)]
        self.assertEqual(compare_records(a,b),0)
        for field,value in [('timestamp',2),('source_ordinal',3),('group_id',2),
                            ('status','below_min_peaks'),('wavelength',1.4),('calibration_attempts',2)]:
            altered=[dict(a[0],**{field:value}),a[1]]
            with self.subTest(field=field),self.assertRaises(ValueError): compare_records(a,altered)
        for field in ('q','peaks'):
            altered=copy.deepcopy(a);altered[0][field][0,1]+=.01
            with self.assertRaises(ValueError): compare_records(a,altered)
        with self.assertRaises(ValueError): compare_records(a,a[:-1])

    def test_calibration_hash_includes_shape_dtype_and_every_pixel(self):
        a=np.arange(12,dtype=np.float32).reshape(2,2,3)
        self.assertEqual(frame_digest(a),frame_digest(a.copy()))
        for b in (a.astype(np.float64),a.reshape(1,4,3),a+1):
            self.assertNotEqual(frame_digest(a)['sha256'],frame_digest(b)['sha256'])
        self.assertIsNone(frame_digest(None))

    def test_audit_wraps_supplied_events_without_extra_calibration(self):
        constants={'geometry':('test',{})};calls=[]
        def calib(evt): calls.append(evt.timestamp);return np.ones((1,2,3),np.float32)
        run=SimpleNamespace(Detector=lambda _:SimpleNamespace(calibconst=constants,
            raw=SimpleNamespace(calib=calib,_segment_numbers=[0])),
            events=lambda:iter([SimpleNamespace(timestamp=2**63)]))
        audit=Audit('jungfrau');proxy=audit.wrap_run(run)
        evt=next(proxy.events());frame=proxy.Detector('jungfrau').raw.calib(evt)
        self.assertEqual(calls,[2**63]);self.assertEqual(audit.seen,calls)
        self.assertEqual(audit.calibrations,[dict(timestamp=2**63,frame=frame_digest(frame))])
        self.assertEqual(audit.constants,[calibration_digest(constants)])
        self.assertEqual(audit.segments,[[0]])

    def test_calibration_map_rejects_missing_duplicate_and_segment_mismatch(self):
        ref=Reference(dict(schema_version=1,experiment='exp',run=1,calibration_digest='a'*64,
                           timestamps=[2**63,2**63+1]))
        rows=[dict(timestamp=t,frame=frame_digest(np.ones((1,2,3)))) for t in ref.ordinals]
        audit=dict(calibrations=rows,physical_segments=[[0]],calibration_digests=['a'*64])
        self.assertEqual(len(calibration_map([audit],ref)),2)
        for bad in (dict(audit,calibrations=rows[:-1]),dict(audit,calibrations=rows+[rows[0]]),
                    dict(audit,physical_segments=[[1]]),dict(audit,calibration_digests=['b'*64])):
            with self.assertRaises(ValueError): calibration_map([bad],ref)

    def test_oriented_lattice_equivalence_is_not_cell_shape_only(self):
        a=np.diag([79.,79.,38.])
        u=np.array([[1,1,0],[0,1,0],[0,0,-1]])
        self.assertTrue(equivalent_basis(a,a@u))
        theta=.1;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        self.assertFalse(equivalent_basis(a,r@a))
        self.assertFalse(equivalent_basis(a,a@np.diag([2,1,1])))
        self.assertFalse(equivalent_basis(a,np.zeros((3,3))))
        self.assertFalse(equivalent_basis(a,np.full((3,3),np.nan)))

    def test_byte_overlap_is_per_file_and_includes_duplicate_bulk_reads(self):
        rows=[dict(path='a',offset=0,returned=10),dict(path='a',offset=5,returned=10),
              dict(path='b',offset=0,returned=10),dict(path='a',offset=0,returned=10)]
        self.assertEqual(read_overlap(rows),dict(returned_bytes=40,unique_interval_bytes=25,overlap_bytes=15))

    def test_stream_comparison_checks_metadata_and_positive_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.stream',Path(tmp)/'b.stream'
            report=dict(frames=[dict(event=2,indexed=True,peak_count=20,matched_strict=20,strict_pass=True,M=np.diag([79.,79.,38.]).tolist())],
                strict_gate=dict(minimum=10,fraction=.25,tolerance=.15),seed=0,torch='test',numpy='test',matmul_precision='highest',tf32=False,
                xtc_python_open_guard=True,mpi_imported=False,psana_imported=False)
            for p in (a,b):
                p.write_text('frozen stream metadata and peaks\n')
                Path(str(p)+'.index.json').write_text(json.dumps(report))
            q_by_event={2:np.ones((20,3))}
            self.assertEqual(compare_streams(a,b,[2],q_by_event),dict(raw=[2],strict=[2]))
            weak=copy.deepcopy(report);weak['frames'][0].update(matched_strict=6,strict_pass=False)
            for path in (a,b): Path(str(path)+'.index.json').write_text(json.dumps(weak))
            weak_q=q_by_event[2].copy();weak_q[6:]=.123
            self.assertEqual(compare_streams(a,b,[2],{2:weak_q}),dict(raw=[2],strict=[]))
            for path in (a,b): Path(str(path)+'.index.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError): compare_streams(a,b,[1],q_by_event)
            for key,value in [('mpi_imported',True),('psana_imported',True),('seed',1),('tf32',True)]:
                Path(str(b)+'.index.json').write_text(json.dumps(dict(report,**{key:value})))
                with self.assertRaises(ValueError): compare_streams(a,b,[2],q_by_event)
            for key,value in [('matched_strict',19),('peak_count',21)]:
                bad=copy.deepcopy(report);bad['frames'][0][key]=value
                Path(str(b)+'.index.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError): compare_streams(a,b,[2],q_by_event)
            bad=copy.deepcopy(report);bad['strict_gate']['minimum']=1
            Path(str(b)+'.index.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError): compare_streams(a,b,[2],q_by_event)
            Path(str(b)+'.index.json').write_text(json.dumps(report))
            b.write_text('changed wavelength/peaks\n')
            with self.assertRaises(ValueError): compare_streams(a,b,[2],q_by_event)

    def test_scientific_configuration_excludes_only_execution_parameters(self):
        c=dict(reference={},processor={'zdist':.356},geometry_digest='a'*64,num_groups=3,cell=None,nbest=1,
            source=dict(exp='exp',run=1,dir='/tmp',max_events=13,batch_size=1),
            psana_prelude='psana',glint_prelude='glint',expected_child_prefix='/glint',devices={'2':0,'3':1})
        b=copy.deepcopy(c);b['source']['batch_size']=5;b['devices']={'2':1,'3':0}
        self.assertEqual(science_config(c),science_config(b))
        b['source']['max_events']=14;self.assertNotEqual(science_config(c),science_config(b))


    def test_worker_rows_use_full_q_and_reject_unusable_cells(self):
        from psana_parity_worker import index_rows
        # Use the CPU-only production cell predicate; no CUDA/Torch imports.
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from glint.stream import has_cell
        gate=dict(minimum=10,fraction=.25,tolerance=.15)
        def matcher(m,q):
            h=q@m
            return int((np.abs(h-np.rint(h)).max(axis=1)<.15).sum())
        good=np.ones((20,3));full=np.ones((100,3));full[20:]=.123
        results=[dict(event=0,M=None),dict(event=1,M=np.zeros((3,3))),
                 dict(event=2,M=np.diag([79.,79.,38.])),
                 dict(event=3,M=np.diag([79.,79.,38.]),q=good)]
        rows=index_rows(results,[0,1,2,3],[good,good,good,full],has_cell,matcher,gate)
        self.assertEqual([r['indexed'] for r in rows],[False,False,True,True])
        self.assertEqual([r['strict_pass'] for r in rows],[False,False,True,False])
        self.assertEqual(rows[3]['peak_count'],100);self.assertEqual(rows[3]['matched_strict'],20)
        self.assertEqual(index_rows(None,[],[],has_cell,matcher,gate),[])

    def test_matrix_gates(self):
        from validate_psana_parity import matrix_gates
        a=dict(owners={'0':2,'1':3},indexed_events=[1])
        b=dict(owners={'0':3,'1':2},indexed_events=[1])
        self.assertEqual(matrix_gates([a,b,b],203,1,[]),([2,2],True))
        self.assertFalse(matrix_gates([a,b,b],203,2,[])[1])
        empty=dict(b,indexed_events=[])
        self.assertFalse(matrix_gates([a,empty,b],203,1,[])[1])
        with self.assertRaisesRegex(ValueError,'partial'): matrix_gates([a,b,b],200,1,[])
        with self.assertRaisesRegex(ValueError,'ownership'): matrix_gates([a,a,a],203,1,[])
        with self.assertRaisesRegex(RuntimeError,'surviving'): matrix_gates([a,b,b],203,1,[123])

    def test_matrix_launcher_only_wraps_native_cases_and_is_recorded(self):
        import shlex
        import validate_psana_parity as matrix
        for launcher in ([], ['srun','--mpi=cray_shasta','-n','4','--gpu-bind=none']):
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); config=root/'config.json'; output=root/'output with spaces'
                config.write_text(json.dumps(dict(source=dict(exp='exp',run=1,dir=tmp,max_events=13),
                    processor=dict(det='jungfrau'),num_groups=3,devices={'2':0,'3':1},
                    psana_prelude='true')))
                cases=[dict(owners={'a':2,'b':3},indexed_events=[1]),
                       dict(owners={'a':3,'b':2},indexed_events=[1])]
                with patch.object(matrix.subprocess,'Popen') as popen, \
                     patch.object(matrix,'load_config',side_effect=lambda p:json.loads(p.read_text())), \
                     patch.object(matrix,'compare',side_effect=[cases[0],cases[1],cases[1]]), \
                     patch.object(matrix,'survivors',return_value=[]):
                    popen.return_value.wait.return_value=0
                    self.assertEqual(matrix.main(['--config',str(config),'--output',str(output),
                        '--revision','test']+(['--',*launcher] if launcher else [])),0)
                commands=json.loads((output/'commands.json').read_text())
                self.assertEqual([c['name'] for c in commands],['gpu-check','freeze',
                    'baseline-0','baseline-1','baseline-2','batch1','batch5','batch5-swapped'])
                self.assertEqual([c['argv'] for c in commands],
                                 [call.args[0] for call in popen.call_args_list])
                expected=launcher or ['mpirun','-n','4']
                for i,c in enumerate(commands):
                    argv=shlex.split(c['argv'][2].split(' && exec ',1)[1])
                    if i<5: self.assertEqual(argv[0],'python')
                    else:
                        self.assertEqual(argv[:len(expected)],expected)
                        self.assertEqual(argv[-2:],['--output',str(output/c['name'])])

    def test_full_artifact_comparison_and_corruption(self):
        self.artifact_fixture(False)

    def test_empty_hit_group_is_valid_parity_without_coverage(self):
        self.artifact_fixture(True)

    def artifact_fixture(self,empty):
        from qspool import RecordWriter,validate_spools,file_digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'baseline';native=root/'native';base.mkdir();native.mkdir()
            ref=dict(schema_version=1,experiment='exp',run=1,calibration_digest='a'*64,
                     timestamps=[2**63+i for i in range(4)])
            c=dict(reference=ref,processor={},geometry_digest=None,num_groups=3,cell=None,nbest=1,
                source=dict(exp='exp',run=1,dir='/tmp',max_events=4,batch_size=1),
                psana_prelude='psana',glint_prelude='glint',expected_child_prefix='/glint',
                max_chunk_bytes=10000,max_group_bytes=100000,devices={'2':0,'3':1})
            def write(path,value): path.write_text(json.dumps(value))
            for d in (base,native): write(d/'config.json',c)
            rows=[record(i) for i in range(4)]
            if empty:
                for r in rows:
                    r.update(status='below_min_peaks',peak_count=0,q=np.empty((0,3)),peaks=np.empty((0,3)))
            frame=frame_digest(np.ones((1,2,3)))
            def audit(selected,seen):
                return dict(seen=seen,calibrations=[dict(timestamp=r['timestamp'],frame=frame) for r in selected],
                    physical_segments=[[0]],calibration_digests=['a'*64],reads=[],seconds={},max_rss_kib=1)
            def stream(path,selected):
                selected=[r for r in selected if r['status']=='hit']
                path.write_text('frozen stream '+str([r['source_ordinal'] for r in selected]))
                write(Path(str(path)+'.index.json'),dict(
                    frames=[dict(event=r['source_ordinal'],indexed=True,peak_count=20,matched_strict=20,strict_pass=True,M=np.diag([79.,79.,38.]).tolist()) for r in selected],
                    strict_gate=dict(minimum=10,fraction=.25,tolerance=.15),seed=0,torch='test',numpy='test',matmul_precision='highest',tf32=False,
                    xtc_python_open_guard=True,mpi_imported=False,psana_imported=False))
            for g in range(3):
                d=base/f'group-{g}';d.mkdir();selected=[r for r in rows if r['group_id']==g]
                w=RecordWriter(d/f'producer-{g}',g,'a'*64,10000)
                for r in selected:w.append(dict(r,producer_rank=g))
                w.finish([]);stream(d/f'group-{g}.stream',selected)
                write(d/'baseline-complete.json',dict(group=g,child_reaped=True,indexed=dict(n_indexed=0 if empty else len(selected))))
                write(d/f'audit-{g}.json',audit(selected,ref['timestamps']))
            reports=[];summaries=[]
            for rank,role in enumerate(('smd0','eb','bd','bd')):
                selected=[] if rank<2 else [r for r in rows if r['source_ordinal']%2==rank-2]
                if rank>=2:
                    w=RecordWriter(native/f'producer-{rank}',rank,'a'*64,10000)
                    for r in selected:w.append(dict(r,producer_rank=rank))
                    summaries.append(w.finish([]))
                write(native/f'audit-{rank}.json',audit(selected,[r['timestamp'] for r in selected]))
                reports.append(dict(rank=rank,role=role,child=None if rank<2 else {},child_reaped=True,groups=[]))
            for g in range(3):
                path=native/f'group-{g}.stream';stream(path,[r for r in rows if r['group_id']==g])
                reports[2+g%2]['groups'].append(dict(group=g,n_indexed=0 if empty else sum(r['group_id']==g for r in rows),sha256=file_digest(path),output_bytes=path.stat().st_size))
            merged=native/'merged.stream';merged.write_text('merged fixture')
            gate=validate_spools(native,summaries,Reference(ref),c)
            write(native/'mpi-complete.json',dict(status='complete',harness=False,production=gate,ranks=reports,
                merged=dict(sha256=file_digest(merged),output_bytes=merged.stat().st_size)))
            result=compare(base,native)
            self.assertTrue(result['parity_pass']);self.assertEqual(result['indexed_events'],[] if empty else [0,1,2,3])
            self.assertEqual(result['calibration_attempts'],dict(baseline=4,native=4))
            def mutate(name,change,error,kind=ValueError):
                path=native/name;good=path.read_bytes()
                try:
                    if change is None: path.unlink()
                    elif callable(change):
                        bad=json.loads(good);change(bad);write(path,bad)
                    else: path.write_text(change)
                    with self.assertRaisesRegex(kind,error): compare(base,native)
                finally: path.write_bytes(good)
            mutate('audit-2.json',lambda x:x.update(calibrations=[]),'BD participation')
            mutate('audit-0.json',lambda x:x.update(seen=[2**63]),'delivery ownership|control role')
            mutate('audit-0.json',lambda x:x.update(seen=[2**63+99]),'control role')
            mutate('group-0.stream','tampered','group stream changed')
            mutate('merged.stream','tampered','merged stream changed')
            mutate('mpi-complete.json',lambda x:x['ranks'][2]['groups'][0].update(n_indexed=99),'writer indexed count')
            mutate('audit-3.json',None,'audit-3',FileNotFoundError)
            mutate('audit-2.json',lambda x:x['calibrations'].pop(),'missing calibration')
            mutate('audit-2.json',lambda x:x['calibrations'][0]['frame'].update(sha256='b'*64),'image diagnostics')
            self.assertTrue(compare(base,native)['parity_pass'])


if __name__=='__main__': unittest.main(verbosity=2)

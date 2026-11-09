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
                                  equivalent_basis, read_overlap, science_config)
from psana_reference import Reference, calibration_digest
from qrecords import validate_record


def record(i=0):
    return validate_record(dict(schema_version=1,config_digest='a'*64,experiment='exp',run=1,
        timestamp=2**63+i,source_ordinal=i,group_id=i%3,num_groups=3,producer_rank=2,
        status='hit',q=np.ones((2,3)),peaks=np.ones((2,3)),wavelength=1.3,
        calibration_attempts=1,peak_count=2))


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
            report=dict(frames=[dict(event=2,indexed=True,M=np.diag([79.,79.,38.]).tolist())],
                seed=0,torch='test',numpy='test',matmul_precision='highest',tf32=False,
                xtc_python_open_guard=True,mpi_imported=False,psana_imported=False)
            for p in (a,b):
                p.write_text('frozen stream metadata and peaks\n')
                Path(str(p)+'.index.json').write_text(json.dumps(report))
            self.assertEqual(compare_streams(a,b,[2]),[2])
            with self.assertRaises(ValueError): compare_streams(a,b,[1])
            for key,value in [('mpi_imported',True),('psana_imported',True),('seed',1),('tf32',True)]:
                Path(str(b)+'.index.json').write_text(json.dumps(dict(report,**{key:value})))
                with self.assertRaises(ValueError): compare_streams(a,b,[2])
            Path(str(b)+'.index.json').write_text(json.dumps(report))
            b.write_text('changed wavelength/peaks\n')
            with self.assertRaises(ValueError): compare_streams(a,b,[2])

    def test_scientific_configuration_excludes_only_execution_parameters(self):
        c=dict(reference={},processor={'zdist':.356},geometry_digest='a'*64,num_groups=3,cell=None,nbest=1,
            source=dict(exp='exp',run=1,dir='/tmp',max_events=13,batch_size=1),
            psana_prelude='psana',glint_prelude='glint',expected_child_prefix='/glint',devices={'2':0,'3':1})
        b=copy.deepcopy(c);b['source']['batch_size']=5;b['devices']={'2':1,'3':0}
        self.assertEqual(science_config(c),science_config(b))
        b['source']['max_events']=14;self.assertNotEqual(science_config(c),science_config(b))


if __name__=='__main__': unittest.main(verbosity=2)

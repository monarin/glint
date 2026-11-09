"""CPU checks for benchmark routing, measurement and paired comparisons."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from psana_ab_trial import legacy_mask
from psana_ab_measure import Metrics,fingerprint
from benchmark_psana_ab import trial_order,paired_summary,compare_outputs


class ABTests(unittest.TestCase):
    def test_order_and_summary_exclude_warmup(self):
        order=trial_order(3)
        self.assertEqual([x[1] for x in order],list('ABABBAAB'))
        trials=[dict(arm=a,pair=p,wall_seconds=(1000 if p is None else 10 if a=='A' else 5)) for _,a,p in order]
        self.assertEqual(paired_summary(trials),dict(pairs=3,median_A_seconds=10,median_B_seconds=5,
            paired_A_over_B=[2,2,2],median_paired_A_over_B=2))
        with self.assertRaises(ValueError):paired_summary(trials[:-1])

    def test_launcher_selects_only_allocated_mask(self):
        self.assertEqual(legacy_mask('GPU-a,GPU-b',1),'GPU-b')
        for mask,rank in [('0',0),('0,1,2',0),('0,1',2),('',0)]:
            with self.assertRaises(ValueError):legacy_mask(mask,rank)

    def test_proxy_calibrates_once_and_preserves_values(self):
        calls=[]
        def calib(evt):calls.append(evt);return np.ones((2,2))
        det=SimpleNamespace(raw=SimpleNamespace(calib=calib),calibconst={'a':1})
        run=SimpleNamespace(Detector=lambda name:det,events=lambda:iter([4]))
        ds=SimpleNamespace(runs=lambda:iter([run]),is_mpi=lambda:True)
        m=Metrics('det');wrapped=m.factory(lambda **kw:ds)()
        r=next(wrapped.runs());d=r.Detector('det');np.testing.assert_array_equal(d.raw.calib(4),np.ones((2,2)))
        self.assertEqual(calls,[4]);self.assertEqual(m.calib_calls,1)
        self.assertEqual(d.calibconst,det.calibconst);self.assertTrue(wrapped.is_mpi())
        self.assertIs(r.Detector('other'),det)

    def test_scalar_read_metrics_preserve_bytes(self):
        class EventManager:
            def _read(self,fd,size,offset):return b'abcd'
        fake=SimpleNamespace(EventManager=EventManager)
        core=SimpleNamespace(frame_q=lambda *a,**kw:('q','p'))
        with patch.dict('sys.modules',{'psana.psexp.event_manager':fake,'xtc_core':core}):
            m=Metrics('det')
            with m.processing():
                self.assertEqual(EventManager()._read(1,8,0),b'abcd')
                self.assertEqual(core.frame_q(None),('q','p'))
            self.assertEqual((m.read_calls,m.read_bytes),(1,4))
            self.assertEqual(set(m.seconds),{'bulk_read','peak_q'})

    def test_q_fingerprint_covers_events_wavelengths_and_arrays(self):
        a=dict(events=[1],lams=[1.3],qframes=[np.ones((2,3))],peaks=[np.ones((2,3))])
        self.assertEqual(fingerprint(a),fingerprint(copy.deepcopy(a)))
        for k in ('qframes','peaks'):
            b=copy.deepcopy(a);b[k][0][0,0]+=1;self.assertNotEqual(fingerprint(a),fingerprint(b))
        for k,v in [('events',[2]),('lams',[1.4])]:
            self.assertNotEqual(fingerprint(a),fingerprint(dict(a,**{k:v})))

    def test_pair_rejects_outputs_and_diagnostic_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'A',Path(tmp)/'B';a.mkdir();b.mkdir()
            for p in (a,b):(p/'merged.stream').write_text('merged')
            for g in range(2):
                row=dict(q_peaks_sha256='a'*64,events=[g],raw_indexed=0,strict_indexed=[],frames=[],
                    seed=0,strict_gate=dict(minimum=10,fraction=.25,tolerance=.15),torch='t',numpy='n',matmul_precision='highest',tf32=False)
                for p in (a/f'merged.stream.rank{g:04d}',b/f'group-{g}.stream'):
                    p.write_text(str(g));Path(str(p)+'.bench.json').write_text(json.dumps(row))
            self.assertEqual(compare_outputs(a,b),dict(hits=2,exact_stream_and_q=True))
            p=b/'group-0.stream.bench.json';row=json.loads(p.read_text());row['q_peaks_sha256']='b'*64;p.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'metadata'):compare_outputs(a,b)
            (b/'merged.stream').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'merged'):compare_outputs(a,b)


if __name__=='__main__':unittest.main(verbosity=2)

"""CPU adapter/record tests: fake psana and V4 detections, real geometry/q/slab math.

No GPU or data needed. This tests policy and plumbing, not the V4 CUDA kernel or
real psana calibration. Compare against the UNCHANGED serial reader, including
its counters, sampled intensities, wavelength skips and consensus sharding.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import qrecords as qr
import xtc_core
import xtc_qreader
from xtc_cpu_processor import CpuEventProcessor
from envbridge import _codec

SHAPE = (2, 4, 6)
X = np.arange(np.prod(SHAPE), dtype=float).reshape(SHAPE) * 100
Y = X + 200
X[1, 1, 1] = np.nan  # A detected peak in a geometry gap must be dropped.


class Geometry:
    loads = 0

    def load_pars_from_str(self, text):
        Geometry.loads += 1

    def get_pixel_coords(self):
        return X.copy(), Y.copy(), np.full(SHAPE, -100000.)


class Finder:
    setups = []

    def __init__(self, mask, **kw):
        self.setups.append((mask.copy(), kw))

    def find(self, frame):
        if frame[0, 0] < 0:
            return dict(x=np.empty(0), y=np.empty(0))
        # Fractional centroids exercise rint, clipping, gap removal and panel stacking.
        return dict(x=np.array([1.2, 2.6, 99.0]), y=np.array([1.1, 2.6, -2.0]))


class Run:
    def __init__(self):
        self.calls, self.energy_calls = [], []
        self.evts = [SimpleNamespace(timestamp=np.uint64(2**63 + i), index=i) for i in range(6)]
        self.frames = [np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE) + i * 100
                       for i in range(6)]
        self.frames[1] = None
        self.frames[3] = np.full(SHAPE, -1., dtype=np.float32)
        self.energies = [10000., 10000., np.inf, 11000., 12000., 1000.]
        status = np.zeros((2,) + SHAPE, dtype=np.uint16)
        status[1, 1, 2, 3] = 1
        self.det = SimpleNamespace(raw=SimpleNamespace(calib=self.calib),
                                   calibconst={"geometry": ["test geometry"], "pixel_status": [status]})
        self.energy = SimpleNamespace(raw=SimpleNamespace(ebeamPhotonEnergy=self.energy_value))

    def calib(self, evt):
        self.calls.append(evt.index)
        return self.frames[evt.index]

    def energy_value(self, evt):
        self.energy_calls.append(evt.index)
        return self.energies[evt.index]

    def Detector(self, name):
        return self.energy if name == "ebeamh" else self.det

    def events(self):
        return iter(self.evts)


def record(ordinal=0, status="hit", producer=2):
    count = 2 if status == "hit" else 0
    return dict(schema_version=1, config_digest="a" * 64, experiment="exp", run=1,
                timestamp=2**63 + ordinal, source_ordinal=ordinal, group_id=ordinal % 2,
                num_groups=2, producer_rank=producer, status=status,
                q=np.full((count, 3), .1), peaks=np.full((count, 3), 2.),
                wavelength=1.2 if status in ("hit", "below_min_peaks") else None,
                calibration_attempts=1, peak_count=count)


def legacy(records, ordinals, **kw):
    options = dict(experiment="exp", run=1, config_digest="a" * 64, group_id=0,
                   num_groups=2, expected_ordinals=ordinals, max_bytes=100000)
    options.update(kw)
    return qr.to_legacy(records, **options)


class RecordTests(unittest.TestCase):
    def test_imports_do_not_load_analysis_runtimes(self):
        subprocess.run([sys.executable, "-c", """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('psana', 'mpi4py', 'cupy', 'torch', 'glint'):
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
import qrecords, xtc_cpu_processor
"""], cwd=Path(__file__).resolve().parent, check=True)

    def test_validation(self):
        for status in qr.STATUSES:
            qr.validate_record(record(status=status))
        cases = dict(timestamp=float(2**63), source_ordinal=True, group_id=1, num_groups=0,
                     producer_rank=-1, schema_version=2, config_digest="bad", calibration_attempts=0,
                     peak_count=3, status="missing_calibration", wavelength=float("nan"),
                     q=np.ones((2, 3), dtype=np.float32), peaks=np.ones((1, 3)))
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                qr.validate_record(dict(record(), **{key: value}))
        for value in (-1, 2**64, np.uint64(2**63), True):
            with self.subTest(timestamp=value), self.assertRaises(ValueError):
                qr.validate_record(dict(record(), timestamp=value))
        for value in (np.full((2, 3), np.nan), np.ones((2, 6))[:, ::2]):
            with self.assertRaises(ValueError):
                qr.validate_record(dict(record(), q=value))

    def test_npz_roundtrip_all_statuses_exact_timestamp_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event.npz"
            for status in qr.STATUSES:
                r = record(status=status)
                r["timestamp"] = 2**64 - 1
                size = qr.record_nbytes(r)
                qr.write_record(path, r, max_bytes=size)
                restored = qr.read_record(path, max_bytes=size)
                self.assertEqual(restored["timestamp"], 2**64 - 1)
                self.assertEqual(restored["status"], status)
                np.testing.assert_array_equal(restored["q"], r["q"])
                np.testing.assert_array_equal(restored["peaks"], r["peaks"])
                with self.assertRaises(ValueError):
                    qr.write_record(path, r, max_bytes=size - 1)
                with self.assertRaises(ValueError):
                    qr.read_record(path, max_bytes=size - 1)
            np.savez(path, metadata=np.array([{}], dtype=object), q=r["q"], peaks=r["peaks"])
            with self.assertRaises(ValueError):
                qr.read_record(path)
            np.savez(path, unexpected=np.zeros(1))
            with self.assertRaises(ValueError):
                qr.read_record(path)

    def test_wire_roundtrip_uses_existing_codec(self):
        class MemorySocket:
            def __init__(self):
                self.buffer = io.BytesIO()
            def sendall(self, value):
                self.buffer.write(value)
            def recv(self, size):
                return self.buffer.read(size)
        sock = MemorySocket()
        original = record()
        _codec.send(sock, original)
        sock.buffer.seek(0)
        restored = qr.validate_record(_codec.recv(sock))
        self.assertEqual(restored["timestamp"], original["timestamp"])
        np.testing.assert_array_equal(restored["q"], original["q"])
        np.testing.assert_array_equal(restored["peaks"], original["peaks"])

    def test_group_order_skips_and_complete_accounting(self):
        rows = [record(0), record(2, "missing_calibration", 9),
                record(4, "invalid_wavelength"), record(6, "below_min_peaks", 9), record(8)]
        out = legacy(reversed(rows), range(0, 10, 2))
        self.assertEqual(out["events"], [0, 8])
        self.assertEqual((out["n_events"], out["n_sent"], out["n_skipped_wl"]), (3, 2, 1))
        self.assertEqual(legacy([], [])["n_sent"], 0)
        for bad, expected in ((rows[:-1], range(0, 10, 2)), (rows + [rows[0]], range(0, 10, 2)),
                              (rows, [0, 2, 4, 6]), (rows, [0, 0]), (rows, [1])):
            with self.assertRaises(ValueError):
                legacy(bad, expected)
        with self.assertRaises(ValueError):
            legacy([record(0), dict(record(2), timestamp=2**63)], [0, 2])
        for key, val in (("config_digest", "b" * 64), ("run", 2), ("experiment", "other")):
            with self.assertRaises(ValueError):
                legacy([dict(record(), **{key: val})], [0])
        with self.assertRaises(ValueError):
            legacy([record()], [0], max_bytes=1)

    def test_config_digest_is_order_independent_and_rejects_nan(self):
        self.assertEqual(qr.digest_config(dict(a=1, b=2)), qr.digest_config(dict(b=2, a=1)))
        with self.assertRaises(ValueError):
            qr.digest_config(dict(wavelength=float("nan")))


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.run = Run()
        self.datasource_calls = []
        mods = {name: ModuleType(name) for name in ("psana", "psana.pscalib",
                "psana.pscalib.geometry", "psana.pscalib.geometry.GeometryAccess", "cupy")}
        def datasource(**kwargs):
            self.datasource_calls.append(kwargs)
            return SimpleNamespace(runs=lambda: iter([self.run]))
        mods["psana"].DataSource = datasource
        mods["psana.pscalib.geometry.GeometryAccess"].GeometryAccess = Geometry
        mods["cupy"].asarray, mods["cupy"].asnumpy = np.asarray, np.asarray
        mods["cupy"].float32 = np.float32
        self.enterContextCompat(patch.dict(sys.modules, mods))
        self.enterContextCompat(patch.object(xtc_core, "load_peakfinder_v4",
                                             return_value=SimpleNamespace(PeakFinderV4=Finder)))
        self.enterContextCompat(patch.dict(os.environ, {"PS_PARALLEL": "mpi"}))
        Finder.setups.clear()
        Geometry.loads = 0

    def enterContextCompat(self, context):
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def processor(self, **kw):
        options = dict(experiment="exp", run_number=1, producer_rank=2, num_groups=3,
                       reference_digest="b" * 64, calibration_digest="c" * 64,
                       zdist=.1, min_peaks=2)
        options.update(kw)
        return CpuEventProcessor(self.run, **options)

    def compare(self, got, wanted):
        for key in ("events", "lams", "n_events", "n_sent", "n_skipped_wl"):
            self.assertEqual(got[key], wanted[key], key)
        for key in ("qframes", "peaks"):
            self.assertEqual(len(got[key]), len(wanted[key]))
            for a, b in zip(got[key], wanted[key]):
                np.testing.assert_array_equal(a, b)

    def test_matches_legacy_across_groups_and_producer_assignment(self):
        processors = [self.processor(producer_rank=4), self.processor(producer_rank=9)]
        rows = [processors[i % 2].process_event(evt, source_ordinal=i)
                for i, evt in enumerate(self.run.evts)]
        self.assertEqual(self.run.calls, list(range(6)))  # One calib even on invalid wavelength.
        self.assertEqual(self.run.energy_calls, [0, 2, 3, 4, 5])
        self.assertEqual([r["status"] for r in rows],
                         ["hit", "missing_calibration", "invalid_wavelength",
                          "below_min_peaks", "hit", "invalid_wavelength"])
        self.assertEqual(self.datasource_calls, [])
        self.assertEqual(os.environ["PS_PARALLEL"], "mpi")
        self.assertEqual(processors[0].config_digest, processors[1].config_digest)
        # Real prep_geometry constructs independent panel masks; flag in panel 1 only.
        self.assertTrue(Finder.setups[0][0][2, 3])
        self.assertFalse(Finder.setups[1][0][2, 3])
        self.assertEqual(Finder.setups[0][1]["thr_high"], xtc_core.PF_THR_HIGH)
        for group in range(3):
            got = qr.to_legacy([r for r in reversed(rows) if r["group_id"] == group],
                               experiment="exp", run=1, config_digest=processors[0].config_digest,
                               group_id=group, num_groups=3, expected_ordinals=range(group, 6, 3),
                               max_bytes=100000)
            wanted = xtc_qreader.run_to_qframes("exp", 1, zdist=.1, min_peaks=2, rank=group, nranks=3)
            self.compare(got, wanted)
        # Known rounded/clipped pixels, slab ordering and sampled intensities (not finder mass).
        np.testing.assert_array_equal(rows[0]["peaks"],
                                      [[1, 1, 7], [3, 3, 21], [5, 0, 5], [3, 7, 45], [5, 4, 29]])
        old = rows[0]["peaks"].copy()
        self.run.frames[0][:] = 12345
        np.testing.assert_array_equal(rows[0]["peaks"], old)

    def test_fixed_wavelength_geometry_override_and_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.geom"
            path.write_text("res = 10000\n" + "".join(
                f"p{i}/min_fs = 0\np{i}/max_fs = 5\np{i}/min_ss = {i*4}\np{i}/max_ss = {i*4+3}\n"
                f"p{i}/fs = +1x\np{i}/ss = +1y\np{i}/corner_x = {i*50}\np{i}/corner_y = 3\n"
                for i in range(2)))
            args = dict(wavelength=1.3, geom=str(path), min_pix=4, son_min=12, thr_high=9, thr_low=3)
            processor = self.processor(**args)
            rows = [processor.process_event(evt, source_ordinal=i) for i, evt in enumerate(self.run.evts)]
            self.assertEqual(Geometry.loads, 0)
            self.assertEqual(self.run.energy_calls, [])
            self.assertEqual(Finder.setups[0][1]["min_pix"], 4)
            self.assertEqual(Finder.setups[0][1]["son_min"], 12)
            for group in range(3):
                got = qr.to_legacy([r for r in rows if r["group_id"] == group], experiment="exp", run=1,
                                   config_digest=processor.config_digest, group_id=group, num_groups=3,
                                   expected_ordinals=range(group, 6, 3), max_bytes=100000)
                wanted = xtc_qreader.run_to_qframes("exp", 1, zdist=.1, min_peaks=2, rank=group,
                                                  nranks=3, **args)
                self.compare(got, wanted)
            waiting = self.processor(**args)
            path.write_text(path.read_text() + "# changed\n")
            with self.assertRaisesRegex(ValueError, "geometry file changed"):
                waiting.process_event(self.run.evts[0], source_ordinal=0)

    def test_guards_precede_calibration(self):
        for kw in (dict(peakfinder="pf8"), dict(zdist=0), dict(zdist=np.inf),
                   dict(wavelength=-1), dict(wavelength=np.nan), dict(min_peaks=True),
                   dict(reference_digest="bad"), dict(num_groups=0)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.processor(**kw)
        processor = self.processor()
        for ordinal in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                processor.process_event(self.run.evts[0], source_ordinal=ordinal)
        with self.assertRaises(ValueError):
            processor.process_event(SimpleNamespace(timestamp=1.5), source_ordinal=0)
        self.assertEqual(self.run.calls, [])

    def test_mask_fallback_geometry_once_and_shape_guard(self):
        del self.run.det.calibconst["pixel_status"]
        p = self.processor(wavelength=1.2)
        p.process_event(self.run.evts[0], source_ordinal=0)
        p.process_event(self.run.evts[4], source_ordinal=4)
        self.assertEqual(len(Finder.setups), 2)
        self.assertTrue(all(mask.all() for mask, _ in Finder.setups))
        self.run.frames[4] = np.zeros((1, 4, 6))
        with self.assertRaisesRegex(ValueError, "shape changed"):
            p.process_event(self.run.evts[4], source_ordinal=4)

    def test_digest_tracks_policy_and_provenance_but_not_rank(self):
        p = self.processor()
        for kw in (dict(thr_low=3), dict(wavelength=1.3), dict(num_groups=2),
                   dict(calibration_digest="d" * 64), dict(reference_digest="e" * 64)):
            self.assertNotEqual(p.config_digest, self.processor(**kw).config_digest)
        self.assertEqual(p.config_digest, self.processor(producer_rank=17).config_digest)

    def test_below_threshold_and_calibration_exception(self):
        p = self.processor(min_peaks=6)
        result = p.process_event(self.run.evts[0], source_ordinal=0)
        self.assertEqual(result["status"], "below_min_peaks")
        self.assertEqual(result["q"].shape, (0, 3))
        with patch.object(self.run.det.raw, "calib", side_effect=RuntimeError("calibration failed")) as calib:
            with self.assertRaisesRegex(RuntimeError, "calibration failed"):
                p.process_event(self.run.evts[0], source_ordinal=0)
            calib.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Opt-in CPU calibration/event adapter for native psana producers.

Consumes supplied events; never creates a DataSource, enumerates/shards events,
changes PS_PARALLEL, starts MPI or calls envbridge. V4 finding still uses CuPy.
The unchanged xtc_qreader is the reference for calibration/geometry/wavelength
policy. Stage 3 supplies reference ordinals and checks full event ownership.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import qrecords
import xtc_core


class CpuEventProcessor:
    def __init__(self, prun, *, experiment, run_number, producer_rank, num_groups,
                 reference_digest, calibration_digest, zdist, det="jungfrau",
                 wavelength=0.0, energy_det="ebeamh", min_peaks=6, peakfinder="v4",
                 min_pix=xtc_core.PF_MIN_PIX, son_min=xtc_core.PF_SON_MIN,
                 thr_high=xtc_core.PF_THR_HIGH, thr_low=xtc_core.PF_THR_LOW, geom=None):
        if peakfinder != "v4":
            raise ValueError("CPU reference path supports only peakfinder='v4'")
        for key, value, minimum in (("run_number", run_number, 0),
                                    ("producer_rank", producer_rank, 0),
                                    ("num_groups", num_groups, 1),
                                    ("min_peaks", min_peaks, 1), ("min_pix", min_pix, 1)):
            qrecords.integer(value, key, minimum)
        if not isinstance(experiment, str) or not experiment.strip():
            raise ValueError("experiment must be nonempty")
        for key, value in (("reference_digest", reference_digest),
                           ("calibration_digest", calibration_digest)):
            qrecords.validate_digest(value, key)
        for key, value, zero in (("zdist", zdist, False), ("wavelength", wavelength, True),
                                 ("son_min", son_min, True), ("thr_high", thr_high, True),
                                 ("thr_low", thr_low, True)):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not np.isfinite(value) or value < 0 or (not zero and value == 0)):
                raise ValueError(f"{key} must be finite and {'nonnegative' if zero else 'positive'}")
        self.detector = prun.Detector(det)
        self.edet = None
        if not wavelength:
            try:
                self.edet = prun.Detector(energy_det)
            except Exception as exc:
                raise ValueError(f"no wavelength and energy detector {energy_det!r} unavailable") from exc
        self.geom = str(Path(geom).resolve()) if geom else None
        if self.geom:
            geometry_bytes = Path(self.geom).read_bytes()
            self.coords = None
        else:
            from psana.pscalib.geometry.GeometryAccess import GeometryAccess
            geometry_text = self.detector.calibconst["geometry"][0]
            geometry_bytes = geometry_text.encode()
            geo = GeometryAccess()
            geo.load_pars_from_str(geometry_text)
            self.coords = geo.get_pixel_coords()
        self.geometry_digest = hashlib.sha256(geometry_bytes).hexdigest()
        self.zdist, self.wavelength, self.min_peaks = float(zdist), float(wavelength), min_peaks
        self.finder_options = dict(min_pix=min_pix, son_min=float(son_min),
                                   thr_high=float(thr_high), thr_low=float(thr_low), peakfinder="v4")
        # These externally supplied digests identify the frozen event manifest and
        # calibration provenance. They must be shared across producers, never PID/rank.
        science = dict(schema_version=qrecords.SCHEMA_VERSION, experiment=experiment,
                       run=run_number, detector=det, energy_det=energy_det,
                       zdist=self.zdist, wavelength=self.wavelength, min_peaks=min_peaks,
                       num_groups=num_groups, reference_digest=reference_digest,
                       calibration_digest=calibration_digest, geometry_digest=self.geometry_digest,
                       geometry_source="geom" if self.geom else "psana", **self.finder_options)
        self.config_digest = qrecords.digest_config(science)
        self.identity = dict(schema_version=qrecords.SCHEMA_VERSION, experiment=experiment,
                             run=run_number, producer_rank=producer_rank, num_groups=num_groups,
                             config_digest=self.config_digest)
        self.prepared = None
        self.shape = None

    def _wavelength(self, evt):
        if self.wavelength:
            return self.wavelength
        try:
            energy = self.edet.raw.ebeamPhotonEnergy(evt)
            if energy is not None and np.isfinite(energy) and float(energy) > 1000.0:
                return xtc_core.HC_EV_A / float(energy)
        except Exception:
            pass
        return None

    def _prepare(self, shape, lam):
        if len(shape) != 3:
            raise ValueError("calibration must have (segments, rows, columns) shape")
        if self.geom:
            # Do not silently use geometry changed after the config was hashed.
            if hashlib.sha256(Path(self.geom).read_bytes()).hexdigest() != self.geometry_digest:
                raise ValueError("geometry file changed after processor setup")
            import geom_coords
            coords = geom_coords.coords_from_geom(self.geom, shape, self.zdist)
        else:
            coords = self.coords
        try:
            status = self.detector.calibconst["pixel_status"][0].reshape((-1,) + tuple(shape))
            good = ~(status != 0).any(axis=0)
        except Exception:
            good = None  # Deliberately preserve the serial reader's fallback.
        self.prepared = xtc_core.prep_geometry(*coords, shape, good, self.zdist,
                                              lam=lam, **self.finder_options)
        self.shape = tuple(shape)

    def process_event(self, evt, *, source_ordinal):
        """Calibrate once and return one detached record. Caller supplies GLOBAL ordinal."""
        qrecords.integer(source_ordinal, "source_ordinal")
        timestamp = evt.timestamp
        # psana timestamps may be NumPy uint64; preserve every bit before JSON conversion.
        if isinstance(timestamp, np.integer):
            timestamp = int(timestamp)
        qrecords.integer(timestamp, "timestamp", maximum=2**64 - 1)
        record = dict(self.identity, timestamp=timestamp, source_ordinal=source_ordinal,
                      group_id=source_ordinal % self.identity["num_groups"],
                      calibration_attempts=1, peak_count=0, wavelength=None,
                      q=np.empty((0, 3), dtype=np.float64), peaks=np.empty((0, 3), dtype=np.float64))
        frame = self.detector.raw.calib(evt)
        if frame is None:
            record["status"] = "missing_calibration"
        else:
            lam = self._wavelength(evt)  # Legacy order: calibration before wavelength.
            if lam is None:
                record["status"] = "invalid_wavelength"
            else:
                if self.prepared is None:
                    self._prepare(frame.shape, lam)
                if tuple(frame.shape) != self.shape:
                    raise ValueError("calibrated frame shape changed within the run")
                X, Y, Zc, kin, finders = self.prepared
                q, px, intensity = xtc_core.frame_q(frame, finders, X, Y, Zc, kin,
                                                  lam, self.min_peaks, return_px=True)
                record.update(status="hit" if len(q) else "below_min_peaks", wavelength=float(lam),
                              peak_count=len(q), q=np.array(q, dtype=np.float64, order="C", copy=True),
                              peaks=np.array(xtc_core.slab_peaks(px, intensity, frame.shape[1]),
                                             dtype=np.float64, order="C", copy=True))
        return qrecords.validate_record(record)

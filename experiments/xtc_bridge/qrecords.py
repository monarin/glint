"""Version-1 compact CPU/GPU producer boundary: NumPy arrays and JSON scalars.

No reader, MPI, CUDA or GLINT imports. A record accounts for one in-range event,
including a non-hit. Out-of-range delivery is a separate Stage-3 audit concern.
The single-record NPZ helpers are for fixtures/replay; bounded chunk spooling and
whole-run manifest validation belong to the MPI producer, not this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

SCHEMA_VERSION = 1
STATUSES = {"hit", "below_min_peaks", "missing_calibration", "invalid_wavelength"}
FIELDS = {"schema_version", "config_digest", "experiment", "run", "timestamp",
          "source_ordinal", "group_id", "num_groups", "producer_rank", "status",
          "q", "peaks", "wavelength", "calibration_attempts", "peak_count"}
DEFAULT_MAX_BYTES = 16 * 1024 * 1024


def integer(value, name, minimum=0, maximum=None):
    """Require a JSON integer, never a bool, float or silently truncated timestamp."""
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def digest_config(config):
    """Hash a caller's complete scientific configuration/provenance, independent of rank."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def validate_record(record):
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("record fields do not match schema")
    integer(record["schema_version"], "schema_version")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    validate_digest(record["config_digest"], "config_digest")
    if not isinstance(record["experiment"], str) or not record["experiment"].strip():
        raise ValueError("experiment must be nonempty")
    for key in ("run", "source_ordinal", "group_id", "producer_rank", "peak_count"):
        integer(record[key], key)
    integer(record["timestamp"], "timestamp", maximum=2**64 - 1)
    integer(record["num_groups"], "num_groups", minimum=1)
    if record["group_id"] != record["source_ordinal"] % record["num_groups"]:
        raise ValueError("group_id must follow the reference source ordinal")
    integer(record["calibration_attempts"], "calibration_attempts", 1, 1)
    status = record["status"]
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError("unknown event status")
    for key in ("q", "peaks"):
        a = record[key]
        if (type(a) is not np.ndarray or a.dtype != np.dtype("float64") or
                a.ndim != 2 or a.shape[1] != 3 or not a.flags.c_contiguous or
                not np.isfinite(a).all()):
            raise ValueError(f"{key} must be finite contiguous float64 (N,3)")
    count = len(record["q"])
    if record["peaks"].shape != record["q"].shape or record["peak_count"] != count:
        raise ValueError("q/peaks/peak_count must align")
    if (status == "hit") != (count > 0):
        raise ValueError("only hits may carry nonempty peak arrays")
    lam = record["wavelength"]
    if status in ("hit", "below_min_peaks"):
        if type(lam) is not float or not np.isfinite(lam) or lam <= 0:
            raise ValueError("processed frames require a finite positive wavelength")
    elif lam is not None:
        raise ValueError("skipped frames must have wavelength=None")
    return record


def _metadata(record):
    return json.dumps({k: v for k, v in record.items() if k not in ("q", "peaks")},
                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def record_nbytes(record):
    """Compact payload bytes, excluding allocator/NPZ/transport overhead."""
    validate_record(record)
    return len(_metadata(record)) + record["q"].nbytes + record["peaks"].nbytes


def write_record(path, record, *, max_bytes=DEFAULT_MAX_BYTES):
    """Atomically replace one replay/fixture file using numeric arrays + JSON, no pickle."""
    integer(max_bytes, "max_bytes", 1)
    if record_nbytes(record) > max_bytes:
        raise ValueError("record exceeds max_bytes")
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, metadata=np.frombuffer(_metadata(record), dtype=np.uint8),
                     q=record["q"], peaks=record["peaks"])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_record(path, *, max_bytes=DEFAULT_MAX_BYTES):
    integer(max_bytes, "max_bytes", 1)
    # Inspect uncompressed sizes before NumPy allocates/decompresses any arrays.
    with open(path, "rb") as stream:
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            if (len(entries) != 3 or {x.filename for x in entries} !=
                    {"metadata.npy", "q.npy", "peaks.npy"}):
                raise ValueError("unexpected record archive members")
            if sum(x.file_size for x in entries) > max_bytes + 4096:
                raise ValueError("record archive exceeds max_bytes (including header allowance)")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            metadata = data["metadata"]
            if metadata.dtype != np.uint8 or metadata.ndim != 1:
                raise ValueError("metadata must be UTF-8 bytes")
            record = json.loads(metadata.tobytes().decode("utf-8"))
            if not isinstance(record, dict) or "q" in record or "peaks" in record:
                raise ValueError("invalid record metadata")
            record.update(q=data["q"], peaks=data["peaks"])
    if record_nbytes(record) > max_bytes:
        raise ValueError("record exceeds max_bytes")
    return record


def to_legacy(records, *, experiment, run, config_digest, group_id, num_groups,
              expected_ordinals, max_bytes):
    """Build one COMPLETE consensus group's legacy reader dictionary, ordered by ordinal.

    expected_ordinals includes non-hits, from a frozen external reference. This
    gate checks group membership/completeness and duplicate timestamps/ordinals;
    Stage 3 must additionally match each identity to the full reference manifest.
    max_bytes caps compact payload, not peak process RSS: arrays are shared with
    the input records, and later envbridge encoding/decoding makes copies.
    """
    integer(max_bytes, "max_bytes", 1)
    integer(run, "run")
    integer(num_groups, "num_groups", 1)
    integer(group_id, "group_id", 0, num_groups - 1)
    validate_digest(config_digest, "config_digest")
    expected = set()
    for ordinal in expected_ordinals:
        integer(ordinal, "expected ordinal")
        if ordinal in expected or ordinal % num_groups != group_id:
            raise ValueError("invalid/duplicate expected group ordinal")
        expected.add(ordinal)
    by_ordinal, identities = {}, set()
    total = 0
    for record in records:
        total += record_nbytes(record)
        if total > max_bytes:
            raise ValueError("complete group exceeds max_bytes; do not split consensus groups")
        for key, wanted in (("experiment", experiment), ("run", run),
                            ("config_digest", config_digest), ("group_id", group_id),
                            ("num_groups", num_groups)):
            if record[key] != wanted:
                raise ValueError(f"mixed group {key}")
        ordinal, timestamp = record["source_ordinal"], record["timestamp"]
        if ordinal not in expected or ordinal in by_ordinal or timestamp in identities:
            raise ValueError("unexpected or duplicate event identity/ordinal")
        by_ordinal[ordinal] = record
        identities.add(timestamp)
    if set(by_ordinal) != expected:
        raise ValueError("incomplete group: missing expected events (including non-hits)")
    ordered = [by_ordinal[i] for i in sorted(by_ordinal)]
    hits = [r for r in ordered if r["status"] == "hit"]
    return dict(qframes=[r["q"] for r in hits], events=[r["source_ordinal"] for r in hits],
                peaks=[r["peaks"] for r in hits], lams=[r["wavelength"] for r in hits],
                n_events=sum(r["status"] in ("hit", "below_min_peaks") for r in ordered),
                n_sent=len(hits), n_skipped_wl=sum(r["status"] == "invalid_wavelength" for r in ordered))

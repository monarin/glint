"""Ordinary GLINT process for the launch probe; deliberately no MPI/psana imports."""
from __future__ import annotations

import os
from pathlib import Path
import time
from types import SimpleNamespace

from glint_probe_common import record_digest
from glint_child_runtime import initialize as initialize_runtime, isolated

_ready = None


def initialize(parent, config):
    global _ready
    _ready = initialize_runtime(parent, config)
    return _ready


def consume(record, mode, output, fault=None, delay=0):
    if _ready is None:
        raise RuntimeError("initialize the child before consuming records")
    isolated()
    digest = record_digest(record)
    if fault == "exception":
        raise RuntimeError("injected GLINT child exception")
    if fault == "crash":
        os._exit(73)
    if fault == "timeout":
        time.sleep(delay)
    result = dict(digest=digest, events=record["events"], n_frames=len(record["events"]),
                  pid=os.getpid(), output=None, n_indexed=0)
    if mode == "index":
        if _ready["gpu"] is None:
            raise ValueError("indexing requires verified GPU placement")
        # Hard-coded fixture options: there is no reader or integration option.
        from glint_xtc import index_and_write
        args = SimpleNamespace(exp="glint-probe", run=0, geom=None, zdist=0.1,
                               wavelength=1.322, cell=["79.02", "79.02", "37.98", "90", "90", "90"],
                               nbest=1, integrate=False)
        _, _, count = index_and_write(record, args, output, report=False)
        if record["events"] and count == 0:
            raise RuntimeError("indexing fixture produced no indexed frames")
        path = Path(output)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("missing indexing output")
        result.update(output=str(path), n_indexed=int(count), output_bytes=path.stat().st_size)
    elif mode != "echo":
        raise ValueError("unknown probe mode")
    isolated()
    return result

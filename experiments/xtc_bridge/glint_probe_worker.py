"""Ordinary GLINT process for the launch probe; deliberately no MPI/psana imports."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from glint_probe_common import gpu_identity, identity, record_digest

_ready = None


def isolated():
    forbidden = {name.split(".")[0] for name in sys.modules} & {"mpi4py", "psana"}
    if forbidden:
        raise RuntimeError(f"GLINT child imported forbidden modules: {sorted(forbidden)}")


def initialize(parent, config):
    global _ready
    isolated()
    # Linux parent-death protection covers calls after this handshake. The scheduler
    # owns cancellation of the activation window before this function can run.
    if not sys.platform.startswith("linux"):
        raise RuntimeError("stage-1 supervision requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent["pid"]:
        raise RuntimeError("bridge parent changed during startup")
    result = identity()
    if result["cuda_visible_devices"] != parent["cuda_visible_devices"]:
        raise ValueError("activation changed CUDA_VISIBLE_DEVICES")
    expected = config.get("expected_child_prefix")
    if expected is not None and Path(result["prefix"]).resolve() != Path(expected).resolve():
        raise ValueError("unexpected child environment prefix")
    if not config["allow_same_environment"] and Path(result["prefix"]).resolve() == Path(parent["prefix"]).resolve():
        raise ValueError("probe requires separate parent and child environments")
    result["gpu"] = None
    if config["check_gpu"]:
        import torch
        ordinal = parent["gpu"]["ordinal"]
        torch.cuda.set_device(ordinal)
        result["gpu"] = gpu_identity(ordinal)
        if result["gpu"]["pci"] != parent["gpu"]["pci"] or torch.cuda.current_device() != ordinal:
            raise ValueError("child GPU does not match producer assignment")
        # A tiny allocation verifies that the selected Torch device is usable.
        torch.ones(1, device=f"cuda:{ordinal}").sum().item()
        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
    import numpy as np
    result["numpy"] = np.__version__
    isolated()
    result["mpi_imported"] = False
    result["psana_imported"] = False
    _ready = result
    return result


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

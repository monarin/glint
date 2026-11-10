"""Initialize an ordinary GLINT child without importing analysis runtimes eagerly."""
import ctypes
import os
from pathlib import Path
import signal
import sys

from glint_psana_common import gpu_identity, identity


def isolated():
    forbidden = {name.split(".")[0] for name in sys.modules} & {"mpi4py", "psana"}
    if forbidden:
        raise RuntimeError(f"GLINT child imported forbidden modules: {sorted(forbidden)}")


def initialize(parent, config):
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
    return result

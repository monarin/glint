"""Process identity and atomic reports shared by the native path and launch probe.

Keep module imports stdlib-only; CUDA is initialized only by gpu_identity.
"""
import json
import os
from pathlib import Path
import socket
import sys


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def identity():
    return dict(pid=os.getpid(), host=socket.gethostname(), executable=sys.executable,
                prefix=sys.prefix, python=sys.version.split()[0],
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_path=os.environ.get("CUDA_PATH"),
                ld_library_path=os.environ.get("LD_LIBRARY_PATH"))


def gpu_identity(ordinal):
    import cupy as cp
    if ordinal >= cp.cuda.runtime.getDeviceCount():
        raise ValueError("assigned device is not visible")
    device = cp.cuda.Device(ordinal)
    device.use()
    # Initial recipe is non-MIG: PCI identity does not distinguish MIG instances.
    return dict(ordinal=ordinal, pci=device.pci_bus_id.upper(), cupy=cp.__version__,
                cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()))

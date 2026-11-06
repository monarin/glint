"""Small, CPU-only contracts for the stage-1 launch probe (not a production codec)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

RANK_ROLES = ("control-smd0", "control-eb", "producer", "producer")


def load_config(path):
    config = json.loads(Path(path).read_text())
    defaults = dict(mode="echo", check_gpu=False, devices={"2": 0, "3": 1},
                    empty_producers=[], fault=None, activation_timeout=120.0,
                    call_timeout=120.0, job_timeout=600.0, allow_same_environment=False)
    allowed = set(defaults) | {"psana_prelude", "glint_prelude", "expected_child_prefix"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unknown probe configuration fields")
    config = dict(defaults, **config)
    for key in ("psana_prelude", "glint_prelude"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be an explicit trusted shell activation")
    for key in ("activation_timeout", "call_timeout", "job_timeout"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("check_gpu", "allow_same_environment"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    prefix = config.get("expected_child_prefix")
    if prefix is not None and (not isinstance(prefix, str) or not prefix.strip()):
        raise ValueError("expected_child_prefix must be a nonempty string or null")
    if config["mode"] not in ("echo", "index"):
        raise ValueError("mode must be echo or index")
    if config["mode"] == "index" and not config["check_gpu"]:
        raise ValueError("index mode requires explicit GPU verification")
    if not isinstance(config["empty_producers"], list) or any(type(r) is not int or r not in (2, 3) for r in config["empty_producers"]):
        raise ValueError("empty_producers must contain only ranks 2 and 3")
    devices = config["devices"]
    if not isinstance(devices, dict) or set(devices) != {"2", "3"} or any(type(d) is not int or d < 0 for d in devices.values()):
        raise ValueError("devices must map ranks 2 and 3 to nonnegative visible ordinals")
    fault = config["fault"]
    if fault is not None and (not isinstance(fault, dict) or set(fault) != {"rank", "kind"} or
                              fault["rank"] not in (2, 3) or fault["kind"] not in ("activation", "exception", "crash", "timeout")):
        raise ValueError("fault must name producer rank and activation/exception/crash/timeout")
    return config


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


def fixture(rank, empty=False):
    """Two fixed reciprocal lattices, independent of GLINT imports and randomness."""
    import numpy as np
    hkl = np.array([(h, k, l) for h in range(-4, 5) for k in range(-4, 5)
                    for l in range(-2, 3) if (h, k, l) != (0, 0, 0)], dtype=np.float64)
    q = hkl / np.array([79.02, 79.02, 37.98])
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    frames = [] if empty else [np.ascontiguousarray(q), np.ascontiguousarray(q @ rotation)]
    return dict(qframes=frames, events=[] if empty else [rank * 2, rank * 2 + 1],
                n_events=len(frames), n_sent=len(frames), n_skipped_wl=0)


def record_digest(record):
    import numpy as np
    if set(record) != {"qframes", "events", "n_events", "n_sent", "n_skipped_wl"}:
        raise ValueError("invalid probe record fields")
    if len(record["events"]) != len(record["qframes"]) or len(set(record["events"])) != len(record["events"]):
        raise ValueError("invalid event alignment")
    if record["n_events"] != len(record["events"]) or record["n_sent"] != len(record["events"]) or record["n_skipped_wl"] != 0:
        raise ValueError("invalid probe counts")
    digest = hashlib.sha256(json.dumps(record["events"]).encode())
    for q in record["qframes"]:
        if not isinstance(q, np.ndarray) or q.dtype != np.float64 or q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError("q must be finite float64 (N,3)")
        digest.update(str(q.shape).encode())
        digest.update(q.tobytes())
    return digest.hexdigest()


def validate_placement(records, check_gpu):
    if len(records) != 4 or [r["rank"] for r in records] != list(range(4)):
        raise ValueError("probe requires exactly four ordered ranks")
    if len({r["host"] for r in records}) != 1:
        raise ValueError("stage-1 probe requires one node")
    for r in records:
        if r["role"] != RANK_ROLES[r["rank"]]:
            raise ValueError("unexpected probe role")
        if r["rank"] < 2 and r.get("gpu") is not None:
            raise ValueError("control rank initialized a GPU")
    if check_gpu and len({r["gpu"]["pci"] for r in records[2:]}) != 2:
        raise ValueError("producer ranks must use distinct physical GPUs")

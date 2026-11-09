"""Launch one MPI probe in a caller-specified psana environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from glint_psana_common import atomic_json


def launch_command(config, config_path, output, launcher, mode="probe"):
    if not launcher:
        raise ValueError("provide an MPI launcher after --")
    probe = Path(__file__).resolve().with_name(
        "psana_mpi_producer.py" if mode == "cpu" else "probe_psana_glint.py")
    command = [*launcher, "python", str(probe), "--config", str(config_path), "--output", str(output)]
    return ["bash", "-c", config["psana_prelude"] + " && exec " + shlex.join(command)]


def stop_launcher(process):
    # Scheduler cancellation handles descendants that created separate sessions.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def interrupted(signum, _frame):
    raise InterruptedError(f"received signal {signum}")


def inside_rank(environ):
    if any(key in environ for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")):
        return True
    # Slurm can set PROCID=0 on the batch script itself. A numbered job step
    # with PROCID is a task context; the batch/extern allocation is not.
    step = environ.get("SLURM_STEP_ID", environ.get("SLURM_STEPID"))
    return "SLURM_PROCID" in environ and step not in (None, "batch", "extern")


def main(argv=None):
    parser = argparse.ArgumentParser(description="glint-psana: launch probe or native psana MPI CPU path")
    parser.add_argument("--mode", choices=("probe", "cpu"), default="probe")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory on the probe node's shared filesystem")
    parser.add_argument("launcher", nargs=argparse.REMAINDER, help="-- mpirun -n 4 (or verified srun arguments)")
    args = parser.parse_args(argv)
    process = None
    output = None
    old_handlers = {}
    try:
        if inside_rank(os.environ):
            keys = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID", "SLURM_STEP_ID", "SLURM_STEPID")
            detected = ", ".join(f"{key}={os.environ[key]}" for key in keys if key in os.environ)
            raise ValueError("glint-psana must run outside MPI ranks and srun steps; "
                             "launch once from the batch script or an salloc shell. Detected: " + detected)
        if args.mode == "cpu":
            from psana_cpu_config import load_config as load_cpu_config
            config = load_cpu_config(args.config)
        else:
            from glint_probe_common import load_config
            config = load_config(args.config)
        launcher = args.launcher[1:] if args.launcher[:1] == ["--"] else args.launcher
        destination = args.output.resolve()
        config_path = destination / "config.json"
        command = launch_command(config, config_path, destination, launcher, args.mode)
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        atomic_json(config_path, config)
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        process = subprocess.Popen(command, start_new_session=True)
        code = process.wait(timeout=config["job_timeout"])
        if code:
            raise RuntimeError(f"MPI launcher exited {code}")
        report = json.loads((output / "mpi-complete.json").read_text())
        if report.get("status") != "complete" or len(report.get("ranks", [])) != 4:
            raise ValueError("missing valid MPI completion report")
        if args.mode == "cpu":
            merged = output / "merged.stream"
            if report.get("mode") != "cpu" or report.get("harness") is not False or not merged.is_file():
                raise ValueError("missing native CPU completion/output")
            digest = hashlib.sha256()
            with merged.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            expected = report.get('merged', {})
            if (expected.get('output') != str(merged) or
                    expected.get('output_bytes') != merged.stat().st_size or
                    expected.get('sha256') != digest.hexdigest()):
                raise ValueError("merged stream differs from MPI completion report")
        atomic_json(output / "complete.json", report)
        print(f"GLINT_PSANA_{args.mode.upper()}_OK {output / 'complete.json'}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # Ignore repeated cancellation while reaping the owned launcher.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            stop_launcher(process)
        if output is not None:
            # A cancellation or logging error can arrive just after the atomic
            # success rename. Never leave success beside a failed CLI outcome.
            (output / "complete.json").unlink(missing_ok=True)
            atomic_json(output / "failed.json", dict(error=f"{type(exc).__name__}: {exc}"))
        print(f"glint-psana: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())

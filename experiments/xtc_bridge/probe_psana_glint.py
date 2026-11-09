"""Four-rank synthetic launch harness; roles represent psana, not a DataSource run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys

from envbridge import Env, Worker
from glint_probe_common import (RANK_ROLES, fixture, load_config,
                                record_digest, validate_placement)
from glint_psana_common import atomic_json, gpu_identity, identity


def interrupted(signum, _frame):
    raise InterruptedError(f"rank interrupted by signal {signum}")


def producer(parent, config, output):
    rank = parent["rank"]
    root = Path(__file__).resolve().parent
    # Narrow audit tag for finding activation descendants during cancellation tests.
    os.environ["GLINT_PSANA_PROBE_ID"] = str(output)
    fault = config["fault"]
    kind = fault["kind"] if fault and fault["rank"] == rank else None
    prelude = "false" if kind == "activation" else config["glint_prelude"]
    env = Env(prelude, label="glint-probe", pythonpath=(str(root), str(root.parents[1])),
              inherit_pythonpath=False)
    worker = Worker(env, timeout=config["activation_timeout"], call_timeout=config["call_timeout"])
    try:
        worker.start()
        child = worker.call("glint_probe_worker:initialize", parent, config)
        atomic_json(output / f"child-{rank}.json", child)
        record = fixture(rank, rank in config["empty_producers"])
        answer = worker.call("glint_probe_worker:consume", record, config["mode"],
                             str(output / f"group-{rank}.stream"), fault=kind,
                             delay=config["call_timeout"] * 2 + 5)
        if answer["digest"] != record_digest(record) or answer["events"] != record["events"] or answer["pid"] != child["pid"]:
            raise ValueError("child result differs from assigned fixture")
        worker.close()
        # Worker owns/reaps the subprocess; inspect its exit status, not a PID
        # existence check which could accidentally identify a reused PID.
        if worker._proc.returncode != 0:
            raise RuntimeError("child did not exit cleanly")
        return dict(parent, child=child, result=answer, child_reaped=True)
    finally:
        worker.close(kill=worker._proc is not None and worker._proc.poll() is None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    output = args.output.resolve()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        config = load_config(args.config)
        if comm.Get_size() != 4:
            raise ValueError("launch probe requires exactly four MPI ranks")
        parent = dict(identity(), rank=rank, role=RANK_ROLES[rank], gpu=None)
        if rank >= 2 and config["check_gpu"]:
            parent["gpu"] = gpu_identity(config["devices"][str(rank)])
        atomic_json(output / f"parent-{rank}.json", parent)
        placements = comm.allgather(parent)
        validate_placement(placements, config["check_gpu"])
        if rank >= 2:
            report = producer(parent, config, output)
        else:
            if "cupy" in sys.modules or "torch" in sys.modules:
                raise RuntimeError("control role initialized an analysis framework")
            report = dict(parent, child=None, result=None, child_reaped=True)
        reports = comm.gather(report, root=0)
        if rank == 0:
            if any(r["child"] is not None for r in reports[:2]) or any(r["child"] is None or not r["child_reaped"] for r in reports[2:]):
                raise RuntimeError("incomplete child ownership report")
            atomic_json(output / "mpi-complete.json", dict(status="complete", harness=True,
                        mode=config["mode"], gpu_checked=config["check_gpu"], ranks=reports,
                        mpi_library=MPI.Get_library_version().rstrip("\x00")))
        comm.Barrier()
        return 0
    except BaseException as exc:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        try:
            atomic_json(output / f"rank-{rank}-failed.json", dict(rank=rank, error=f"{type(exc).__name__}: {exc}"))
        finally:
            print(f"glint-psana rank {rank}: {exc}", file=sys.stderr, flush=True)
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# Concurrent duplicated-reader versus native psana timing

This opt-in experiment compares A: `glint_xtc_mpi.main` with two independent
serial psana2 readers, and B: `psana_mpi_producer.main` with SMD0, EB and two BDs.
Both use two identical consensus groups and the same two-GPU/eight-CPU allocation.
A's launcher selects one allocated GPU token per rank before MPI/CUDA imports;
B retains the allocation mask and uses its existing explicit device map.
The MPI launcher uses `--bind-to none` in both arms; all processes stay within
the scheduler CPU allocation. Thread limits belong to the batch script.

`psana_ab_trial.py` wraps the existing applications. A retains the original
reader/bridge/index/merge path; the benchmark supplies the same explicit source
directory and DataSource event guard as B. It substitutes the configured reader
activation prelude to prevent cross-version PYTHONPATH leakage, preserving the
envbridge transport. B retains its normal spooling and completion checks.
`psana_ab_measure.py` provides aggregate read bytes/calls, host phase timers,
calibration-call counts, and output-only diagnostics. Neither arm hashes full
calibrated images or records per-read traces. Source configuration preparation,
GPU preflight, output comparisons and the frozen-reference acquisition occur
outside timed trials. Tiny metric files and output fingerprints are inside the
trial and are disclosed overhead. Production scientific code is unchanged.

Run `benchmark_psana_ab.py` once outside MPI with an existing frozen CPU config
having `num_groups=2`, positive source limit, and matching reference. It launches
one warm-up per arm, then three paired fresh-process trials in AB, BA, AB order.
Primary wall time includes environment activation, MPI startup, bridge startup,
processing, output merge and teardown. Shared filesystem and CuPy caches persist;
this is a warm-cache comparison, with no claim about cold-storage performance.
The warm-ups are excluded from medians. Exact stream bytes and q/peak fingerprints
must match within each pair. Calibration calls must equal the frozen event count.
All tagged processes must be gone before the next trial starts.

A 203-event r51 sample is the initial bounded workload. It has few hits and earlier
checks found no strict-indexing-positive frame; timing it does not satisfy Stage4
scientific acceptance or establish throughput on indexing-heavy data. The number
of groups differs from the earlier K=3 parity check: expected duplicated bulk
read volume here is twice native, not three times. Device utilization is sampled
once per second for allocated GPUs only. RSS is per process; peaks must not be
summed as if simultaneous. Phase intervals can overlap or nest; the primary
comparison uses coordinator wall time, never sums per-rank seconds.

Outputs: source/config provenance, command and per-trial logs, GPU samples,
process metrics, index diagnostics, `trials.json`, and `summary.json` with paired
A/B wall ratios. A ratio above one favors B; below one favors A. Three pairs
support a descriptive comparison, not a claim of statistical significance.
Failed trials stop collection and remain separate from accepted results.

CPU checks: `python experiments/xtc_bridge/test_psana_ab.py`.

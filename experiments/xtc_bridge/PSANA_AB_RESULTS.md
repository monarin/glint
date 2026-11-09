# Duplicated readers versus glint-psana: first A/B trials

The native path halved application bulk-read bytes, but did **not** improve
end-to-end runtime on this 203-event sample. Three measured pairs gave median
wall times of 178.849 s for A and 189.862 s for B: B took 6.16% longer by the
ratio of medians. The median paired slowdown was 5.60%; individual pairs ranged
from 3.61% to 9.34%. These are descriptive results, not significance estimates. Persistent GPU
activity indicates an external workload; exclusive access was not verified,
so these timings include a possible contention effect.

A is the existing GLINT MPI path with two concurrent serial psana readers.
B is native glint-psana MPI with SMD0, EB, two BDs and two GLINT children.
Both use two consensus groups and the same allocation. See
[benchmark methods](PSANA_AB_BENCHMARK.md) for the wrappers and timing boundary.

| Pair | Execution order | A seconds | B seconds | B additional time |
| --- | --- | ---: | ---: | ---: |
| 1 | A then B | 174.439 | 190.739 | 9.34% |
| 2 | B then A | 179.796 | 189.862 | 5.60% |
| 3 | A then B | 178.849 | 185.299 | 3.61% |
| Median | | 178.849 | 189.862 | 6.16% (ratio of medians) |

One warm-up per arm preceded these pairs and is excluded: A 170.179 s,
B 192.291 s. Processes were fresh each time; filesystem and CuPy caches persisted.
All times include environment activation, MPI/bridge startup, processing, merge
and teardown. No claim is made about cold-storage or sustained-run throughput.

## What improved, and what the timings show

Every A trial read 14,586,873,966 bytes through the instrumented application
bulk-read path; B read 7,293,436,983 bytes, exactly half. These are application
bytes, not physical disk traffic. Both arms made 203 calibration calls: the
baseline duplicates reads, not calibration of assigned frames.

| Measured phase statistic, median across trials | A seconds | B seconds |
| --- | ---: | ---: |
| Bulk-read time summed across reader ranks | 3.604 | 2.126 |
| Calibration time, maximum rank | 32.716 | 32.516 |
| Peak/q time, maximum rank | 6.291 | 5.933 |
| Indexing time, maximum group | 0.595 | 0.709 |

Bulk reads occupied only a few aggregate seconds and indexing was brief.
Startup, initialization and orchestration are substantial relative to this
small workload. B's maximum child start plus initialization interval had a
median of 81.238 s. This is **not** 81 seconds of incremental overhead over A:
A loads GLINT in its parent and activates separate reader children, so its phase
boundaries differ. Timers overlap or nest, and rank totals must not be added to
reconstruct wall time. The measurements do not isolate the slowdown's cause.

GPU telemetry contains a confound: UUID
`GPU-3b4c2d1a-0e2b-ec2a-5d7d-519650b2ee9b` already showed 927 MiB resident
and 25% utilization at the first warm-up sample, before the benchmark's CUDA
work. Repeated bursts reached 92–98% and 2625 MiB, and a similar pattern persisted
through all eight trials. This is evidence of a persistent external workload
(a co-tenant or an earlier surviving process); its identity was not recorded.
The roughly 24–29% sampled mean utilization and peak memory for this UUID cannot
be attributed to this benchmark. The benchmark's own indexing work was brief.

Both arms used the same devices and alternated order, but equal contention
impact is not established. **Exclusive GPU access was not verified.** The wall
times remain observed outcomes under those conditions, not an uncontended
performance comparison. The telemetry records UUIDs, whereas placement records
PCI IDs; their mapping was not captured. A post-run identity query was denied
because the allocation had ended, so the affected consensus group is unknown.

Maximum individual-process RSS had medians of 3,396,208 KiB (A) and
5,119,684 KiB (B); these are not simultaneous job totals. A longer representative
input, with GPU ownership and pre-trial activity checked, is needed to test
whether read-volume savings improve sustained throughput.

## Validation and provenance

- Runtime source: `e9514a8b7b1bf8dba707de759e704e693e8823cb`.
- Slurm job `43992026`, `sdfada008`, completed in 24:56 with exit 0.
- Two GPUs on the ada partition, eight allocated logical CPUs, 64 GB memory; one thread per
  configured numerical library. Both launchers use `--bind-to none` within the
  same scheduler affinity. Every recorded reader/parent affinity matches it.
- Input: `mfx100848724`, run 51, Jungfrau16M, frozen 203-event prefix, batch 5,
  two groups, fixed geometry/configuration and random seed.
- All four comparisons (warm-up plus three measured pairs) passed exact merged
  and per-group stream equality, q/peak fingerprints, event lists and indexing
  diagnostics. Every trial returned 14 hits and five raw cells, with zero
  strict-indexing passes. Native completion/child-reaping checks passed;
  recorded tagged-survivor lists were empty between trials.
- Post-run analysis rechecked summary hashes/arithmetic, trial order, all output
  comparisons, both A parent metric files, CPU affinities, and more than ten
  parseable samples on each of the same two GPU UUIDs in every trial.
- Six new CPU tests passed under NumPy 1 and 2; all six GitHub CI checks passed
  on the runtime revision.

Raw artifacts, including commands, source hashes, configuration, scalar metrics,
GPU samples and scheduler log, are under
`/sdf/scratch/users/m/monarin/gpu-validation/glint-ab/run-43992026/` and
`/sdf/scratch/users/m/monarin/gpu-validation/glint-ab/job-43992026.log`.
The immutable source snapshot is `glint-ab/source-e9514a8/`; post-run analysis
and independent review records are archived under `glint-ab/review/` in the same
scratch tree. The post-run analyzer does not modify runtime artifacts.

The exact floating-point output gate was retained for the frozen trials;
it can reject harmless variation but all actual pairs passed. Monitor exit
codes were not captured, so telemetry was validated from actual sample content.
Per-process thread environment snapshots were not captured; thread settings
come from the frozen batch script. Cleanup evidence is the tagged-process check
and native child-reaping records, not an exhaustive process audit.

This timing benchmark did not close Stage4 acceptance blocker S4-01. A later
[correctness matrix](GLINT_PSANA_STAGE4_ACCEPTANCE.md) supplies strict-positive
coverage and a complete single-invocation run; it does not change these timings.

[Claude's final review](https://github.com/monarin/glint/pull/4#pullrequestreview-5830638691) passed the corrected report at
`f6e39efbef5e082347eab60dc7a2082139928994`, with no remaining blocking A/B
findings. AB-01 (runtime evidence), AB-05 (GPU activity attribution) and AB-06
(provenance) are resolved; documented frozen-run deferrals remain accepted.
Measured code stays at `e9514a8b7b1bf8dba707de759e704e693e8823cb`.
This review-status update changes no code, measurements, or interpretation.

# Stage 4 correctness acceptance

The complete acceptance matrix passed on a fixed 1,003-event prefix of
`mfx100848724` run 51. All three native cases reproduce the independent serial
reference, including three frames passing the unchanged strict indexing gate.
The coordinator produced `acceptance-complete.json` in one invocation.
Independent review of this new evidence is pending.

Runtime code: `727a91076649c7d33b8434b9708f360e1f7ebbce`.
Slurm job `44002628`, node `sdfada018`, completed with exit 0 in 28:24, including
preflight and sample discovery. The full matrix itself took 1,417.7 seconds
(23:37.7), within its 3,000-second deadline. This is a diagnostic correctness
run with image hashing and event/read audits, not a performance benchmark.

| Check | Batch 1 | Batch 5 | Batch 5, swapped GPUs |
| --- | ---: | ---: | ---: |
| Delivered events | 1,003 | 1,003 | 1,003 |
| Hits / below peak threshold | 67 / 936 | 67 / 936 | 67 / 936 |
| Raw returned-cell frames | 35 | 35 | 35 |
| Strict-gate-passing frames | **3** | **3** | **3** |
| Calibration calls, native | 1,003 | 1,003 | 1,003 |
| Maximum absolute q difference | 0 | 0 | 0 |
| Changed BD owner versus batch 1 | — | 513 | 491 |
| Native overlapping bulk-read bytes | 0 | 0 | 0 |

All cases pass calibrated-image hash and shape equality, exact peak-array and
serialized group-stream equality, q comparisons, equivalent indexed bases,
independently recomputed strict matches, and group/merged output integrity.
Both BD workers participate, all in-range events have one producer owner, and
batch 5 exercises a three-event tail. Control ranks process no detector events;
GLINT children remain outside psana/MPI and pass the Python-level XTC-open guard.
Child-reaping records and the final tagged-process survivor check pass. These
are scoped cleanup/isolation checks, not an exhaustive audit of arbitrary
C-library I/O or every process on the node.

## Strict-positive coverage and frozen input

The unchanged gate requires at least 10 strict matches and at least 25% of the
complete input q array, with tolerance 0.15. The same three events pass in the
reference and every native case:

| Source ordinal | Strict matches | Input peaks |
| --- | ---: | ---: |
| 242 | 10 | 17 |
| 428 | 12 | 30 |
| 432 | 12 | 46 |

The candidate was a contiguous 1,003-event prefix, chosen before discovery.
Discovery used existing baseline instrumentation on group 0 and found event
432; the full coordinator then independently froze the reference and reran all
three baseline groups. It did not reuse discovery outputs as matrix results.
Geometry, calibration, wavelength 1.322 A, distance 0.356470 m, lysozyme cell,
V4 defaults, three consensus groups, and the strict gate were unchanged from the
previous 203-event recipe. No selected-event list, synthetic injection or
threshold tuning was used. This establishes the declared positive-coverage
criterion and parity, not a broad indexing-accuracy or yield claim.

The reference made 1,003 calibration calls total, as did every native case.
Three independent serial readers returned **108,108,137,349 bytes** through the
instrumented bulk-read path; each native case returned **36,036,045,783 bytes**,
with no overlapping returned intervals. Thus the application bulk-read volume
is one third of the reference. This excludes SMD/metadata reads and does not
measure physical storage traffic, cache behavior or speedup.

## Provenance, checks and reproduction

Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/glint-stage4-acceptance/`

- `source-727a910/`: immutable runtime source snapshot.
- `run-44002628/matrix/`: complete single-invocation evidence, command logs,
  frozen reference/geometry/config, baseline/native records, image/read audits,
  indexing diagnostics, `comparison.json` and `acceptance-complete.json`.
- `run-44002628/discovery/`: separate candidate discovery evidence.
- `run-44002628/allocation.json`, `cuda-preflight.log`, `gpu-preflight.json`:
  affinity, environment mask, ordinal/PCI/UUID mapping and initial idle samples.
- `job-44002628.log`: wrapper completion; matrix log is within the run directory.
- `review/`: post-run analyzer and results, test/CI evidence and independent review.

Two L40S GPUs were mapped to PCI `03:00.0` and `85:00.0`, with eight allocated
logical CPUs and 64 GB memory. After CUDA-probe activity settled, both devices
had three consecutive samples of 0% utilization and 0 MiB used. Initial idle
samples do not prove exclusive GPU use throughout the run. This run makes no
performance claim.

Post-run verification reran all three comparisons, checked the source hashes
of 111 files, recomputed the final comparison digest, verified command order and
successful exits, matched discovery/acceptance references, and checked coverage,
ownership changes, the partial tail, read/calibration counts and final cleanup.
Twelve acceptance-checker tests passed. All six CI jobs passed on runtime code
727a910 with zero annotations; production code was unchanged by this task.

Use the updated `psana_parity.example.json` (1,003 events, 900-second subprocess
limit), the same site environments, an allocation with two visible GPUs, and:

```bash
python experiments/xtc_bridge/validate_psana_parity.py \
  --config experiments/xtc_bridge/psana_parity.example.json \
  --output /shared/scratch/new-acceptance-run \
  --revision SOURCE_COMMIT --timeout 3000 --min-indexed 1
```

Generated runtime copies and caches belong on scratch. Preserve the recorded
geometry/calibration/configuration identities when reproducing this result.

## Earlier attempts remain distinct

Jobs 44002184 and 44002321 stopped before reading data on `sdfada011`: the
first preflight saw one device record, and the corrected CUDA preflight reported
`cudaErrorUnknown`. Job 44002488 on `sdfada018` verified two CUDA devices but its
immediate idle sample caught 9% utilization with 0 MiB after our own probe.
The wrapper was corrected to allow settling, requiring three consecutive idle
samples within 30 seconds; job 44002628 then passed. Original wrapper versions
and logs remain in the artifact root. No acceptance result combines these jobs.

The earlier 203-event validation and isolated swapped-case retry remain
[historical evidence](GLINT_PSANA_PARITY_VALIDATION.md). They are superseded by
this complete matrix for the Stage 4 correctness criterion. The earlier A/B
performance result and its GPU-contention caveat remain separate and unchanged.

# PR5 cleanup acceptance on Perlmutter

Fresh acceptance passed at implementation
`a74cf58ad111b712ac5711336b18f8edcb8ddddc`. Slurm job **61061576** completed
`0:0` on `nid001457` in 17:28. One matrix coordinator invocation completed all
three independent serial baselines and all three native cases in 994.65 seconds.
This is new runtime evidence; the historical Stage4 report is unchanged.

## Recipe and results

The sample is the original mfx100848724 run 51 Jungfrau-16M contiguous prefix,
physical segments 0–31, shape `(32, 512, 1024)`, three consensus groups, fixed
wavelength 1.322 Å, distance 0.356470 m, cell `[79.17,79.17,37.96,90,90,90]`,
the historical frozen geometry, and unchanged V4 defaults. A newly frozen reference
matches **all 1,003 historical timestamps and the full calibration-value digest**.
The calibration digest is
`532ab4eb3304ee71aee1224e39eec1897df5a8d7a0839e4c19b8f45e354c4831`.

| Check | Batch 1 | Batch 5 | Batch 5, swapped GPUs |
|---|---:|---:|---:|
| Events / hits | 1003 / 67 | 1003 / 67 | 1003 / 67 |
| Calibration calls, baseline / native | 1003 / 1003 | 1003 / 1003 | 1003 / 1003 |
| Maximum absolute q error | 0 | 0 | 0 |
| Strict-positive event ordinals | 242, 428, 432 | 242, 428, 432 | 242, 428, 432 |
| Ownership changes versus batch 1 | — | 536 | 451 |
| Native overlapping bulk-read bytes | 0 | 0 | 0 |

Calibrated-image diagnostics and peak arrays match exactly. Group/merged output
validation, exact group stream comparison, strict indexing gates (minimum 10,
fraction 0.25, tolerance 0.15), complete/unique ownership, both-BD participation,
and the three-event partial tail passed. Three serial readers returned
108,108,137,349 application bulk-read bytes; each native case returned
36,036,045,783. All child acknowledgements, separate-prefix/import-isolation checks,
GPU matching, and child reaping passed. Coordinator and driver found no tagged
survivors. The allocated synthetic probe also passed with `mode=index`,
`check_gpu=true`, two ordinary children, and distinct physical GPUs.

An independent post-run checker, importing the detached review checkout, reran
all three comparisons and matrix gates; verified source and artifact digests,
completion markers, historical reference/geometry equality, and child reports;
and confirmed the 619 installed psana Python/extension hashes stayed unchanged.

## Runtime and verification

- Parent: Python 3.9.20, NumPy 1.26.4, CuPy 13.6.0, Cray MPICH 9.0.1.498;
  psana built from `7058d9961d6eb79eabf2d2f9cdf54df7ad293a8e` into `install_psana`.
- Separate child venv: Torch 2.7.1+cu126, CuPy 13.6.0, NumPy 1.26.4.
  CuPy used CUDA runtime 12.9; Torch reported CUDA 12.6. Actual allocations and
  CuPy compilation were checked in both environments.
- One premium `lcls_g` node, four A100-SXM4-40GB GPUs allocated; analysis used
  ordinals 0/1. Four MPI ranks, 16 CPU slots per rank; coordinator in the batch
  step. Launcher: `srun --mpi=cray_shasta --nodes=1 --ntasks=4
  --cpus-per-task=16 --gpu-bind=none`, recorded for each native case.
- Bounds fixed before launch: 75-minute allocation, 3600-second matrix deadline,
  1800-second per-launch deadline, 180-second activation, 600-second child calls.
- Effective calibration URL: `https://pswww.slac.stanford.edu/calib_ws/`.
  This psana revision appends `/calib_ws/`, so its environment variable holds
  only `https://pswww.slac.stanford.edu`. No TLS or credential changes.
- All six local suites passed. Initial sandbox socket-bind failures and passing
  retries are retained. The parity suite includes a launcher orchestration test.
  [Push CI](https://github.com/monarin/glint/actions/runs/42726133198) and
  [PR CI](https://github.com/monarin/glint/actions/runs/42726143209) passed all six
  jobs, including bridge steps and legacy/current Torch, at the implementation SHA.
- Preflight job 61061513 completed `0:0`, validating dataset layout/calibration,
  MPI, both GPU environments and indexed probe before the full allocation.

Raw records, audits, streams, manifests, environments, activation/launch commands,
failed attempts, scheduler records and independent checks are retained outside
Git history under `~/glint-cleanup/.claude/perlmutter-acceptance/`; final runtime
is `run-61061576/`, independent results are `summary.json` and
`independent-analysis.log`, and review records are in `review/`.

## Limits

This establishes correctness on Perlmutter A100s with the recorded psana,
Torch/CUDA and Cray launcher versions. It does not establish performance equivalence
to the historical L40S/OpenMPI run or any speedup. Read counts cover application
`EventManager` bulk reads, excluding SMD/metadata and physical storage. Diagnostic
hashing adds overhead. Successful child reaping does not newly establish
cancellation during activation; scheduler containment remains required.
Integration and Stage5 GPU producer work remain outside this acceptance.

## Independent review

[Claude round 2](https://github.com/monarin/glint/pull/5#pullrequestreview-5834893546)
reported **PASS** on `a74cf58ad111b712ac5711336b18f8edcb8ddddc` and resolved
CL-01. No blocking findings remain. CL-02 LOC reporting is addressed; optional
CL-03 error wording and CL-04 indexer-name import guards remain deferred.
The reviewer used a detached checkout with file-read tools only and did not run
tests. Its review relied on supplied runtime evidence and inspected source;
the post-run verification reused the unchanged comparator, rather than a second
scientific implementation. A stale CI status note was corrected after review;
the final passing CI snapshots were already included in the reviewed evidence.
This report is an evidence-only addition after the reviewed implementation.

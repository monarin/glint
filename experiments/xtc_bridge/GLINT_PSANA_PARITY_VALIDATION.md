# Historical Stage 4 validation record

A later [1,003-event acceptance matrix](GLINT_PSANA_STAGE4_ACCEPTANCE.md)
completed in one invocation with strict-positive indexing coverage. The
incomplete status below describes only the original 203-event attempts.

Allocated-runtime implementation: `aa7c94163d1221a6b85061cdfe8e0a748e7e9380`.
Numerical parity passes for three cases; **full acceptance remains incomplete**.
No `acceptance-complete.json` was produced. This record does not replace that gate.

## Frozen comparison

The example configuration used mfx100848724 run 51, the first 203 events,
Jungfrau-16M (dense physical segments 0–31), three consensus groups, the existing
refined geometry and lysozyme cell, wavelength 1.322 A, and unchanged V4 defaults.
Both environments ran on two allocated L40S GPUs on `sdfada008`.
The immutable source, frozen configuration, geometry, calibration fingerprint,
timestamps and runtime identities are retained with the artifacts below.

| Check | Batch 1 | Batch 5 | Batch 5, swapped GPUs (retry) |
|---|---:|---:|---:|
| Events / hits | 203 / 14 | 203 / 14 | 203 / 14 |
| Calibration calls, native | 203 | 203 | 203 |
| Maximum absolute q difference | 0 | 0 | 0 |
| Events changing BD owner versus batch 1 | — | 107 | 100 |
| Raw returned-cell frames | 5 | 5 | 5 |
| Strict-gate-passing frames | **0** | **0** | **0** |

All three completed comparisons pass exact calibrated-image hash, peak-array,
record, and stream comparisons. Both BDs participated. Batch 5 exercised a
three-event final tail. The baseline also made 203 calibration calls total;
its existing modulo filter already avoids duplicate calibration work.

Three independent baseline readers returned **21,880,310,949 bytes** from
`EventManager._read`; every native case returned **7,293,436,983 bytes**, with no
overlapping returned intervals. This is one third of the baseline's application
bulk reads. It excludes Cython SMD/metadata reads and does not establish physical
disk traffic, cache behavior or speedup. Hashing and observers add overhead;
phase times are nested, and RSS covers psana-side processes only.

Raw returned-cell events were 10, 26, 66, 92 and 97. None met both the existing
`GATE_MIN=10` and `GATE_FRAC=0.25` criteria using `matched_strict` on the complete
input q array. For example, event 66 had 125 peaks and 20 strict matches, below
25%. A returned known-cell basis is therefore not counted as scientific coverage.
A representative frozen sample with a strict indexing pass is still required.

## Attempts and limitations

- Slurm **43981873**: initial attempt failed because GPU 1 was not visible on
  `sdfada013`. This prompted the early two-GPU preflight.
- Slurm **43984008**: the fixed implementation passed preflight, reference
  creation, all three baselines and the first two native cases. The overall
  1200-second deadline stopped the final swapped case (20:06 total allocation).
  The failure artifact is retained. No tagged workers survived the timeout,
  verified on the node while the subsequent allocation was active.
- Slurm **43986490**: only the unfinished swapped case was rerun using the same
  source snapshot and copied frozen configuration. It completed in 3:26 with a
  600-second subprocess deadline. MPI exited successfully and all BD child-reaped
  flags are true. An additional post-allocation process scan was denied by
  `pam_slurm_adopt`; it is not reported as a passed survivor check.

The assembled comparison combines the two completed original cases with the
isolated retry. It is **not** a successful single invocation of the matrix
coordinator. A future full acceptance run needs sufficient overall time, a
strict-indexing-positive fixture, and the coordinator's final survivor gate.
Keep failed attempts distinct; do not waive the coverage gate or tune production
thresholds to manufacture acceptance.

## Evidence and checks

Shared artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/glint-stage4/`

- `source-aa7c941/`: immutable tracked source snapshot.
- `runtime-43984008/`: original source manifest, frozen inputs, command log,
  baseline/native spools, audits, streams, indexing diagnostics, and failure.
- `swap-43986490/` and `swap-43986490.log`: retry provenance and complete output.
- `review/`: review contexts, responses, test logs and assembled runtime comparison.

Twelve focused checker tests now pass with NumPy 1.26 and 2.3.5. They include complete
artifact fixtures, missing/corrupted outputs, calibration ownership, idle BD
rejection, empty-hit groups and matrix gates. Seventeen Stage-3 regression tests
passed; production code is unchanged. All six GitHub CI checks pass on the
implementation SHA (runs 42657279405 and 42657291146).

After the allocated runs, review fixes raised the default overall deadline to
1800 seconds, added independent NumPy recomputation of strict matches from
spooled q and recorded M, pinned the published gate, cross-checked writer counts,
and factored/tested child row construction. The stronger comparator was rerun
against all three saved outputs and still reports exact parity with zero strict
passes. The final code has not had a new allocated full-matrix run; that remains
blocked on a suitable frozen fixture. The CI runs above apply to the allocated
runtime SHA, not the later checker fixes.

Independent Claude review is recorded separately on PR #4. This evidence-only
record may have a later commit than the implementation SHA above.

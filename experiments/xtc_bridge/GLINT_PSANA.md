# glint-psana: stage-1 launch probe

`glint-psana` is an optional, checkout-local workflow entry point. This first
stage tests one MPI application in a psana environment calling ordinary GLINT
children through the existing envbridge. It does not open XTC, instantiate
DataSource, calibrate detector data, or prove native psana event ownership.
Existing GLINT commands and envbridge are unchanged.

The four-rank harness explicitly represents SMD0/EB as two idle control roles
and BDs as two producer roles. Only producers initialize CUDA and create a
child. Each sends a fixed pair of synthetic q arrays (or an explicit empty
group). A child never imports psana or mpi4py. `echo` verifies exact record
identity; `index` passes a frozen known-cell fixture to GLINT's existing
`index_and_write()`, with integration disabled, and requires a nonempty stream
and at least one indexed frame for a nonempty group. These synthetic lattices
test execution, not realistic detector science or consensus parity.

## Run on an allocated node

Copy `glint_psana_probe.example.json` and verify both activation commands and
the expected child prefix for your site. These preludes are trusted shell code,
not an untrusted configuration format. They must not launch MPI or background
services. The producer prelude activates the environment **before** the MPI
launcher, so `mpirun` and mpi4py must use the same MPI installation.

The S3DF example activates the GLINT prefix directly, avoiding an intermediate
default environment. Its CuPy build requires CUDA-12 NVRTC, while that GLINT
prefix contains CUDA-11 compiler libraries. The child explicitly loads CUDA
libraries and headers from the psana prefix's `targets/x86_64-linux` toolkit
subtree. Its loader path includes that CUDA-only library directory and the
GLINT prefix's library directory; it does not prepend the entire psana `lib`
directory or append the parent's loader path. Python packages still come from
the GLINT environment. Both parent and child report `CUDA_PATH` and
`LD_LIBRARY_PATH`. Replace this toolkit path with your site's compatible CUDA
installation. An echo/placement check
alone does not establish that GLINT's compiled kernels can run: test `index` too.

Within a single-node allocation with two non-MIG GPUs and sufficient CPU slots
for four MPI ranks plus two children:

```bash
python3 experiments/xtc_bridge/glint-psana \
  --config /shared/probe.json --output /shared/new-probe-output \
  -- mpirun -n 4
```

The launcher must run once from a batch script or an `salloc` shell, outside
MPI ranks and numbered `srun` steps. An interactive `srun --pty bash` shell is
already a step and is deliberately refused; its inherited rank/step variables
are included in the error message. The MPI command is supplied as
arguments after `--`; it can instead be a verified `srun` recipe. The output
directory must be new and readable by all ranks on the single probe node.
Do not run the harness on a login node. Scheduler/account/resource options are
site-specific; the example does not reserve resources itself.

All ranks must see the launcher-provided GPU mask. The example maps producer
ranks 2 and 3 to visible devices 0 and 1; rank-local single-device masks require
both ordinals to be 0. Physical PCI identities must still differ. Parents use
CuPy; children select both Torch and CuPy devices before GLINT imports and
check their PCI identity against the parent's. CuPy is required in both
environments for this non-MIG probe. Neither process rewrites
`CUDA_VISIBLE_DEVICES`, and a prelude that changes it fails verification.

`check_gpu=false` permits a CPU-only **echo** test, recording that GPU placement
was not checked. `allow_same_environment=true` permits local bridge tests;
it is not evidence of cross-environment activation. The default requires
different environment prefixes. Choose `mode=index` only with `check_gpu=true`.

## Evidence and failure cases

The output contains per-rank parent identities, per-producer child identities,
and an MPI completion report. `complete.json` is written only by the outer
launcher after every child has been reaped and the MPI launcher exits zero.
It includes Python/NumPy and, when checked, CUDA/CuPy/Torch versions, device
identities, fixture digests, and output counts. Failures produce nonzero status
and diagnostic records without the final success marker. A fresh output
directory prevents stale success markers from another attempt.

Repeat the allocated-node run with these overrides in separate configurations:

| Case | Configuration override | Expected |
| --- | --- | --- |
| Idle producer | `"empty_producers": [2]` | Both children run; one consumes zero frames |
| All empty | `"empty_producers": [2, 3]` | Successful empty participation |
| Failed activation | `"fault": {"rank": 2, "kind": "activation"}` | Bounded startup failure |
| Child exception | `"fault": {"rank": 2, "kind": "exception"}` | Remote exception fails MPI |
| Child crash | `"fault": {"rank": 2, "kind": "crash"}` | Socket failure fails MPI |
| Slow child | `"fault": {"rank": 2, "kind": "timeout"}` | Finite call deadline fails MPI |

Set shorter activation/call deadlines for fault tests. Each producer has one
activation and two bridge calls (initialize and consume). The default 600-second
overall budget exceeds their 120 + 120 + 120-second inner limits and leaves
room for the outer environment/MPI startup and cleanup. The overall deadline
is independent and can intentionally be tighter in a fault test; for normal
runs size it for all these phases. A failed rank aborts MPI instead
of waiting on a success-only collective. Normal errors close/reap its child.
After initialization, the Linux child installs a parent-death SIGKILL guard,
including while it is busy in a call. Before that handshake, envbridge's
activation runs in a separate session: **the scheduler must contain and clean
up the whole job/step**, including activation processes. Test job cancellation
during activation and during a busy call on the actual scheduler, and check
for surviving children on the node. Do not infer that a local process-group
kill cleans up all descendants. Arbitrary unscheduled activation and automatic
restart/resume are outside this prototype's contract.

## Code and tests

- `glint_psana.py`: outside-MPI launch, job deadline and final status.
- `probe_psana_glint.py`: synthetic roles, placement, bridge calls and MPI failure policy.
- `glint_probe_worker.py`: child isolation, device selection and echo/index fixture.
- `glint_probe_common.py`: small CPU-only configuration and fixture helpers.
- `test_glint_psana_probe.py`: CPU contract and real subprocess failure tests.

```bash
cd experiments/xtc_bridge
python test_glint_psana_probe.py
python test_envbridge_local.py
```

The CPU suite cannot establish MPI collective progress, actual GPU placement
or scheduler cancellation. Record those allocated-node results separately.
Later stages add native DataSource processing and compact record/group storage,
then replace CPU calibration with a GpuTask. This probe does not introduce that
production schema or change the indexing algorithm.

## Stage 2: CPU event adapter and compact records

The new modules are independently callable; the `glint-psana` CLI still runs
only the Stage-1 synthetic probe. Stage 2 does not start a serial reader or an
MPI application. Its call path is:

```text
caller supplies Run, event and frozen reference ordinal
  -> CpuEventProcessor.process_event(event, source_ordinal=...)
       -> det.raw.calib(event) once
       -> wavelength validation
       -> xtc_core.prep_geometry (once per processor)
       -> existing V4 finder through xtc_core.frame_q
       -> xtc_core.slab_peaks
       -> validated compact event record
  -> qrecords.to_legacy(complete group, expected_ordinals=..., ...)
       -> existing qframes/events/peaks/lams dictionary
```

`xtc_cpu_processor.py` owns run-level detector, geometry and finder state.
Calibration is on CPU; V4 finding still requires CuPy/GPU. Select the producer's
GPU before processing the first event. The class never creates a DataSource,
sets `PS_PARALLEL`, enumerates events, or applies an ownership modulo filter.
The caller must pass a global source ordinal obtained from the frozen reference
manifest, including non-hits; a BD-local counter is not valid. Group assignment
is `source_ordinal % num_groups`, independent of the producer rank.

A caller that already owns a psana Run can construct the adapter as follows:

```python
from xtc_cpu_processor import CpuEventProcessor

processor = CpuEventProcessor(
    prun, experiment=exp, run_number=run_number,
    producer_rank=rank, num_groups=reference_group_count,
    reference_digest=reference_manifest_sha256,
    calibration_digest=calibration_provenance_sha256,
    det="jungfrau", zdist=0.1, wavelength=1.322,
)
record = processor.process_event(evt, source_ordinal=reference_ordinal)
```

The two input digests must be SHA-256 hashes of the frozen manifest and
calibration provenance (including the selected constants/mask), identical
across producers. They are supplied by the caller, not discovered or verified
against a calibration database by this adapter. Its `config_digest` also hashes
the geometry contents/source, V4 thresholds, wavelength policy, detector/run,
distance and number of consensus groups. Producer rank is excluded.

Only V4 is supported initially. The adapter preserves the serial reader's
calibration-before-wavelength order, energy threshold, bad-pixel mask fallback,
geometry conventions, centroid rounding/clipping and sampled calibrated-pixel
intensities. `xtc_qreader.py` remains unchanged as the reference. Invalid fixed
wavelengths, changed frame shapes and nonfinite output arrays fail explicitly.
Calibration/finder exceptions propagate; they are not converted into non-hits.

`qrecords.py` defines schema version 1. Each record is a plain dictionary with
an exact uint64-range timestamp represented as a Python integer, experiment/run,
source ordinal, group/count, producer rank, configuration digest, status,
calibration-attempt count, wavelength, and detached contiguous float64 `(M,3)`
`q` and `peaks` arrays. Peak rows contain slab fs, slab ss and sampled intensity.
No event, detector or calibrated image crosses this boundary. The existing
NumPy/JSON envbridge codec accepts the dictionary without changes.

Statuses are `hit`, `below_min_peaks`, `missing_calibration` and
`invalid_wavelength`. Every supplied valid event attempts calibration exactly
once. Skipped events retain identity with empty arrays; missing calibration and
invalid wavelength have `wavelength=None`. `peak_count` counts retained q rows,
not pre-threshold detections: the existing finder interface discards those
counts for below-threshold frames.

`write_record`/`read_record` offer atomic single-record NPZ replay/fixture files
using numeric arrays and JSON bytes, with pickle disabled and a configurable
payload limit. They are not the production chunk spool. `to_legacy` accepts one
complete group from any producers, rejects duplicate timestamps/ordinals and
mixed configurations, requires all expected ordinals (including non-hits), and
sorts by source ordinal. Empty groups are valid. Legacy counters retain their
original meanings: `n_events` counts frames that reach finding; `n_skipped_wl`
counts calibrated frames rejected for wavelength. Calibration attempts and
missing calibration remain available on the records.

The group limit is a **compact payload** byte limit, not a process-memory cap.
Dictionary conversion shares the record arrays, while envbridge makes later
copies; callers must budget that overhead. An oversized group fails rather
than being split into scientifically different consensus groups.

Stage 3 will add native psana MPI, the full timestamp-to-ordinal manifest gate,
out-of-range auditing, bounded chunk spools, child consumption and final output
assembly. Stage 2 does not yet verify manifest timestamps against XTC or prove
unique event ownership across MPI workers.

```bash
python experiments/xtc_bridge/test_cpu_records.py
```

These CPU tests reuse the fake detector/geometry pattern of
`test_psana2_geom_thr.py`. They compare against the unchanged serial reader
using real `prep_geometry`, `frame_q` and `slab_peaks` math, with synthetic
calibrated frames and controlled V4 detections. Coverage includes both geometry
sources, panel masks, fixed/event wavelengths, missing calibration, empty and
below-threshold results, sampled intensities, independent producer/group counts,
exact timestamp serialization, existing wire-codec round trips and group gates.
They do not exercise real psana calibration or the V4 CUDA kernel; real-data
scientific equivalence and native MPI no-duplication remain later acceptance gates.

# glint-psana: native MPI CPU-calibration path

`glint-psana` is an optional, checkout-local entry point. `--mode cpu` runs one
native psana MPI application, then passes compact peak/q records to ordinary
GLINT children in a separate environment. Calibration uses `det.raw.calib`;
V4 peak finding and indexing still require GPUs. Integration is disabled.
Existing GLINT commands and vendored envbridge are unchanged.

The default `--mode probe` runs synthetic launch/transport checks instead of
reading XTC. It does not establish native event ownership or scientific parity.

## Requirements

The CPU path supports one offline run on one node, four MPI ranks (SMD0, EB,
two BDs), and two distinct non-MIG GPUs. Roles come from
`DataSource.comms.node_type()`. Reserve CPU slots for four ranks and two children.
All ranks need access to data, geometry and the shared output directory.

Launch once from a batch script or `salloc` shell, outside MPI and numbered
`srun` steps. An `srun --pty bash` shell is already a step and is refused.
Supply `mpirun` or a verified `srun` recipe after `--`. The script does not reserve
resources; run on an allocated compute node, not a login node.

Configuration preludes are trusted shell activation commands; they must not
launch MPI or background services. The psana prelude runs before MPI, so the
launcher and mpi4py must use the same MPI installation. Both environments need
CuPy; the child also needs GLINT and Torch. Keep Python ABI paths consistent.
The CPU example unsets the Python 3.9 `PYTHONPATH` left by site activation before
using Python 3.11 psana, and selects the SLAC calibration endpoint with TLS
verification enabled. A custom checkout may require its own matching path.

The S3DF child example activates GLINT directly and supplies CUDA-12 NVRTC
libraries/headers from the psana prefix's `targets/x86_64-linux` subtree because
GLINT's prefix contains CUDA-11 compiler libraries. Its loader path includes
that CUDA-only directory and GLINT's `lib`, not the entire psana `lib` or inherited
loader path. Python packages still come from GLINT. Replace these paths with
compatible site installations. Echo alone does not verify compiled GLINT kernels.

The device map uses actual BD world ranks and visible CUDA ordinals. All ranks
inherit the allocation's GPU mask. Rank-local masks may require ordinal 0 on both
BDs, but physical PCI identities must differ. Children select Torch and CuPy
devices and verify the parent's physical GPU. Neither process rewrites
`CUDA_VISIBLE_DEVICES`; activation that changes it is rejected.

## Prepare and launch

Freeze a strictly increasing event **prefix** once, outside MPI, in the same
psana environment and with the same calibration endpoint as the CPU run:

```bash
python experiments/xtc_bridge/freeze_psana_reference.py \
  --exp mfx100848724 --run 51 --dir /sdf/data/lcls/ds/prj/public01/xtc \
  --max-events 8 --output /shared/reference.json
```

This separate preparation read stores every timestamp, including non-hits, and
a hash of calibration values (including geometry/masks, excluding database
metadata). It does no finding/indexing. Duplicate/jumping timestamps and short
prefixes fail. Preparation is outside the measured native execution.

Copy `glint_psana_cpu.example.json`, set reference/site paths, then launch:

```bash
python experiments/xtc_bridge/glint-psana --mode cpu \
  --config /shared/cpu.json --output /shared/new-cpu-output -- mpirun -n 4
```

The output directory must be new. The launcher snapshots the reference and
configuration. `source.max_events` is a positive native delivery guard at least
as large as the reference, not an exact serial-prefix selection guarantee.
BDs verify their calibration hashes against the reference before processing.

## Runtime call path

```text
glint-psana -> glint_psana.main -> psana environment -> one MPI launch
  -> psana_mpi_producer.main
     -> produce: DataSource.runs / Run.events on every rank
        -> BD: Reference.ordinal(timestamp)
           -> CpuEventProcessor.process_event
              -> det.raw.calib once -> wavelength/geometry
              -> existing xtc_core.frame_q (V4 peaks and q)
              -> existing slab_peaks -> compact record
           -> qspool.RecordWriter.append / finish
     -> validate_spools: complete reference, unique ownership, group budgets
     -> consume: BD starts ordinary GLINT child through envbridge
        -> load_group across producer spools
        -> glint_qconsumer.index_group -> qrecords.to_legacy
        -> existing glint_xtc.index_and_write
     -> reap children -> finalize: verify group files -> merge atomically
  -> successful MPI exit and verified output -> complete.json
```

This is a two-phase path: exhaust psana production and release run resources
before starting GLINT children. Every rank invokes `Run.close_shared_memory`
when available; older releases without shared windows expose no hook. BDs
synchronize their GPU and release unused CuPy pool blocks; CUDA contexts and
still-live allocations remain. Control ranks must not import CuPy/Torch.
The entry point sets `PS_PARALLEL=mpi`, `PS_EB_NODES=1`, `PS_EB_NODE_LOCAL=0`,
and `PS_SRV_NODES=0` before importing psana.

The event adapter receives a Run and events; it never creates a DataSource,
enumerates events or filters worker ownership. It preserves calibration before
wavelength validation, mask fallback, geometry, V4 thresholds, centroid
rounding/clipping and sampled calibrated intensities. Exceptions, changing frame
shapes and nonfinite q/peak output fail instead of silently dropping rows.
The unchanged `xtc_qreader` is the independent reference.

## Records, grouping and memory

Each record contains an exact uint64-range timestamp as a Python integer,
experiment/run, global ordinal, producer rank, group/count, configuration digest,
status, calibration-attempt count, wavelength, and detached contiguous float64
`(M,3)` q and peak arrays. Peak rows hold slab fs, slab ss and sampled intensity.
No event, detector or calibrated image crosses envbridge.

Statuses are `hit`, `below_min_peaks`, `missing_calibration`, and
`invalid_wavelength`. Every in-range event attempts calibration once. Non-hits
retain identity and empty arrays; missing calibration/invalid wavelength retain
`wavelength=None`. `peak_count` counts retained q rows, not discarded detections.
The config hash includes reference/calibration/geometry, detector, finding and
wavelength settings and group count, but excludes producer rank.

Group assignment is `source_ordinal % num_groups`, independent of BD ownership;
a BD-local counter is invalid. Missing, duplicate, unexpected in-range or
pre-prefix events fail. Post-prefix deliveries are audited without calibration
or finding, up to four times the configured event guard. There is no additional
modulo rejection in the event loop. Group owners are round-robin over actual BDs.

`RecordWriter` writes bounded numeric NPZ chunks with JSON metadata and no pickle,
renames after close, then finalizes producer manifests with counts/hashes. Idle
BDs also write manifests. Rank 0 scans compact chunks to check ownership without
gathering detector images or q arrays through MPI. Consumers load complete groups
across all spools, sort by ordinal and index each in one call. Each group currently
rescans/verifies all chunks; a group-to-chunk index is deferred.

`max_chunk_bytes` and `max_group_bytes` cap compact payload, not RSS. Oversized
records/groups fail rather than changing consensus grouping. Configuration requires
`transport_memory_bytes >= 6*max_group_bytes + 2*max_chunk_bytes` for copies and
chunk loading; this is arithmetic validation, not runtime memory enforcement.
Reserve extra memory for Python, psana, calibration and CUDA workspaces. Loaded
group rows detach from chunk backing arrays.

The experimental single-record `qrecords.write_record` / `read_record` API has
been removed. Use `qspool.RecordWriter` and `iter_records` / `read_chunk` for
fixtures and replay of the same chunk format used at runtime.
`to_legacy` validates complete groups including non-hits and rejects mixed configs
and duplicate identities. Empty groups are valid. Legacy `n_events` counts frames
reaching finding; `n_skipped_wl` counts calibrated frames rejected for wavelength.

## Outputs and failures

Reports record parent/child identities and GPU/environment versions.
`production-complete.json` marks validated spools. Child acknowledgements and
all group files' identities, sizes and hashes must match before merging.
`merged.stream` and `mpi-complete.json` precede the launcher's `complete.json`,
which requires successful MPI exit and matching merged-file hash. Empty groups
produce valid header-only streams. Failures retain diagnostics and return nonzero
without the final success marker.

Children reuse `glint_child_runtime.initialize` for environment/GPU checks and
parent-death protection; they never import psana or mpi4py. Integration is
rejected in configuration and at the child boundary; indexing gets an explicit
argument allowlist with `integrate=False`.

Activation, bridge-call and overall deadlines are finite and independent; size
the overall budget for all phases. A failed rank aborts MPI instead of waiting on
a success-only collective. Normal errors reap children. After initialization,
the Linux parent-death SIGKILL guard also covers busy calls. Before the handshake,
activation runs in a separate session: **the scheduler must contain and clean up
the whole job/step**. Test cancellation during activation and busy calls on the
actual scheduler and check for survivors. A local process-group kill does not
prove descendant cleanup. Arbitrary unscheduled activation and restart/resume
are outside this prototype's contract.

## Synthetic probe

Copy `glint_psana_probe.example.json` and verify activation paths. It uses synthetic
control/producer roles, never a DataSource. `mode=echo` verifies record identity;
`mode=index` with `check_gpu=true` exercises existing indexing on synthetic known-cell
lattices and requires indexed frames for nonempty groups, not detector-science parity.

```bash
python experiments/xtc_bridge/glint-psana --mode probe \
  --config /shared/probe.json --output /shared/new-probe-output -- mpirun -n 4
```

`check_gpu=false` permits CPU-only echo; `allow_same_environment=true` permits local
bridge tests. Neither demonstrates separate-environment GPU execution. Use separate
configs/output directories for these cases, with shorter deadlines for fault tests:

| Override | Expected |
| --- | --- |
| `"empty_producers": [2]` or `[2, 3]` | Successful empty participation |
| `"fault": {"rank": 2, "kind": "activation"}` | Bounded startup failure |
| `"fault": {"rank": 2, "kind": "exception"}` | Child exception fails MPI |
| `"fault": {"rank": 2, "kind": "crash"}` | Socket failure fails MPI |
| `"fault": {"rank": 2, "kind": "timeout"}` | Finite call deadline fails MPI |

## Modules and tests

| Modules | Responsibility |
| --- | --- |
| `glint_psana`, `psana_cpu_config` | Launch, configuration and final status |
| `psana_reference`, `freeze_psana_reference` | Prefix identity and calibration hashes |
| `psana_mpi_producer` | Native lifecycle, production, consumption, finalization |
| `xtc_cpu_processor` | CPU calibration and existing V4/q adapter |
| `qrecords`, `qspool` | Record contract, storage and complete groups |
| `glint_qconsumer` | Ordinary GLINT child and indexing adapter |
| `glint_psana_common`, `glint_child_runtime` | Process/GPU identity, reports and child initialization |
| `glint_probe_common`, `glint_probe_worker` | Probe configuration, fixtures and echo/index child |
| `probe_psana_glint` | Synthetic MPI launch/failure probe |
| `psana_parity_*`, `validate_psana_parity` | Independent scientific acceptance |
| `psana_ab_*`, `benchmark_psana_ab` | Lightweight paired performance trials |

```bash
cd experiments/xtc_bridge
python test_glint_psana_probe.py
python test_cpu_records.py
python test_native_psana_pipeline.py
python test_psana_parity.py
python test_psana_ab.py
python test_envbridge_local.py
```

These tests use fake detectors/finder results and real subprocesses where useful.
They cover geometry/wavelength, timestamps, wire transport, storage, ownership,
group gates and failures. They cannot establish real calibration/V4 CUDA results,
native MPI progress, GPU placement or scheduler cancellation.

## Scientific acceptance and timing

Run the correctness matrix once outside MPI on an allocated two-GPU node:

```bash
python experiments/xtc_bridge/validate_psana_parity.py \
  --config experiments/xtc_bridge/psana_parity.example.json \
  --output /shared/new-parity-run --revision SOURCE_COMMIT --timeout 3000
```

The native cases default to `mpirun -n 4`. To use a verified site launcher,
append it after `--`, for example `-- srun --mpi=cray_shasta --nodes=1
--ntasks=4 --cpus-per-task=16 --gpu-bind=none`. The coordinator must stay outside
MPI and numbered Slurm steps; only the three native cases use this launcher.
Reference freezing and the three independent serial readers stay serial.
Every executed command is recorded in `commands.json`.

The example uses a 1,003-event r51 Jungfrau-16M prefix, three groups, refined geometry,
distance 0.356470 m, wavelength 1.322 A, existing lysozyme cell and default V4 thresholds.
It is a comparison recipe, not new instrument calibration or general indexing-accuracy
evidence. Verify site paths. Each launch is bounded by `job_timeout` (900 seconds in
the example) and the overall deadline (`--timeout` defaults to 1800 seconds). The
recorded matrix took over 20 minutes; reserve 30 minutes plus scheduler/setup margin.

The coordinator freezes geometry/reference and source hashes, runs K independent
serial references, then native batch 1, batch 5, and batch 5 with swapped GPU assignments.
Validation-only source injection gives the old reader the same directory/delivery guard.
It still traverses the full prefix per reader and retains its modulo calibration filter.
Both paths should calibrate N events total; read savings are not calibration savings.
Baseline reconstruction is checked against the unchanged reader's output.

Audits record event identity, calibration calls/full-image hashes, physical segments,
application bulk-read offsets/bytes and phase times. Hashing adds substantial overhead;
intervals are nested, not additive. Reads cover `EventManager._read`, excluding Cython
SMD/metadata, physical storage and cache effects. This is not a speed benchmark.

Validation children seed Python/NumPy/current-device Torch to zero per group, record
precision settings, and test a Python XTC-open guard (not a C-library I/O tracer or
production hook). Strict coverage uses `matched_strict(M,q)` on complete input arrays
and requires `GATE_MIN` and `GATE_FRAC`; returned cells alone are insufficient.
`--min-indexed` defaults to one and must be positive. Matching unindexed output cannot
produce `acceptance-complete.json`.

The comparator requires matching event/status/group/wavelength, exact calibrated-image
diagnostics and peaks, q agreement (`rtol=1e-10, atol=1e-12`), indexed event sets and
equivalent bases (`rtol=1e-5, atol=0.002 A`). Exact stream bytes are also required, a stricter
gate than basis equivalence. Both BDs must participate, ownership must change across cases,
and a partial batch tail must occur. Missing/corrupted output and tagged survivors fail.

See [Stage 4 acceptance](GLINT_PSANA_STAGE4_ACCEPTANCE.md) for the original completed
matrix and exact revision; it does not automatically validate later code changes.
See [A/B instructions](PSANA_AB_BENCHMARK.md) and [observed timings](PSANA_AB_RESULTS.md)
for performance trials and contention caveats. Nonduplicated application reads do not
establish a speedup.

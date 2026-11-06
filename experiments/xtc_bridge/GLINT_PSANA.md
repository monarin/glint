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

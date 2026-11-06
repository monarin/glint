"""CPU contract/subprocess tests; real MPI/GPU probes are a separate acceptance gate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from envbridge import Env, RemoteError, Worker
from glint_probe_common import (RANK_ROLES, fixture, identity, load_config,
                                record_digest, validate_placement)
from glint_psana import inside_rank, launch_command, main as launcher_main
from probe_psana_glint import producer

HERE = Path(__file__).resolve().parent


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "config with spaces.json"
        self.raw = dict(psana_prelude="true", glint_prelude="export PATH=" + shlex.quote(str(Path(sys.executable).parent)) + ':"$PATH"',
                        allow_same_environment=True, activation_timeout=2, call_timeout=2, job_timeout=10)
        self.config = self.configure()
        self.parent = dict(identity(), rank=2, role="producer", gpu=None)

    def configure(self, **changes):
        self.config_path.write_text(json.dumps(dict(self.raw, **changes)))
        return load_config(self.config_path)

    def worker(self):
        worker = Worker(Env(self.config["glint_prelude"], pythonpath=(str(HERE),), inherit_pythonpath=False),
                        timeout=2, call_timeout=2)
        worker.start()
        self.addCleanup(worker.close)
        return worker

    def launch(self, launcher, config=None):
        if config:
            self.configure(**config)
        env = dict(os.environ)
        for key in ("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
            env.pop(key, None)
        return subprocess.run([sys.executable, str(HERE / "glint-psana"), "--config", str(self.config_path),
                               "--output", str(self.root / "out"), "--", *launcher],
                              env=env, capture_output=True, text=True, timeout=20)

    def test_config_rejects_unsupported_or_unbounded_work(self):
        for change in (dict(integrate=True), dict(mode="index"), dict(call_timeout=0),
                       dict(activation_timeout=float("inf")), dict(job_timeout=-1),
                       dict(devices={"2": -1, "3": 0}), dict(empty_producers=[0]),
                       dict(expected_child_prefix=[]), dict(expected_child_prefix="  "),
                       dict(fault={"rank": 0, "kind": "crash"})):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.configure(**change)

    def test_launcher_quotes_arguments_and_starts_one_mpi_command(self):
        command = launch_command(self.config, self.config_path, self.root / "out with spaces", ["mpirun", "-n", "4"])
        self.assertEqual(command[:2], ["bash", "-c"])
        self.assertEqual(shlex.split(command[2].split(" && exec ", 1)[1]),
                         ["mpirun", "-n", "4", "python", str(HERE / "probe_psana_glint.py"),
                          "--config", str(self.config_path), "--output", str(self.root / "out with spaces")])

    def test_fixture_digest_detects_changes_and_bad_alignment(self):
        record = fixture(2)
        expected = record_digest(record)
        record["qframes"][0][0, 0] += 1
        self.assertNotEqual(expected, record_digest(record))
        record["events"].pop()
        with self.assertRaises(ValueError):
            record_digest(record)
        record = fixture(2)
        record["qframes"][0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            record_digest(record)

    def test_placement_rejects_shared_gpu_and_multinode(self):
        records = [dict(rank=r, role=RANK_ROLES[r], host="node", gpu=None if r < 2 else dict(pci=str(r))) for r in range(4)]
        validate_placement(records, True)
        records[3]["gpu"] = records[2]["gpu"]
        with self.assertRaises(ValueError):
            validate_placement(records, True)
        records[3]["host"] = "other"
        with self.assertRaises(ValueError):
            validate_placement(records, False)

    def test_echo_and_empty_producer_reap_children(self):
        for empty in ([], [2]):
            config = self.configure(empty_producers=empty)
            report = producer(self.parent, config, self.root)
            self.assertTrue(report["child_reaped"])
            self.assertFalse(report["child"]["mpi_imported"])
            self.assertFalse(report["child"]["psana_imported"])
            self.assertEqual(report["result"]["n_frames"], 0 if empty else 2)
            self.assertEqual(report["result"]["digest"], record_digest(fixture(2, bool(empty))))

    def test_environment_and_mask_mismatch_fail(self):
        for config, parent in ((dict(self.config, allow_same_environment=False), self.parent),
                               (dict(self.config, expected_child_prefix="/not-this-environment"), self.parent),
                               (self.config, dict(self.parent, cuda_visible_devices="impossible"))):
            with self.subTest(config=config):
                worker = self.worker()
                with self.assertRaises(RemoteError):
                    worker.call("glint_probe_worker:initialize", parent, config)
                worker.close()

    def test_child_requires_initialization_and_valid_record(self):
        worker = self.worker()
        with self.assertRaises(RemoteError):
            worker.call("glint_probe_worker:consume", fixture(2), "echo", "unused")
        worker.call("glint_probe_worker:initialize", self.parent, self.config)
        with self.assertRaises(RemoteError):
            worker.call("glint_probe_worker:consume", {}, "echo", "unused")
        with self.assertRaises(RemoteError):
            worker.call("glint_probe_worker:consume", fixture(2), "index", "unused")

    def test_faults_fail_without_completion(self):
        for kind in ("activation", "exception", "crash", "timeout"):
            with self.subTest(kind=kind):
                config = self.configure(fault=dict(rank=2, kind=kind),
                                        activation_timeout=0.5 if kind == "activation" else 10,
                                        call_timeout=2)
                with self.assertRaises(Exception) as caught:
                    producer(self.parent, config, self.root)
                error = caught.exception
                if kind == "activation":
                    self.assertIn("never connected", str(error))
                elif kind == "exception":
                    self.assertIn("injected GLINT child exception", str(error))
                elif kind == "crash":
                    self.assertIsInstance(error, (EOFError, ConnectionError))
                else:
                    self.assertIsInstance(error, socket.timeout)
            self.assertFalse((self.root / "complete.json").exists())

    def test_launcher_rejects_missing_completion_and_nonzero_exit(self):
        for executable in ("true", "false"):
            if (self.root / "out").exists():
                import shutil
                shutil.rmtree(self.root / "out")
            result = self.launch([executable])
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((self.root / "out/failed.json").exists())
            self.assertFalse((self.root / "out/complete.json").exists())

    def test_launcher_deadline(self):
        start = time.monotonic()
        result = self.launch([sys.executable, "-c", "import time; time.sleep(60)"], dict(job_timeout=0.2))
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - start, 15)
        self.assertFalse((self.root / "out/complete.json").exists())

    def test_nested_launch_rejected(self):
        self.assertFalse(inside_rank(dict(SLURM_PROCID="0", SLURM_STEP_ID="batch")))
        self.assertTrue(inside_rank(dict(SLURM_PROCID="0", SLURM_STEP_ID="0")))
        self.assertTrue(inside_rank(dict(PMIX_RANK="0", SLURM_STEP_ID="batch")))
        result = subprocess.run([sys.executable, str(HERE / "glint-psana"), "--config", str(self.config_path),
                                 "--output", str(self.root / "out"), "--", "true"],
                                env=dict(os.environ, PMI_RANK="0"), capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside MPI", result.stderr)
        self.assertIn("PMI_RANK=0", result.stderr)
        self.assertFalse((self.root / "out").exists())

    def test_late_cancellation_removes_success_marker(self):
        from glint_probe_common import atomic_json
        output = self.root / "out"

        def started(*_args, **_kwargs):
            atomic_json(output / "mpi-complete.json", dict(status="complete", ranks=[{}, {}, {}, {}]))
            process = Mock()
            process.wait.return_value = 0
            return process

        def interrupted_after_rename(path, value):
            atomic_json(path, value)
            if path.name == "complete.json":
                raise InterruptedError("cancelled immediately after success rename")

        with patch("glint_psana.subprocess.Popen", side_effect=started), \
             patch("glint_psana.stop_launcher"), \
             patch("glint_psana.atomic_json", side_effect=interrupted_after_rename), \
             patch.dict(os.environ, {}, clear=True):
            code = launcher_main(["--config", str(self.config_path), "--output", str(output), "--", "true"])
        self.assertEqual(code, 1)
        self.assertFalse((output / "complete.json").exists())
        self.assertTrue((output / "failed.json").exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "parent-death guard is Linux-specific")
    def test_child_dies_when_initialized_parent_is_killed(self):
        # Real separate parent so SIGKILL bypasses its Python finally blocks.
        child_file = self.root / "child-pid.json"
        program = """
import json, os, time
from pathlib import Path
from envbridge import Env, Worker
from glint_probe_common import identity, load_config
c = load_config(os.environ['PROBE_CONFIG'])
w = Worker(Env(c['glint_prelude'], inherit_pythonpath=False), timeout=5)
w.start()
p = dict(identity(), gpu=None)
r = w.call('glint_probe_worker:initialize', p, c)
Path(os.environ['CHILD_FILE']).write_text(json.dumps(r))
time.sleep(60)
"""
        process = subprocess.Popen([sys.executable, "-c", program], cwd=HERE,
                    env=dict(os.environ, PROBE_CONFIG=str(self.config_path), CHILD_FILE=str(child_file), PYTHONPATH=str(HERE)))
        try:
            deadline = time.monotonic() + 10
            while not child_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(child_file.exists())
            pid = json.loads(child_file.read_text())["pid"]
            process.kill()
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] == "Z":
                    break
                time.sleep(0.05)
            else:
                self.fail("initialized GLINT child survived parent SIGKILL")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()

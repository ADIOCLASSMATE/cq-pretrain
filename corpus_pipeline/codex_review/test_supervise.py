"""Bounded supervisor tests: temporary roots and children owned by this test only."""
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from . import supervise


class OwnedChildTests(unittest.TestCase):
    def setUp(self):
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=3)

    def child(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        self.children.append(process)
        return process

    def test_identity_is_stable_and_exact_for_owned_child(self):
        process = self.child()
        identity = supervise.process_identity(process.pid)
        self.assertEqual(identity, supervise.process_identity(process.pid))
        self.assertEqual(identity["pid"], process.pid)
        self.assertTrue(identity["boot_id"])
        self.assertTrue(identity["host"])
        self.assertEqual(len(identity["argv_sha256"]), 64)
        self.assertIsNone(supervise.RecoveredChild(identity).poll())

    def test_matching_recovered_child_forwards_sigterm(self):
        process = self.child()
        recovered = supervise.RecoveredChild(supervise.process_identity(process.pid))
        recovered.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=3), -signal.SIGTERM)
        self.assertEqual(recovered.poll(), 0)

    def test_identity_mismatch_never_signals_even_when_pid_is_live(self):
        process = self.child()
        identity = supervise.process_identity(process.pid)
        for field, value in (("start_ticks", "-1"), ("boot_id", "different-boot"),
                             ("host", "different-host"), ("argv_sha256", "0" * 64)):
            with self.subTest(field=field):
                recovered = supervise.RecoveredChild({**identity, field: value})
                self.assertEqual(recovered.poll(), 0)
                with mock.patch.object(supervise.os, "kill") as kill:
                    recovered.send_signal(signal.SIGTERM)
                    kill.assert_not_called()
                self.assertIsNone(process.poll())


class FakeChild:
    pid = 999999999

    def __init__(self):
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        self.returncode = -signum


class SignalBoundaryTests(unittest.TestCase):
    """Patch signals and process creation; never signal an existing real process."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.handlers = {}
        self.writes = []
        self.child = FakeChild()
        self.identity = {"pid": self.child.pid, "host": "fake", "boot_id": "fake",
                         "start_ticks": "1", "argv_sha256": "fake"}

    def tearDown(self):
        self.tmp.cleanup()

    def install(self, stack, *, spawn=None, save=None, sleep=None):
        stack.enter_context(mock.patch.object(sys, "argv", ["supervise", "--root", str(self.root)]))
        stack.enter_context(mock.patch.object(supervise.signal, "signal", side_effect=lambda sig, callback: self.handlers.__setitem__(sig, callback)))
        stack.enter_context(mock.patch.object(supervise, "counts", return_value={"pending": 1}))
        stack.enter_context(mock.patch.object(supervise, "process_identity", return_value=self.identity))
        stack.enter_context(mock.patch.object(supervise.subprocess, "Popen", side_effect=spawn or (lambda *a, **k: self.child)))
        stack.enter_context(mock.patch.object(supervise, "atomic_json", side_effect=save or (lambda path, value: self.writes.append((path, value)))))
        stack.enter_context(mock.patch.object(supervise.time, "sleep", side_effect=sleep or (lambda _: self.handlers[signal.SIGTERM](signal.SIGTERM, None))))

    def test_sigterm_handler_drains_live_child(self):
        with ExitStack() as stack:
            self.install(stack)
            supervise.main()
        self.assertEqual(self.child.signals, [signal.SIGTERM])
        self.assertEqual(self.writes[-1][1]["status"], "interrupted_drained")

    def test_sigterm_during_popen_is_forwarded_after_assignment(self):
        def spawning(*args, **kwargs):
            self.handlers[signal.SIGTERM](signal.SIGTERM, None)
            self.assertEqual(self.child.signals, [])
            return self.child
        with ExitStack() as stack:
            self.install(stack, spawn=spawning)
            supervise.main()
        self.assertEqual(self.child.signals, [signal.SIGTERM])
        self.assertEqual(self.writes[-1][1]["status"], "interrupted_drained")

    def test_sigterm_after_adoption_before_loop_drains_recovered_child(self):
        """Known boundary: the adopted child is not yet in the signal closure."""
        (self.root / "SUPERVISOR_STATUS.json").write_text(json.dumps({"child_identity": self.identity, "action": "run"}))
        self.child.identity = self.identity
        def saving(path, value):
            self.writes.append((path, value))
            if path.name == "SUPERVISOR.json":
                self.handlers[signal.SIGTERM](signal.SIGTERM, None)
        with ExitStack() as stack:
            self.install(stack, save=saving)
            stack.enter_context(mock.patch.object(supervise, "RecoveredChild", return_value=self.child))
            supervise.main()
        self.assertEqual(self.child.signals, [signal.SIGTERM])
        self.assertEqual(self.writes[-1][1]["status"], "interrupted_drained")


if __name__ == "__main__":
    unittest.main()

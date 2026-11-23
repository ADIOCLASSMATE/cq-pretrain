"""Leaving priority 4 at the cheapest checkpoint boundary (scripts/priority_controller.py)."""

from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import priority_controller as pc


class LeavePlanTests(unittest.TestCase):
    def test_table(self):
        cases = [
            # step, checkpoint, banked_after, every, busy -> plan
            ((310, 300, 300, 50, False), (False, 350)),  # nothing banked by this job yet
            ((349, 300, 300, 50, False), (False, 350)),
            ((350, 350, 300, 50, False), (True, None)),  # just banked: leave
            ((360, 350, 300, 50, False), (True, None)),  # 10 behind, 40 ahead: leave
            ((374, 350, 300, 50, False), (True, None)),
            ((375, 350, 300, 50, False), (False, 400)),  # halfway: let the next one land
            ((380, 350, 300, 50, False), (False, 400)),
            ((400, 350, 300, 50, False), (False, 400)),  # its save is being written
            ((500, 500, 300, 50, True), (False, 500)),  # validation of 500 still running
            ((330, 300, -1, 100, False), (True, None)),  # the running job saves every 100
            ((359, 300, -1, 100, False), (False, 400)),
            ((1, -1, -1, 50, False), (False, 50)),  # fresh run, save-first still writing
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(pc.leave_plan(*arguments), expected)


class FakeRun:
    """A run directory with the files the controller reads."""

    def __init__(self, root, every=50, validate_every=500):
        self.root = Path(root)
        (self.root / "train").mkdir()
        (self.root / "checkpoints").mkdir()
        (self.root / "priority").mkdir()
        arguments = dict(save_every=every, validate_every=validate_every)
        (self.root / "train/run.json").write_text(json.dumps(dict(arguments=arguments)))
        (self.root / "train/steps.jsonl").write_text("")

    def steps(self, *steps, seconds=20.0):
        with (self.root / "train/steps.jsonl").open("a") as handle:
            for step in steps:
                handle.write(json.dumps(dict(step=step, seconds=seconds, loss=1.0)) + "\n")

    def checkpoint(self, step):
        (self.root / "checkpoints/latest.json").write_text(json.dumps(dict(step=step)))

    def validated(self, step):
        report = self.root / f"train/validation/step_{step:07d}"
        report.mkdir(parents=True)
        (report / "results.json").write_text("{}")


class ControllerFixture(unittest.TestCase):
    """A controller wired to a fake run directory, with no platform calls."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fake = FakeRun(self.temporary.name)
        self.clock = [0.0]
        self.sleeps = []
        self.room = [9]
        self.stopped = []
        patches = [
            patch.object(pc, "RUN", self.fake.root),
            patch.object(pc, "HOME", self.fake.root / "priority"),
            patch.object(pc, "free_nodes", lambda: self.room[0]),
            patch.object(pc, "needed_for_switch", lambda: 5),
            patch.object(pc.time, "monotonic", lambda: self.clock[0]),
            patch.object(pc.time, "sleep", self.sleep),
            patch.object(pc.Controller, "stop", lambda controller, reason: self.stopped.append(reason)),
            patch.object(pc.Controller, "start_job", lambda controller: None),
            patch("builtins.print"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temporary.cleanup)
        self.on_sleep = lambda: None

    def sleep(self, seconds):
        self.clock[0] += seconds
        self.sleeps.append(seconds)
        self.on_sleep()

    def controller(self, banked_after):
        controller = pc.Controller()
        submitted = (pc.now() - timedelta(hours=1)).isoformat()
        controller.state["job"] = dict(name="job", priority=4, submitted=submitted, checkpoint_at_submit=banked_after)
        return controller

    def poll(self, controller, times=1):
        for _ in range(times):
            controller.maybe_leave_high(controller.state["job"])


class ControllerTests(ControllerFixture):
    def test_checkpoint_between_the_two_room_polls_leaves_at_once(self):
        self.fake.checkpoint(300)
        self.fake.steps(*range(330, 349))
        controller = self.controller(banked_after=250)
        self.poll(controller)
        self.fake.checkpoint(350)
        self.fake.steps(349, 350, 351, 352)
        self.poll(controller)
        self.assertEqual(self.stopped, ["switch to LOW"])

    def test_waits_for_its_own_first_checkpoint(self):
        self.fake.checkpoint(300)
        self.fake.steps(*range(301, 311))
        controller = self.controller(banked_after=300)
        self.poll(controller, times=3)
        self.assertEqual(self.stopped, [])
        self.assertEqual(controller.state["switch"]["target"], 350)
        self.assertEqual(self.sleeps, [])  # 40 updates away: no watching, next poll decides

    def test_far_past_the_checkpoint_waits_for_the_next_and_leaves_right_after_it(self):
        self.fake.checkpoint(350)
        self.fake.steps(*range(351, 381))
        controller = self.controller(banked_after=300)
        self.poll(controller, times=2)
        self.assertEqual((self.stopped, controller.state["switch"]["target"]), ([], 400))
        self.fake.steps(*range(381, 397))

        def train():
            self.fake.steps(397, 398, 399, 400)
            self.fake.checkpoint(400)

        self.on_sleep = train
        self.poll(controller)
        self.assertEqual(self.stopped, ["switch to LOW"])
        self.assertEqual(self.sleeps, [pc.WATCH])

    def test_validation_after_the_checkpoint_finishes_first(self):
        self.fake.checkpoint(500)
        self.fake.steps(*range(480, 501))
        controller = self.controller(banked_after=450)
        self.on_sleep = lambda: self.fake.validated(500)
        self.poll(controller, times=2)
        self.assertEqual(self.stopped, ["switch to LOW"])
        self.assertEqual(self.sleeps, [pc.WATCH])

    def test_room_gone_while_watching_stays(self):
        self.fake.checkpoint(350)
        self.fake.steps(*range(351, 398))
        controller = self.controller(banked_after=300)

        def train():
            self.fake.steps(398, 399, 400)
            self.fake.checkpoint(400)
            self.room[0] = 3

        self.on_sleep = train
        self.poll(controller, times=2)
        self.assertEqual(self.stopped, [])
        self.assertNotIn("switch", controller.state)

    def test_restart_in_place_counts_from_the_resumed_checkpoint(self):
        self.fake.checkpoint(350)
        self.fake.steps(*range(351, 381))  # 380 trained, then the attempt failed
        controller = self.controller(banked_after=300)
        past = pc.now().timestamp() - 60
        os.utime(self.fake.root / "train/steps.jsonl", (past, past))
        os.utime(self.fake.root / "train/run.json")  # the new attempt has started
        self.assertEqual(pc.training_progress(controller.state["job"])["step"], 350)
        self.poll(controller, times=2)
        self.assertEqual(self.stopped, ["switch to LOW"])

    def test_files_of_an_earlier_job_are_ignored(self):
        self.fake.checkpoint(350)
        self.fake.steps(*range(351, 361))
        old = (pc.now() - timedelta(hours=2)).timestamp()
        for name in ("train/run.json", "train/steps.jsonl"):
            os.utime(self.fake.root / name, (old, old))
        controller = self.controller(banked_after=300)
        self.assertIsNone(pc.training_progress(controller.state["job"]))
        self.poll(controller, times=2)
        self.assertEqual(self.stopped, [])


class LastRowsTests(unittest.TestCase):
    def test_skips_cut_and_partial_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "steps.jsonl"
            rows = [json.dumps(dict(step=step, pad="x" * 50)) for step in range(100)]
            path.write_text("\n".join(rows) + '\n{"step": 10')
            got = pc.last_rows(path, 5, window=600)
            self.assertEqual([row["step"] for row in got], [95, 96, 97, 98, 99])


class RestartLoopTests(ControllerFixture):
    """A priority-4 job the platform keeps restarting must stop costing points.

    The platform retries a failed job in place instead of ending it, so a
    deterministic failure -- a record the resumed data cursor reaches again on
    every attempt -- never reaches the crash accounting in close(). On
    2026-11-23 that loop ran five restarts in 90 minutes with the checkpoint
    stuck at step 800.
    """

    def setUp(self):
        super().setUp()
        self.room = [0]  # no room for LOW, so leaving priority 4 is never considered
        self.statuses = []
        item = patch.object(pc.Controller, "status", lambda controller, name: (self.statuses.pop(0), {}))
        item.start()
        self.addCleanup(item.stop)

    def running_job(self):
        controller = self.controller(banked_after=-1)
        controller.state["job"]["last_running"] = pc.now().isoformat()
        controller.state["job"]["running_since"] = pc.now().isoformat()
        return controller

    def restart(self, controller):
        """One platform retry: the job leaves RUNNING, then comes back."""
        self.statuses.extend(["QUEUING", "RUNNING"])
        controller.cycle()
        controller.cycle()

    def test_restarts_without_a_checkpoint_halt_the_controller(self):
        self.fake.checkpoint(800)
        controller = self.running_job()
        for _ in range(pc.STALLED_RESTARTS - 1):
            self.restart(controller)
        self.assertEqual(controller.state["job"]["restarts"], pc.STALLED_RESTARTS - 1)
        self.statuses.append("QUEUING")
        with self.assertRaises(SystemExit) as caught:
            controller.cycle()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("restarting without committing a checkpoint", self.stopped)
        attention = json.loads((self.fake.root / "priority/ATTENTION.json").read_text())
        self.assertEqual(attention["checkpoint"], 800)
        self.assertEqual(attention["restarts"], pc.STALLED_RESTARTS)

    def test_a_restart_that_banked_work_starts_a_fresh_streak(self):
        """Ordinary node failures must never accumulate into a halt."""
        self.fake.checkpoint(800)
        controller = self.running_job()
        for step in (850, 900, 950, 1000):
            self.restart(controller)
            self.assertEqual(controller.state["job"]["restarts"], 1)
            self.fake.checkpoint(step)
        self.assertFalse((self.fake.root / "priority/ATTENTION.json").exists())

    def test_a_low_priority_job_is_left_to_the_ordinary_down_path(self):
        """LOW is free and preemptible; repeated requeues there are expected."""
        self.fake.checkpoint(800)
        controller = self.running_job()
        controller.state["job"]["priority"] = 3
        for _ in range(pc.STALLED_RESTARTS + 2):
            self.statuses.extend(["RUNNING"])
            controller.cycle()
            controller.state["job"]["left_running"] = pc.now().isoformat()
            controller.state["job"]["restarts"] = controller.state["job"].get("restarts", 0) + 1
        self.assertFalse((self.fake.root / "priority/ATTENTION.json").exists())


if __name__ == "__main__":
    unittest.main()

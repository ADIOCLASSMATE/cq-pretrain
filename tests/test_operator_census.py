"""The operator census attributes dispatches to this repository's lines (ops/census.py)."""

import json
import tempfile
import unittest
from pathlib import Path

import ops.census as census


def issue_two_adds(counter):
    for _ in range(2):
        counter.record("add.Tensor")  # one line, so both land on one site


# Code compiled under a filename outside the repository stands in for the
# library and autograd-engine frames that sit between a line here and a dispatch.
ELSEWHERE = compile("counter.record('mm.default')", "<generated>", "exec")
ELSEWHERE_FUNCTION = compile("def run():\n    counter.record('mm.default')\n", "<generated>", "exec")


def issue_indirectly(counter):
    exec(ELSEWHERE, {"counter": counter})


class SiteAttributionTests(unittest.TestCase):
    def test_a_call_site_in_this_repository_is_named_with_its_line(self):
        counter = census.OperatorCensus()
        counter.record("empty_strided.default")
        site = next(iter(counter.sites))
        path, line = site.rsplit(":", 1)
        self.assertEqual(path, "tests/test_operator_census.py")
        self.assertTrue(line.isdigit())

    def test_the_census_module_is_never_the_reported_site(self):
        counter = census.OperatorCensus()
        counter.record("empty.memory_format")
        self.assertNotIn("ops/census.py", next(iter(counter.sites)))

    def test_separate_call_sites_stay_separate(self):
        counter = census.OperatorCensus()
        counter.record("mul.Tensor")
        issue_two_adds(counter)
        self.assertEqual(len(counter.sites), 2)
        self.assertEqual(counter.ops["add.Tensor"], 2)
        self.assertEqual(max(counter.sites.values()), 2)

    def test_a_stack_with_no_repository_frame_reports_the_engine(self):
        counter = census.OperatorCensus()
        # A thread whose whole target is defined outside the repository has no
        # repository frame anywhere beneath it, which is what the autograd
        # engine's own thread looks like from Python.
        import threading

        namespace = {"counter": counter}
        exec(ELSEWHERE_FUNCTION, namespace)
        thread = threading.Thread(target=namespace["run"])
        thread.start()
        thread.join()
        self.assertEqual(list(counter.sites), ["<engine>"])
        self.assertEqual(counter.direct["<engine>"], 0)

    def test_a_call_written_on_the_line_is_separated_from_one_issued_beneath_it(self):
        counter = census.OperatorCensus()
        counter.record("mul.Tensor")
        issue_indirectly(counter)
        rows = {row["site"]: row for row in counter.report()["by_site"]}
        written = [row for row in rows.values() if row["written_on_this_line"]]
        beneath = [row for row in rows.values() if row["issued_beneath_it"]]
        self.assertEqual([row["calls"] for row in written], [1])
        self.assertEqual([row["calls"] for row in beneath], [1])
        # The indirect one still names a repository line: it is where the work
        # was entered, which is exactly why it must not be read as the culprit.
        self.assertTrue(beneath[0]["site"].startswith("tests/test_operator_census.py:"))


class ReportTests(unittest.TestCase):
    def counted(self):
        counter = census.OperatorCensus()
        counter.steps = 2
        for _ in range(6):
            counter.record("empty_strided.default")
        for _ in range(4):
            counter.record("_local_scalar_dense.default")
        return counter

    def test_counts_are_reported_per_step(self):
        report = self.counted().report()
        self.assertEqual(report["total_dispatches"], 10)
        self.assertEqual(report["dispatches_per_step"], 5)
        self.assertEqual(report["by_operator"][0], dict(operator="empty_strided.default", calls=6, per_step=3))

    def test_device_to_host_syncs_are_called_out_separately(self):
        report = self.counted().report()
        self.assertEqual(report["device_to_host_syncs_per_step"], 2)
        self.assertEqual([row["operator"] for row in report["device_to_host_syncs"]], ["_local_scalar_dense.default"])

    def test_the_report_is_written_as_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.counted().write(Path(temporary) / "nested" / "census.json")
            self.assertEqual(json.loads(Path(path).read_text())["total_dispatches"], 10)


class InstallationTests(unittest.TestCase):
    def test_disabled_counts_nothing_and_leaves_the_step_count_alone(self):
        counter = census.OperatorCensus()
        with census.census(counter, False):
            counter_steps = counter.steps
        self.assertEqual(counter_steps, 0)

    def test_no_counter_is_accepted(self):
        with census.census(None, True):
            pass

    def test_an_uninstallable_mode_does_not_stop_training(self):
        counter = census.OperatorCensus()
        broken = lambda: (_ for _ in ()).throw(ImportError("no dispatch mode"))
        original = census._mode_class
        census._mode_class = broken
        try:
            with census.census(counter, True):
                pass
        finally:
            census._mode_class = original
        self.assertIn("<census unavailable: ImportError>", counter.ops)


if __name__ == "__main__":
    unittest.main()

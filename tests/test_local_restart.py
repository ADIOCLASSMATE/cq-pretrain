"""Local launcher restart tests."""
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest


def cohort_worker(root, rank, output):
    from train_production import join_attempt
    output.put((rank, join_attempt(root, rank, 2, timeout=8, interval=.02)))



class LocalRestartTests(unittest.TestCase):
    def test_restarted_workers_do_not_accept_stale_generation(self):
        context = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = context.Queue()
            previous = None
            for _ in range(2):
                worker = context.Process(target=cohort_worker, args=(root,1,output))
                worker.start()
                time.sleep(.15)
                leader = context.Process(target=cohort_worker, args=(root,0,output))
                leader.start()
                values = [output.get(timeout=10), output.get(timeout=10)]
                worker.join(10);leader.join(10)
                self.assertEqual(worker.exitcode,0)
                self.assertEqual(leader.exitcode,0)
                self.assertEqual(values[0][1],values[1][1])
                self.assertNotEqual(values[0][1],previous)
                previous = values[0][1]
            output.close()

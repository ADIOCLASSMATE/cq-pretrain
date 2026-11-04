import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import sscd_npu_pool as pool


class PoolFailurePersistenceTests(unittest.TestCase):
    def check_persisted_failure(self, *, update_during_wait=False):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name); directory = output / 'fleet/pools'
            state = directory / 'audit.json'
            pool.atomic_json(state, {'failures': {'0': 0 if update_during_wait else 2}})
            if not update_during_wait:
                pool.atomic_json(directory / 'audit.device0.last_exit.json', {'failures': 3, 'returncode': 1})
            def acquire(*args):
                if update_during_wait:
                    pool.atomic_json(state, {'failures': {'0': 3}, 'fatal': True})
                return True
            with patch.dict(os.environ, {'RESOURCE_MANAGER_ALLOCATION_ID': 'audit', 'RESOURCE_MANAGER_CARDS': '1'}), \
                 patch.object(pool, 'qualified', return_value=True), \
                 patch.object(pool, 'complete', return_value=False), \
                 patch.object(pool, 'await_allocation_lock', side_effect=acquire), \
                 patch.object(pool.signal, 'signal'), \
                 patch.object(pool.subprocess, 'Popen', side_effect=AssertionError('fatal worker must not restart')):
                self.assertEqual(pool.run(output, '/not-executed.sh'), 2)
            result = pool.read(state)
            self.assertEqual(result['failures']['0'], 3)
            self.assertTrue(result['fatal'])
            self.assertEqual(result['status'], 'await_worker_repair')

    def test_failure_written_while_waiting_for_inherited_lock_is_not_lost(self):
        self.check_persisted_failure(update_during_wait=True)

    def test_last_exit_committed_before_interrupted_heartbeat_is_not_lost(self):
        self.check_persisted_failure()


if __name__ == '__main__':
    unittest.main()

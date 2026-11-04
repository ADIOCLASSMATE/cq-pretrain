import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from . import sscd_cluster_runner as s

class ThroughputTests(unittest.TestCase):
    def test_five_minutes_required_and_phase_specific(self):
        first={'at':100.,'phase':'catalog','completed':0,'total':1000,'unit':'vectors'}
        stats={'completed_vectors':600,'total_vectors':1000}
        self.assertIsNone(s.throughput_estimate([first],'catalog',stats,399.,1000.)['throughput_units_per_second'])
        value=s.throughput_estimate([first],'catalog',stats,400.,1000.)
        self.assertEqual(value['throughput_units_per_second'],2.)
        self.assertEqual(value['phase_remaining'],400)
        self.assertEqual(value['phase_eta_epoch'],600.)
        self.assertIsNone(s.throughput_estimate([first],'build',stats,400.,1000.)['throughput_units_per_second'])

    def test_overdue_never_implies_stop_or_infinite_json(self):
        value=s.throughput_estimate([{'at':100.,'phase':'search','completed':0,'total':1000}],
            'search',{'completed_queries':600,'total_queries':1000},400.,300.)
        self.assertTrue(value['deadline_overdue'])
        self.assertFalse(value['deadline_is_stop_condition'])
        self.assertIsNone(value['required_speed_to_meet_deadline'])
        self.assertAlmostEqual(value['required_speed_to_finish_within_one_hour'],400/3600)
        json.dumps(value,allow_nan=False)

    def test_stationary_and_counter_reset_do_not_invent_eta(self):
        stats={'completed_queries':5,'total_queries':100}
        for before in (5,50):
            result=s.throughput_estimate([{'at':10.,'phase':'search','completed':before,'total':100}],
                'search',stats,400.,1000.)
            self.assertIsNone(result['phase_eta_epoch'])

class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.cp=self.root/'config.json'
        self.cp.write_text('{}')
        self.c={'root':str(self.root),'output':str(self.root/'output'),'notebooks':['owned-cpu'],
                'max_instances':1,'deadline_epoch':1.0,'runtime':str(self.root),'threads':3}
        with patch.object(s,'config',return_value=self.c):self.co=s.Coordinator(self.cp)
    def tearDown(self):self.temp.cleanup()

    def test_finalized_complete_breaks_wait_when_ready_false(self):
        value={'complete':True,'ready_to_finalize':False,'completed_vectors':7,'total_vectors':7}
        with patch('corpus_pipeline.postprocess.sscd_parallel_catalog.status',return_value=value),\
             patch.object(s.time,'sleep',side_effect=AssertionError('must not wait after complete')):
            self.co.parallel('catalog')
        rows=[json.loads(x) for x in (self.co.d/'history.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['completed'],7)
        self.assertEqual(self.co.remaining_work['catalog']['remaining'],0)
        self.assertTrue(s.read(self.co.d/'PROGRESS.json')['forecast']['deadline_overdue'])

    def test_explicit_stop_only_deadline_and_failed_worker_do_not_abort(self):
        d=self.co.d/'workers'; d.mkdir(); s.save(d/'owned-cpu.json',{'fatal':True,'worker_id':'owned-cpu'})
        self.co.check()
        self.assertEqual(len(self.co.worker_warnings),1)
        (self.co.d/'STOP').touch()
        with self.assertRaises(InterruptedError):self.co.check()

    def test_monitor_failure_recorded_without_terminating_active_work(self):
        def fail(name):
            self.co.monitor_stop.set()
            raise OSError('temporary API outage')
        with patch.object(self.co,'remote',side_effect=fail),patch.object(s,'spawn',side_effect=AssertionError('no late spawn')):
            self.co.monitor()
        self.assertEqual(self.co.monitor_errors['owned-cpu']['error_type'],'OSError')
        self.co.check()

    def test_stopping_monitor_does_not_touch_platform(self):
        self.co.monitor_stop.set()
        with patch.object(s.subprocess,'run',side_effect=AssertionError('no platform call')):
            self.co.remote('owned-cpu')

    def test_fatal_remote_worker_is_not_blindly_relaunched(self):
        s.save(self.co.d/'workers'/'owned-cpu.json',
               {'fatal':True,'worker_id':'owned-cpu','updated_at':0,'status':'failed'})
        with patch.object(s.subprocess,'run',side_effect=AssertionError('fatal worker must await repair')):
            self.co.remote('owned-cpu')

    def test_fatal_local_worker_is_not_blindly_relaunched(self):
        s.save(self.co.d/'workers'/'coordinator-local.json',
               {'fatal':True,'worker_id':'coordinator-local','updated_at':0,'status':'failed'})
        with patch.object(self.co,'remote',return_value=None),\
             patch.object(self.co.monitor_stop,'wait',side_effect=lambda _: self.co.monitor_stop.set()),\
             patch.object(s,'spawn',side_effect=AssertionError('fatal worker must await repair')):
            self.co.monitor()

    def test_partial_history_tail_can_resume_and_host_guard_fails_before_lock(self):
        (self.co.d/'history.jsonl').write_text('{"at":1,"phase":"catalog"}\n{"interrupted":')
        with patch.object(s,'config',return_value={**self.c,'coordinator_host':'different-host'}):
            resumed=s.Coordinator(self.cp)
        self.assertEqual(len(resumed.history),1)
        with self.assertRaisesRegex(ValueError,'configured host'):resumed.run()
        self.assertFalse((self.co.d/'COORDINATOR.lock').exists())

if __name__=='__main__':unittest.main()

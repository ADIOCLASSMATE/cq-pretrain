"""CPU-only checks for progress accuracy and stopped-process detection."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from scripts.monitor_latents import Monitor, process_matches, progress_rows, seconds_per_pair


class MonitorTests(unittest.TestCase):
    def test_rate_excludes_replay_and_startup(self):
        rows=[]
        for i,(seconds,encoded) in enumerate([(0,0),(1,0),(2,0),(80,1),(85,2),(90,3),(95,4)],1):
            rows.append(dict(completed=i,elapsed_seconds=seconds,encoded=dict(qwen=encoded,wan=encoded)))
        self.assertEqual(seconds_per_pair(rows),5)
        self.assertIsNone(seconds_per_pair(rows[:3]))

    def test_reused_pid_and_zombie_are_not_alive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'42';p.mkdir()
            (p/'cmdline').write_bytes(b'python\0-m\0scripts.prepare_latents\0/data/samples.jsonl\0')
            (p/'stat').write_text('42 (python) S 1 2 3')
            self.assertTrue(process_matches(42,'scripts.prepare_latents','/data/samples.jsonl',root))
            self.assertFalse(process_matches(42,'scripts.prepare_openvid81',proc_root=root))
            (p/'stat').write_text('42 (python) Z 1 2 3')
            self.assertFalse(process_matches(42,'scripts.prepare_latents',proc_root=root))

    def test_logs_respect_restart_offset_and_partial_last_line(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'log'
            old='Traceback (most recent call last):\n'
            p.write_text(old+'{"sample_id":"x","completed":1}\n{"sample_id":')
            rows,errors=progress_rows(p,len(old.encode()))
            self.assertEqual(rows,[{'sample_id':'x','completed':1}])
            self.assertEqual(errors,[])

    def test_counts_active_profile_only_and_stopped_hides_eta(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'metadata/latent_validation';state.mkdir(parents=True)
            stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
            (state/'latest_restart.json').write_text(json.dumps({'pid':99999999,'started_utc':stamp}))
            (state/'full_status.json').write_text(json.dumps({'pid':99999999,'devices':[0],'phase':'extracting','complete':False}))
            profile={'video':{'mode':'native'},'encoders':{'qwen':{'weights':'q'},'wan':{'weights':'w'}}}
            for res in ('832x480','960x544'):
                d=root/res;d.mkdir();(d/'profile.json').write_text(json.dumps(profile))
                rows=[{'sample_id':str(i),'latent_profile':'profile.json'} for i in range(3)]
                (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                for kind in ('qwen','wan'):
                    branch=hashlib.sha256(json.dumps({'encoder':profile['encoders'][kind],'preprocessing':profile['video']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
                    p=d/'latents'/kind/branch/'aa';p.mkdir(parents=True)
                    (p/'paired.pt').touch()
                    if kind=='qwen':(p/'qwen_only.pt').touch()
                    extra=d/'latents'/kind/'old-profile'/'aa';extra.mkdir(parents=True);(extra/'extra.pt').touch()
            with patch('scripts.monitor_latents.process_matches',return_value=False):
                result=Monitor(root).collect()
            self.assertEqual(result['state'],'stopped')
            self.assertIsNone(result['eta'])
            self.assertEqual(result['paired'],2)
            self.assertEqual(result['total'],6)
            for row in result['resolutions']:
                self.assertEqual((row['qwen'],row['wan'],row['indexed']),(2,1,0))


if __name__=='__main__':
    unittest.main()

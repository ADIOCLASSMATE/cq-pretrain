"""Fresh-run integration and incident regressions. All compute is offline/fake."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from corpus_pipeline.production.common import atomic_json, file_sha
from corpus_pipeline.unified.genund_loose import BLIP, prepare_blip_units
from corpus_pipeline.unified.test_genund_loose import image, DATASET
from corpus_pipeline.unified.parallel_ingest import worker, finalize_source, registered
from corpus_pipeline.unified import workflow, fleet
from corpus_pipeline.coordination import locked, available_lock
from .config import SCHEMA, SOURCES, check_config
from .prepare import prepare, verify_runtime, launcher
from .ingestion import register
from .normalization import adopt_native, ingest_integrity_verified
from .controller import prepare_source, review_action
from . import cpu


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        mount=root/'mount';image(mount/BLIP/'raw/a.jpg')
        (mount/BLIP/'raw/a.txt').write_text('Original complete caption')
        plans={}
        for name in SOURCES:
            target=root/'plans'/name
            prepare_blip_units(mount,target,dataset_id=DATASET)
            plans[name]=str(target/'PLAN.json')
        for name in ('frontpass','sscd'):(root/(name+'.json')).write_text('{}')
        (root/'key').write_text('fixture-not-a-real-key')
        config={'schema':SCHEMA,'run':str(root/'run'),'python':sys.executable,'sdk_python':sys.executable,
            'account':'fixture','workspace':'fixture','deadline_utc':'2099-01-01T00:00:00Z',
            'max_cards':512,'name_prefix':'fixture','queue_port_base':28001,
            'review_notebook':'review-cpu','api_key_file':str(root/'key'),
            'frontpass_qualification':str(root/'frontpass.json'),'sscd_qualification':str(root/'sscd.json'),
            'baseline_index':str(root/'baseline'),'baseline_reconciliation':str(root/'baseline-recon'),
            'model_assets':str(root/'models'),'dataset_roots':{DATASET:str(mount)},'source_plans':plans,
            'platform':{'account':'fixture','workspace':'fixture','project':'fixture','group':'fixture','image':'fixture',
                        'datasets':['fixture:01']},
            'cpu_workers':[{'name':'ingest-cpu','notebook_id':'cpu1','workspace_id':'ws','roles':['ingest'],'workers':1},
                           {'name':'review-cpu','notebook_id':'cpu2','workspace_id':'ws','roles':['ingest','review'],'workers':1}]}
        path=root/'request.json';atomic_json(path,config)
        result=prepare(path);cfg=check_config(result['config']);register(cfg)
        return cfg,mount

    def test_fresh_run_compiles_current_defaults_and_adopts_real_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,mount=self.fixture(root);run=Path(cfg['run'])
            self.assertTrue(ingest_integrity_verified(cfg))
            # This fixture's all-seven registrations use the same tiny carrier
            # to exercise coverage/orchestration, not to assert real source coverage.
            for _ in range(10):
                worker(run,'ingest-cpu-0',once=True);worker(run,'review-cpu-0',once=True)
            for reg,units in registered(run):finalize_source(run,reg,units)
            source=prepare_source(cfg,BLIP)
            pipeline=json.loads(Path(source['pipeline']).read_text())
            self.assertEqual(pipeline['policy']['profile'],'task_aware_v1')
            self.assertFalse(pipeline['policy']['i2t_image_loss'])
            self.assertEqual(pipeline['policy']['t2i_uncertain'],'exclude')
            fleet_cfg=json.loads((run/'sources'/BLIP/'fleet/frontpass.json').read_text())
            self.assertEqual(fleet_cfg['workload']['allowed_cards'],[1])
            self.assertEqual(fleet_cfg['adapter_config']['inner_config']['priority'],1)
            self.assertEqual(fleet_cfg['adapter_config']['inner_config']['global_max_cards'],512)
            pool=json.loads((run/'sources'/BLIP/'review_pool/config.json').read_text())
            self.assertEqual((pool['max_cards'],pool['replicas'],pool['cards_per_replica']),(512,256,2))
            self.assertEqual(pool['backend_kwargs']['account_card_ceiling'],512)
            ops=json.loads((run/'sources'/BLIP/'ops.json').read_text())
            self.assertIn('corpus_pipeline.orchestration.fleet',ops['commands']['frontpass_manager']['argv'])
            self.assertNotIn('sdk_compat',json.dumps(ops))
            self.assertNotIn('bootstrap',json.dumps(ops))
            result=adopt_native(source['pipeline'])
            self.assertTrue(result['complete'])
            self.assertEqual(result['coverage']['processed'],2)
            self.assertFalse((run/'COMPLETE.json').exists())
            # Completed CPU roles retire; the future review consumer stays.
            _,retired,_=cpu.policy(cfg)
            self.assertEqual(retired,{'ingest-cpu'})
            marker=run/'normalized_sources'/BLIP/'COMPLETE.json'
            marker.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Retirement'):
                cpu.policy(cfg)

    def test_snapshot_rejects_mutation_and_has_no_runtime_patch_or_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,_=self.fixture(Path(tmp));runtime=Path(cfg['runtime'])
            proof=verify_runtime(runtime)
            self.assertTrue(any(name.endswith('http_transport.py') for name in proof['files']))
            self.assertFalse(any('/attempts/' in name or 'sdk_compat' in name or 'private/' in name for name in proof['files']))
            result=subprocess.run([sys.executable,'-m','corpus_pipeline','--help'],cwd=runtime,
                env={**os.environ,'PYTHONPATH':str(runtime)},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('register-sources',result.stdout)
            # Help alone doesn't import the real worker's transitive modules.
            result=subprocess.run([sys.executable,'-c',
                'from corpus_pipeline.postprocess import api_screen; from corpus_pipeline.production import ai; from corpus_pipeline.cluster import inference'],
                cwd=runtime,env={**os.environ,'PYTHONPATH':str(runtime),'TORCH_DEVICE_BACKEND_AUTOLOAD':'0'},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            path=runtime/'corpus_pipeline/production/http_transport.py';path.write_text(path.read_text()+'\n# drift\n')
            with self.assertRaisesRegex(ValueError,'runtime changed'):verify_runtime(runtime)

    def test_cann_paths_are_preserved_and_no_bootstrap_is_generated(self):
        cfg={'runtime':'/safe/runtime','source_roots':'/safe/roots.json','python':'/safe/python'}
        script=launcher(cfg,npu=True)
        self.assertLess(script.index('set_env.sh'),script.index('export PYTHONPATH='))
        self.assertIn('${PYTHONPATH:+:$PYTHONPATH}',script)
        self.assertNotIn('portable_review',script)

    def test_cpu_retirement_never_restarts_workers_and_keeps_review_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,_=self.fixture(Path(tmp))
            resources={row['name']:NS(name=row['name'],ref=NS(key=row['notebook_id']),
                status='STOPPED',priority='6',resource={'gpu_count':0},
                to_dict=lambda:{'project':'fixture','workspace':'fixture'}) for row in cfg['cpu_workers']}
            notebooks=NS(_make_ref=Mock(side_effect=lambda _,name,*__:name),get=Mock(side_effect=lambda name,**_:resources[name]),
                         start=Mock(),stop=Mock())
            with patch.object(cpu,'completion_proofs',return_value=[]):
                result=cpu.step(cfg,NS(notebooks=notebooks))
                notebooks.start.assert_called_once_with('review-cpu',workspace='fixture')
                self.assertEqual(result['retired'],['ingest-cpu'])
                # Subsequent observation of STOPPED must still never resurrect ingestion.
                cpu.step(cfg,NS(notebooks=notebooks))
                self.assertTrue(all(call.args[0]=='review-cpu' for call in notebooks.start.call_args_list))

    def test_unknown_cpu_launch_reuses_same_durable_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,_=self.fixture(Path(tmp))
            resources={row['name']:NS(name=row['name'],ref=NS(key=row['notebook_id']),status='RUNNING',
                priority=6,resource={'gpu_count':0},to_dict=lambda:{'project':'fixture','workspace':'fixture'})
                for row in cfg['cpu_workers']}
            notebooks=NS(_make_ref=Mock(side_effect=lambda _,name,*__:name),get=Mock(side_effect=lambda name,**_:resources[name]))
            with patch('corpus_pipeline.cluster.notebook_exec.execute',return_value={'completed':False,'error_type':'TransportError'}) as execute:
                cpu.step(cfg,NS(notebooks=notebooks));cpu.step(cfg,NS(notebooks=notebooks))
                calls=execute.call_args_list
                self.assertEqual(calls[0].args,calls[2].args)
                self.assertEqual(calls[1].args,calls[3].args)
                self.assertEqual(calls[0].kwargs['notebook_id'],'cpu1')

    def test_review_resume_does_not_reenter_workflow_owned_by_live_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,_=self.fixture(Path(tmp));root=Path(cfg['run'])/'sources'/BLIP
            for name in ('reconciliation','screening','receipts','frontpass'):(root/name).mkdir(parents=True)
            atomic_json(root/'reconciliation/COMPLETE.json',{'complete':True})
            atomic_json(root/'receipts/reconcile.json',{'complete':True})
            atomic_json(root/'screening/PENDING.json',{
                'identity':{'reconciliation_complete_sha256':file_sha(root/'reconciliation/COMPLETE.json')},
                'api_selection':{'assets':1}})
            atomic_json(root/'frontpass/FRONTPASS_COMPLETE.json',{'counts':{'image:ai_suspect_pending_api':1}})
            with patch.object(workflow,'run',side_effect=AssertionError('Live consumer owns workflow lock')), \
                    patch('corpus_pipeline.orchestration.controller.subprocess.run'), \
                    patch('corpus_pipeline.orchestration.controller.subprocess.Popen',return_value=NS(pid=1,wait=lambda:0)) as launch, \
                    patch('corpus_pipeline.orchestration.controller.signal.signal'):
                self.assertEqual(review_action(cfg,BLIP,'review-start'),0)
            self.assertIn('guard',launch.call_args.args[0])

    def test_record_lock_excludes_threads_and_other_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock';result=[]
            with locked(path):
                def contender():
                    with available_lock(path) as allowed:result.append(allowed)
                thread=threading.Thread(target=contender);thread.start();thread.join(2)
                self.assertEqual(result,[False])
                code='import fcntl,sys\nf=open(sys.argv[1],"a")\ntry: fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nsys.exit(1)'
                self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)]).returncode,0)
            with available_lock(path) as allowed:self.assertTrue(allowed)

    def test_parallel_artifact_verification_still_rejects_tampering_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            items=[]
            for index in range(16):
                path=Path(tmp)/str(index);path.write_bytes(bytes([index])*4096)
                items.append({'path':str(path),'sha256':file_sha(path),'size_bytes':4096})
            workflow.verify_artifacts(items)
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                workflow.verify_artifacts([items[0],{**items[0],'sha256':'f'*64}])
            Path(items[0]['path']).write_bytes(b'wrong'+bytes(4091))
            with self.assertRaisesRegex(ValueError,'Changed stage artifact'):workflow.verify_artifacts(items)


if __name__=='__main__':unittest.main()

import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from .tuned_configuration import configuration,select_source_profile,CODE_FILES,EVIDENCE_CODE_FILES,EXPECTED_CASES
from .node_worker import worker_module
from corpus_pipeline.production.common import file_sha

class QualificationTests(unittest.TestCase):
    def fixture(self, profiles=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        (root/'plan.json').write_text('{"frozen":"test plan"}')
        evidence=root/'verification.json'
        cfg=dict(batch_size=64,decode_workers=8,cpu_threads=4,prefetch_batches=2,fast_prepare=True)
        agg=dict(target_images=256,max_records=1024,max_record_bytes=64*1024**2)
        hashes={n:file_sha(Path(__file__).with_name(n)) for n in EVIDENCE_CODE_FILES}
        cases=[]
        for (case,repetition),(source,records) in EXPECTED_CASES.items():
            profile=(profiles or {}).get(source,dict(configuration=cfg,aggregation=agg))
            cases.append(dict(case=case,repetition=repetition,source=source,comparison=dict(
                record_outcomes_equal=True,image_metadata_and_outcomes_equal=True,
                threshold_decision_disagreements=0,records=records,image_occurrences=records,
                max_probability_error={'cf384_prob':0.,'unifd_prob':0.},minimum_sscd_cosine=1.),
                candidate_configuration=copy.deepcopy(profile['configuration']),
                candidate_aggregation=copy.deepcopy(profile['aggregation']),
                timings={mode:dict(counts=dict(records=records,image_occurrences=records),elapsed_seconds=1.)
                         for mode in ('baseline','candidate')}))
        e=dict(success=True,dtype='float32',device='npu:0',
               scope='bounded_real_frontpass_numerical_and_throughput_comparison_not_AI_accuracy',
               candidate_configuration=copy.deepcopy(cfg),candidate_aggregation=copy.deepcopy(agg),
               baseline_configuration=dict(batch_size=32,threads=4),
               baseline_aggregation=dict(target_images=128,max_records=1024,max_record_bytes=64<<20),
               code_sha256=hashes,plan_sha256=file_sha(root/'plan.json'),results=cases,
               model_identity=dict(dtype='float32',code_sha256=hashes['inference.py']),
               candidate_execution_identity={**copy.deepcopy(cfg),'model_identity_unchanged':True,
                   'base_inference_code_sha256':hashes['inference.py'],
                   'optimized_inference_code_sha256':hashes['optimized_inference.py'],
                   'tuned_inference_code_sha256':hashes['tuned_inference.py'],
                   'fast_prepare_code_sha256':hashes['fast_prepare.py']})
        v={'success':True,'dtype':'float32','configuration':cfg,'aggregation':agg,'processes_per_device':1,'code_sha256':hashes,'validation_receipt':str(evidence),'validation_receipt_sha256':None}
        if profiles is not None:
            e['source_profiles']=copy.deepcopy(profiles)
            v['source_profiles']=copy.deepcopy(profiles)
        evidence.write_text(json.dumps(e))
        v['validation_receipt_sha256']=file_sha(evidence)
        return root,v
    def write(self,root,v): (root/'TUNED_QUALIFIED.json').write_text(json.dumps(v))
    def test_absent_keeps_legacy_entrypoint(self):
        root,_=self.fixture();self.assertIsNone(configuration(root));self.assertEqual(worker_module(root),'corpus_pipeline.cluster.frontpass')
    def test_verified_selects_tuned(self):
        root,v=self.fixture();self.write(root,v);self.assertEqual(configuration(root),v);self.assertEqual(worker_module(root),'corpus_pipeline.cluster.tuned_frontpass')
    def test_code_drift_fails_closed(self):
        for name in CODE_FILES:
            with self.subTest(name=name):
                root,v=self.fixture();v['code_sha256'][name]='wrong';self.write(root,v)
                with self.assertRaisesRegex(RuntimeError,'code changed'):configuration(root)
    def test_changed_validation_fails_closed(self):
        root,v=self.fixture();self.write(root,v);Path(v['validation_receipt']).write_text('{"success":false}')
        with self.assertRaisesRegex(RuntimeError,'receipt changed'):configuration(root)
    def test_settings_and_bound_rejected(self):
        for key,val in [('batch_size',0),('decode_workers',33),('cpu_threads',True),('fast_prepare',1),('prefetch_batches',5)]:
            root,v=self.fixture();v['configuration'][key]=val;self.write(root,v)
            with self.assertRaises(RuntimeError):configuration(root)
        root,v=self.fixture();v['aggregation']['target_images']=128;self.write(root,v)
        with self.assertRaisesRegex(RuntimeError,'aggregation bound'):configuration(root)
    def test_multiple_processes_not_implicitly_enabled(self):
        for value in (2,True,1.0):
            root,v=self.fixture();v['processes_per_device']=value;self.write(root,v)
            with self.assertRaisesRegex(RuntimeError,'one qualified'):configuration(root)

    def mutate_evidence(self,root,receipt,mutate):
        path=Path(receipt['validation_receipt']);value=json.loads(path.read_text());mutate(value)
        path.write_text(json.dumps(value));receipt['validation_receipt_sha256']=file_sha(path)
        self.write(root,receipt)

    def test_rehashed_unrelated_success_is_not_qualification(self):
        root,v=self.fixture()
        self.mutate_evidence(root,v,lambda e:(e.clear(),e.update(success=True)))
        with self.assertRaisesRegex(RuntimeError,'required NPU path'):configuration(root)

    def test_configuration_and_aggregation_match_actual_evidence(self):
        for section,key,value in [('candidate_configuration','decode_workers',16),
                                   ('candidate_aggregation','target_images',512)]:
            root,v=self.fixture()
            self.mutate_evidence(root,v,lambda e:e[section].update({key:value}))
            with self.assertRaisesRegex(RuntimeError,'differs from validation'):configuration(root)

    def test_all_evidence_code_hashes_are_verified(self):
        for name in EVIDENCE_CODE_FILES:
            with self.subTest(name=name):
                root,v=self.fixture()
                self.mutate_evidence(root,v,lambda e:e['code_sha256'].update({name:'wrong'}))
                with self.assertRaisesRegex(RuntimeError,'validation code changed'):configuration(root)

    def test_wrong_plan_and_candidate_execution_rejected(self):
        mutations=[lambda e:e.update(plan_sha256='wrong'),
                   lambda e:e['candidate_execution_identity'].update(batch_size=128),
                   lambda e:e['candidate_execution_identity'].update(fast_prepare_code_sha256='wrong'),
                   lambda e:e['model_identity'].update(dtype='bfloat16')]
        for mutate in mutations:
            root,v=self.fixture();self.mutate_evidence(root,v,mutate)
            with self.assertRaises(RuntimeError):configuration(root)

    def test_missing_duplicate_and_shortened_cases_rejected(self):
        mutations=[lambda e:e['results'].pop(),
                   lambda e:e['results'].__setitem__(1,copy.deepcopy(e['results'][0])),
                   lambda e:e['results'][0]['comparison'].update(records=64),
                   lambda e:e['results'][0]['timings']['candidate']['counts'].update(records=64)]
        for mutate in mutations:
            root,v=self.fixture();self.mutate_evidence(root,v,mutate)
            with self.assertRaises(RuntimeError):configuration(root)

    def test_numerical_and_decision_failures_rejected_even_with_success_true(self):
        mutations=[lambda e:e['results'][0]['comparison'].update(threshold_decision_disagreements=1),
                   lambda e:e['results'][0]['comparison'].update(threshold_decision_disagreements=False),
                   lambda e:e['results'][0]['comparison'].update(minimum_sscd_cosine=.98),
                   lambda e:e['results'][0]['comparison'].update(minimum_sscd_cosine=float('nan')),
                   lambda e:e['results'][0]['comparison']['max_probability_error'].update(cf384_prob=.002),
                   lambda e:e['results'][0]['comparison']['max_probability_error'].update(unifd_prob=float('inf')),
                   lambda e:e['results'][0]['comparison'].update(record_outcomes_equal=False)]
        for mutate in mutations:
            root,v=self.fixture();self.mutate_evidence(root,v,mutate)
            with self.assertRaises(RuntimeError):configuration(root)

    def test_present_but_malformed_or_dangling_receipt_never_falls_back(self):
        root,_=self.fixture();path=root/'TUNED_QUALIFIED.json';path.write_text('{')
        with self.assertRaises(RuntimeError):worker_module(root)
        path.unlink();path.symlink_to(root/'missing.json')
        with self.assertRaises(RuntimeError):worker_module(root)

    def profile(self, *, fast=True):
        return dict(configuration=dict(batch_size=32,decode_workers=8,cpu_threads=1,
                                       prefetch_batches=2,fast_prepare=fast),
                    aggregation=dict(target_images=256,max_records=1024,max_record_bytes=64<<20))

    def test_optional_empty_profiles_and_exact_source_override(self):
        for profiles in ({}, {'datacomp_1b':self.profile()}):
            root,v=self.fixture(profiles);self.write(root,v)
            self.assertEqual(configuration(root),v)
            self.assertEqual(select_source_profile(v,'unlisted'),dict(configuration=v['configuration'],aggregation=v['aggregation']))
            self.assertEqual(select_source_profile(v,'datacomp_1b'),profiles.get('datacomp_1b',dict(configuration=v['configuration'],aggregation=v['aggregation'])))
            self.assertEqual(select_source_profile(v,'datacomp_1b/suffix')['configuration'],v['configuration'])
            selected=select_source_profile(v,'datacomp_1b');selected['configuration']['batch_size']=999
            self.assertNotEqual(select_source_profile(v,'datacomp_1b')['configuration']['batch_size'],999)

    def test_source_profile_fast_false_is_valid_not_false_mapping(self):
        root,v=self.fixture({'datacomp_1b':self.profile(fast=False)});self.write(root,v)
        self.assertEqual(configuration(root),v)
        for invalid in (False,None,[],1):
            root,v=self.fixture();v['source_profiles']=invalid;self.write(root,v)
            with self.assertRaisesRegex(RuntimeError,'source profiles'):configuration(root)

    def test_unverified_source_extra_keys_and_out_of_bounds_rejected(self):
        mutations=[lambda p:p.update(unmeasured=self.profile()),
                   lambda p:p['datacomp_1b'].update(extra=True),
                   lambda p:p['datacomp_1b']['configuration'].update(decode_workers=33),
                   lambda p:p['datacomp_1b']['configuration'].update(cpu_threads=True),
                   lambda p:p['datacomp_1b']['configuration'].update(fast_prepare=0),
                   lambda p:p['datacomp_1b']['aggregation'].update(target_images=32),
                   lambda p:p['datacomp_1b']['aggregation'].update(max_records=4097),
                   lambda p:p['datacomp_1b']['aggregation'].update(max_record_bytes=(64<<20)+1)]
        for mutate in mutations:
            root,v=self.fixture({'datacomp_1b':self.profile()});mutate(v['source_profiles']);self.write(root,v)
            with self.assertRaises(RuntimeError):configuration(root)

    def test_profiles_must_match_evidence_exactly(self):
        for mutation in (lambda e:e.pop('source_profiles'),
                         lambda e:e['source_profiles']['datacomp_1b']['configuration'].update(decode_workers=9),
                         lambda e:e.update(source_profiles=False)):
            root,v=self.fixture({'datacomp_1b':self.profile()});self.mutate_evidence(root,v,mutation)
            with self.assertRaises(RuntimeError):configuration(root)

    def test_every_result_binds_its_selected_configuration_and_aggregation(self):
        for key,value in [('candidate_configuration',None),('candidate_aggregation',None),
                          ('candidate_configuration',dict(self.profile()['configuration'],batch_size=64)),
                          ('candidate_aggregation',dict(self.profile()['aggregation'],target_images=512))]:
            root,v=self.fixture({'datacomp_1b':self.profile()})
            def mutate(e):
                result=next(r for r in e['results'] if r['source']=='datacomp_1b')
                result[key]=value
            self.mutate_evidence(root,v,mutate)
            with self.assertRaises(RuntimeError):configuration(root)
        root,v=self.fixture({'datacomp_1b':self.profile()})
        self.mutate_evidence(root,v,lambda e:e['results'][0].update(candidate_configuration=self.profile()['configuration']))
        with self.assertRaisesRegex(RuntimeError,'source profile'):configuration(root)

    def test_primary_source_cases_are_mandatory_even_without_overrides(self):
        primary=[key for key in EXPECTED_CASES if key[0].startswith('primary-')]
        self.assertEqual(len(primary),3)
        for key in primary:
            root,v=self.fixture()
            self.mutate_evidence(root,v,lambda e:e.update(results=[r for r in e['results'] if (r['case'],r['repetition'])!=key]))
            with self.assertRaisesRegex(RuntimeError,'coverage incomplete'):configuration(root)

    def test_source_profile_requires_real_images_and_correct_source_name(self):
        for mutate in (lambda r:r['comparison'].update(image_occurrences=0),
                       lambda r:r.update(source='unmeasured')):
            root,v=self.fixture({'datacomp_1b':self.profile()})
            self.mutate_evidence(root,v,lambda e:mutate(next(r for r in e['results'] if r['source']=='datacomp_1b')))
            with self.assertRaises(RuntimeError):configuration(root)

    def test_runtime_switches_only_changed_configuration_and_records_task_identity(self):
        from .tuned_frontpass import apply_source_profile
        _,v=self.fixture({'datacomp_1b':self.profile(), 'anyword3m':self.profile()})
        worker=Mock();worker.execution_identity=copy.deepcopy(v['configuration'])
        def configure(**cfg):worker.execution_identity.update(cfg)
        worker.configure.side_effect=configure
        agg=apply_source_profile(worker,v,'as100m','receipt-sha')
        worker.configure.assert_not_called();self.assertEqual(agg,v['aggregation'])
        for source in ('datacomp_1b','datacomp_1b','anyword3m'):
            apply_source_profile(worker,v,source,'receipt-sha')
            self.assertEqual(worker.execution_identity['source'],source)
            self.assertEqual(worker.execution_identity['configuration'],self.profile()['configuration'])
            self.assertEqual(worker.execution_identity['aggregation'],self.profile()['aggregation'])
            self.assertEqual(worker.execution_identity['qualification_sha256'],'receipt-sha')
        self.assertEqual(worker.configure.call_count,1)
        apply_source_profile(worker,v,'unlisted','receipt-sha')
        self.assertEqual(worker.configure.call_count,2)
        self.assertEqual(worker.execution_identity['configuration'],v['configuration'])
        self.assertEqual(worker.execution_identity['source_profile'],'default')
        with self.assertRaisesRegex(RuntimeError,'nonempty'):apply_source_profile(worker,v,None,'receipt-sha')

    def test_aggregation_only_switch_does_not_recreate_decode_pool(self):
        from .tuned_frontpass import apply_source_profile
        _,v=self.fixture();v['source_profiles']={'datacomp_1b':dict(configuration=copy.deepcopy(v['configuration']),aggregation=dict(v['aggregation'],target_images=512))}
        worker=Mock();worker.execution_identity=copy.deepcopy(v['configuration'])
        self.assertEqual(apply_source_profile(worker,v,'datacomp_1b','sha')['target_images'],512)
        worker.configure.assert_not_called()

    def test_real_configure_switches_process_cpu_threads_only_between_idle_tasks(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from .tuned_frontpass import apply_source_profile
        from .tuned_inference import TunedResidentInferenceWorker
        _,v=self.fixture({'datacomp_1b':self.profile(fast=False)})
        worker=TunedResidentInferenceWorker.__new__(TunedResidentInferenceWorker)
        worker.closed=worker._active_stream=False
        worker.execution_identity=copy.deepcopy(v['configuration'])
        worker.pool=ThreadPoolExecutor(max_workers=1)
        changes=[]
        worker.torch=SimpleNamespace(set_num_threads=changes.append)
        models=worker.bundles={'resident':'same loaded model objects'}
        try:
            apply_source_profile(worker,v,'datacomp_1b','sha')
            selected_pool=worker.pool
            self.assertEqual(changes,[1]);self.assertEqual(selected_pool._max_workers,8)
            apply_source_profile(worker,v,'datacomp_1b','sha')
            self.assertIs(worker.pool,selected_pool);self.assertEqual(changes,[1])
            worker._active_stream=True
            before=copy.deepcopy(worker.execution_identity)
            with self.assertRaisesRegex(RuntimeError,'idle'):
                apply_source_profile(worker,v,'as100m','sha')
            self.assertEqual(worker.execution_identity,before);self.assertEqual(changes,[1])
            worker._active_stream=False
            apply_source_profile(worker,v,'as100m','sha')
            self.assertEqual(changes,[1,4]);self.assertIs(worker.bundles,models)
        finally:
            worker.pool.shutdown(wait=True,cancel_futures=True)

    def test_lease_loop_identical_except_explicit_selection_inside_heartbeat_try(self):
        def loop(filename):
            tree=ast.parse(Path(__file__).with_name(filename).read_text())
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_worker')
            return next(n for n in ast.walk(function) if isinstance(n,ast.While))
        candidate=loop('tuned_frontpass.py')
        guarded=next(n for n in candidate.body if isinstance(n,ast.Try))
        selection,identity=guarded.body[:2]
        self.assertEqual(ast.unparse(selection),"aggregation = apply_source_profile(worker, qualified, claim['task']['source'], qualification_sha256)")
        self.assertEqual(ast.unparse(identity),"atomic_json(worker_root / 'execution_identity.json', worker.execution_identity)")
        guarded.body=guarded.body[2:]
        self.assertEqual(ast.dump(loop('frontpass.py'),include_attributes=False),ast.dump(candidate,include_attributes=False))

if __name__=='__main__':unittest.main()

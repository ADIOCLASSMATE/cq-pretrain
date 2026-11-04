import unittest
from .policy import (TASK_POLICY, config, evaluate_image, evaluate_record, policy_hash,
                     quality_decision)


def observation(collection='dcvlm', **values):
    return {'asset_id': 'a'*64, 'collection_id': collection, 'width': 1024,
            'height': 1024, 'origin': 'mixed', **values}


class PolicyTests(unittest.TestCase):
    def test_uncertain_completed_is_retained_but_infra_failure_is_not_done(self):
        uncertain = evaluate_image([observation(detector={'status': 'suspect'},
            review={'status': 'uncertain', 'evidence_ids': ['review-1']})])
        self.assertEqual(uncertain['action'], 'retain')
        self.assertTrue(uncertain['review_pending'] and uncertain['processing_complete'])
        missing = evaluate_image([observation(detector={'status': 'suspect'}, review={'status': 'error'})])
        self.assertEqual(missing['action'], 'pending')
        self.assertFalse(missing['processing_complete'])

    def test_cross_collection_exact_asset_confirmation_is_shared_and_conflict_is_visible(self):
        generated = observation('genund', origin='ai_generated', origin_confirmed=True, origin_evidence=['dataset-page'])
        self.assertEqual(evaluate_image([observation(), generated])['action'], 'exclude')
        natural = observation(review={'status': 'confirmed_natural', 'evidence_ids': ['human-1']})
        result = evaluate_image([natural, generated])
        self.assertEqual(result['action'], 'pending')
        self.assertIn('conflicting_confirmed_origin_evidence', result['conflicts'])
        with self.assertRaises(ValueError):
            evaluate_image([observation(), observation(asset_id='b'*64)])

    def test_real_parent_edit_and_programmatic_protection_require_evidence(self):
        render = evaluate_image([observation(origin='rendered', origin_confirmed=True, origin_evidence=['clevr-paper'])])
        self.assertEqual(render['action'], 'retain')
        edited = observation(origin='ai_edited', real_parent_verified=True, real_parent_asset_id='b'*64,
                             parent_evidence=['edit-lineage'])
        self.assertEqual(evaluate_image([edited])['action'], 'retain')
        edited.pop('parent_evidence')
        self.assertFalse(evaluate_image([edited])['processing_complete'])
        self.assertTrue(evaluate_image([observation(origin='rendered')])['needs_detector'])

    def test_size_boundary_and_only_potential_flags(self):
        self.assertEqual(evaluate_image([observation(width=127)])['action'], 'exclude')
        result = evaluate_image([observation(width=128, detector={'status': 'clear', 'evidence_ids': ['detector']})])
        self.assertEqual(result['action'], 'retain')
        result = evaluate_image([observation(layout_source_tags=['layout-rich'])])
        self.assertTrue(result['recaption_resolution_potential'])
        self.assertEqual(result['layout_api_action'], 'none')

    def test_bad_caption_is_record_scoped_and_ai_text_is_not_filtered(self):
        quality = {'scope': 'source_record', 'record_id': 'one', 'source_record_sha256':'1'*64, 'reason': 'bad_caption',
                   'confirmed': True, 'evidence_ids': ['verified-mismatch']}
        decisions = {'a'*64: evaluate_image([observation(detector={'status':'clear','evidence_ids':['d']})])}
        one = {'record_id': 'one', 'source_record_sha256':'1'*64, 'category': 'single_t2i', 'text_origin': 'ai_generated', 'image_ids': ['a'*64]}
        two = {**one, 'record_id': 'two', 'source_record_sha256':'2'*64, 'category': 'multi_i2t'}
        self.assertEqual(evaluate_record(one, decisions, [quality])['action'], 'exclude')
        kept = evaluate_record(two, decisions, [quality])
        self.assertEqual(kept['action'], 'retain')
        self.assertTrue(kept['ai_text_annotated'])
        self.assertEqual(kept['task_type'], 'multi_i2t')
        self.assertEqual(kept['ordered_image_ids'], ['a'*64])
        with self.assertRaises(ValueError):
            quality_decision({**quality, 'scope':'asset','asset_id':'a'*64})


class TaskPolicyTests(unittest.TestCase):
    cfg={'task_routing':TASK_POLICY}

    def image(self,**values):
        return observation('genund',processing_policy=TASK_POLICY,task_route='i2t',
            image_loss_allowed=False,status='i2t_dedup_only',pixel_sha256='b'*64,**values)

    def record(self,**values):
        return {'record_id':'i2t','source_record_sha256':'1'*64,'image_ids':['a'*64],
                'processing_policy':TASK_POLICY,'task_route':'i2t','image_loss_allowed':False,**values}

    def test_legacy_policy_hash_stays_identical_and_new_mode_is_bound(self):
        self.assertEqual(policy_hash(),'e8ac12b2b48f00e2dd05de9245ebae9b51d25b63211dac0aa4bce87d01b9633b')
        self.assertNotIn('task_routing',config())
        self.assertNotEqual(policy_hash(self.cfg),policy_hash())
        for value in (None,False,'typo'):
            with self.assertRaisesRegex(ValueError,'task routing'):
                config({'task_routing':value})

    def test_i2t_ai_and_small_image_are_retained_but_t2i_and_legacy_still_exclude(self):
        observed=self.image(width=8,origin='ai_generated',origin_confirmed=True,origin_evidence=['source'])
        image=evaluate_image([observed],self.cfg)
        self.assertEqual(image['action'],'retain')
        self.assertFalse(image['needs_detector'] or image['review_pending'] or image['image_loss_allowed'])
        decisions={'a'*64:image}
        self.assertEqual(evaluate_record(self.record(),decisions,policy_config=self.cfg)['action'],'retain')
        for record in (self.record(task_route='t2i',image_loss_allowed=True),
                       {'record_id':'legacy','image_ids':['a'*64]}):
            self.assertEqual(evaluate_record(record,decisions,policy_config=self.cfg)['action'],'exclude')
        self.assertEqual(evaluate_image([observed])['action'],'exclude')

    def test_mixed_usage_keeps_distinct_asset_decisions(self):
        image=evaluate_image([self.image(origin='ai_generated',origin_confirmed=True,origin_evidence=['source']),
                              observation()],self.cfg)
        self.assertEqual(image['action'],'exclude')
        self.assertEqual(image['task_decisions']['i2t']['action'],'retain')
        self.assertEqual(evaluate_record(self.record(),{'a'*64:image},policy_config=self.cfg)['action'],'retain')

    def test_i2t_skips_blur_and_caption_quality_but_corruption_and_binding_stay_blocked(self):
        blur={'scope':'asset','asset_id':'a'*64,'reason':'blur','confirmed':True,'evidence_ids':['blur']}
        caption={'scope':'source_record','source_record_sha256':'1'*64,
                 'reason':'bad_caption','confirmed':True,'evidence_ids':['caption']}
        image=evaluate_image([self.image(quality_decisions=[blur])],self.cfg)
        self.assertEqual(evaluate_record(self.record(),{'a'*64:image},[caption],self.cfg)['action'],'retain')
        for row in (self.record(binding_valid=False),self.record(issues=['image_reference_unresolved'])):
            result=evaluate_record(row,{'a'*64:image},policy_config=self.cfg)
            self.assertEqual(result['action'],'pending');self.assertFalse(result['processing_complete'])
        broken=evaluate_image([self.image(quality_decisions=[{**blur,'reason':'corrupt_image'}])],self.cfg)
        self.assertEqual(evaluate_record(self.record(),{'a'*64:broken},policy_config=self.cfg)['action'],'exclude')

    def test_i2t_missing_decode_and_conflicting_dimensions_are_not_complete(self):
        base=self.image();base.pop('status');base.pop('pixel_sha256')
        self.assertFalse(evaluate_image([base],self.cfg)['processing_complete'])
        for values in ({'error_stage':'decode'},{'integrity_status':'failed'},
                       {'has_invalid_observation':True},{'pixel_sha256_conflict':True}):
            self.assertFalse(evaluate_image([self.image(**values)],self.cfg)['processing_complete'])
        image=evaluate_image([self.image(),self.image(width=512)],self.cfg)
        self.assertFalse(image['processing_complete'])
        other=self.image();other['pixel_sha256']='c'*64
        self.assertFalse(evaluate_image([self.image(),other],self.cfg)['processing_complete'])

    def test_unbound_or_inconsistent_i2t_cannot_bypass_global_decision(self):
        observed=observation(origin='ai_generated',origin_confirmed=True,origin_evidence=['source'])
        image=evaluate_image([observed],self.cfg)
        missing=evaluate_record(self.record(),{'a'*64:image},policy_config=self.cfg)
        self.assertEqual(missing['action'],'pending')
        row=self.record();row.pop('processing_policy')
        self.assertEqual(evaluate_record(row,{'a'*64:image},policy_config=self.cfg)['action'],'exclude')
        with self.assertRaisesRegex(ValueError,'image loss policy'):
            evaluate_record(self.record(image_loss_allowed=True),{'a'*64:image},policy_config=self.cfg)


if __name__ == '__main__': unittest.main()

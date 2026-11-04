"""Task direction, rather than image identity alone, selects the AI gate."""
import unittest

from corpus_pipeline.policy import evaluate, normalize
from corpus_pipeline.production.task_policy import VERSION


class TaskPolicyTests(unittest.TestCase):
    def record(self, direction='i2t', origin='ai_generated', exists=True):
        return normalize({'collection':'genund','source':'fixture',
            'locator':{'path':'/fixture/annotations.json'},
            'raw':{'caption':'A detailed caption.','direction':direction,'image_origin':origin},
            'image_refs':[{'kind':'file','path':'/fixture/image.png','exists':exists}]})

    def test_i2t_keeps_ai_sources_and_unknown_with_text_supervision(self):
        for origin in ('ai_generated','ai_edited','unknown','mixed','natural'):
            with self.subTest(origin=origin):
                result=evaluate(self.record(origin=origin),{'task_routing':VERSION})
                self.assertEqual(result['decision'],'accepted')
                self.assertEqual(result['task_route'],'i2t')
                self.assertIs(result['image_loss_allowed'],False)
                self.assertEqual(result['eligible_views'],['single_i2t'])
                self.assertEqual(result['provenance']['images'][0]['origin'],origin)
                self.assertEqual(result['segments'][-1]['text'],'A detailed caption.')

    def test_t2i_keeps_previous_gate_and_no_inverse_view(self):
        result=evaluate(self.record('t2i'),{'task_routing':VERSION})
        self.assertEqual(result['decision'],'rejected')
        self.assertEqual(result['eligible_views'],['single_t2i'])
        self.assertEqual(result['task_route'],'t2i')
        result=evaluate(self.record('t2i','unknown'),{'task_routing':VERSION})
        self.assertIn('image_provenance_unknown',result['reason_codes'])

    def test_structural_failures_and_evaluation_data_still_gated(self):
        result=evaluate(self.record(exists=False),{'task_routing':VERSION})
        self.assertEqual(result['decision'],'quarantine')
        self.assertIn('image_reference_unresolved',result['reason_codes'])
        row=self.record();row['issues']=['unresolved_original_image_binding']
        self.assertEqual(evaluate(row,{'task_routing':VERSION})['decision'],'quarantine')
        row=self.record();row['metadata']['usage']='evaluation'
        self.assertEqual(evaluate(row,{'task_routing':VERSION})['decision'],'rejected')

    def test_legacy_unchanged_and_bad_version_rejected(self):
        result=evaluate(self.record(),{})
        self.assertEqual(result['decision'],'rejected')
        self.assertNotIn('processing_policy',result)
        with self.assertRaisesRegex(ValueError,'Unsupported task routing'):
            evaluate(self.record(),{'task_routing':'guess'})

    def test_interleaved_user_images_are_i2t_without_reordering(self):
        row=self.record();row['format']='interleaved'
        row['segments'][0]['role']='user'
        result=evaluate(row,{'task_routing':VERSION})
        self.assertEqual(result['category'],'interleaved')
        self.assertEqual(result['task_route'],'i2t')
        self.assertIs(result['image_loss_allowed'],False)
        self.assertEqual(result['segments'],row['segments'])

    def test_interleaved_explicit_t2i_keeps_generation_gate(self):
        row=self.record(direction='t2i');row['format']='interleaved'
        result=evaluate(row,{'task_routing':VERSION})
        self.assertEqual(result['task_route'],'t2i')
        self.assertEqual(result['decision'],'rejected')


if __name__=='__main__':unittest.main()

"""Strict visible-blur gate and transport tests; no paid calls."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
from corpus_pipeline.codex_review.test_worker import MockCLI
from corpus_pipeline.context_review.test_worker import item
from corpus_pipeline.context_review import blur_worker as worker


def result(**changes):
    value = {'item_id':'opaque-a','task_type':'caption_pair','blur_status':'clear','blur_types':[],
             'decision':'retain_no_detected_blur','visible_content':'表面具有清晰的细颗粒与纹理边缘。',
             'evidence':[{'location':'整个材质表面','observation':'细小颗粒和纹理边缘清楚，没有可见模糊。','kind':'clear'}],
             'reason':'纹理本身有柔和色差，但细部仍明确，未见实际模糊。','prior_reviews_not_used':True}
    value.update(changes)
    return value


def blurred(kind='localized_anonymization',**changes):
    value=result(blur_status='blur_present',blur_types=[kind],decision='exclude_blur',
                 visible_content='人物身体仍可识别，但脸部存在明显局部模糊。',
                 evidence=[{'location':'人物脸部','observation':'五官区域变成低频色块，边缘显著柔化。','kind':'blur'}],
                 reason='虽然主体仍可辨识，最新规则要求有可见局部模糊就排除。')
    value.update(changes)
    return value


class BlurTests(unittest.TestCase):
    def test_clear_texture_passes_only_blur_gate(self):
        worker.validate_result(result(),'opaque-a',item())

    def test_recognizable_subject_and_background_blur_still_excluded(self):
        for kind in worker.BLUR_TYPES:
            worker.validate_result(blurred(kind),'opaque-a',item())
            with self.assertRaisesRegex(ValueError,'visible_blur'):
                worker.validate_result(blurred(kind,decision='retain_no_detected_blur'),'opaque-a',item())

    def test_clear_cannot_hide_blur_evidence_or_type(self):
        for bad in [result(blur_types=['global_defocus']),result(evidence=blurred()['evidence']),result(decision='exclude_blur')]:
            with self.assertRaises(ValueError):worker.validate_result(bad,'opaque-a',item())

    def test_uncertainty_is_not_forced_binary(self):
        value=result(blur_status='uncertain',blur_types=['unknown'],decision='needs_review',
                     evidence=[{'location':'柔软色块区域','observation':'有限分辨率下不能区分天然软纹理与模糊。','kind':'ambiguous'}])
        worker.validate_result(value,'opaque-a',item())
        value['decision']='exclude_blur'
        with self.assertRaises(ValueError):worker.validate_result(value,'opaque-a',item())

    def test_strict_fields_id_task_and_unique_cues(self):
        variants=[result(item_id='wrong'),result(task_type='interleaved'),result(prior_reviews_not_used=1),
                  result(evidence=result()['evidence']*2),blurred(blur_types=['global_defocus']*2)]
        for bad in variants:
            with self.assertRaises(ValueError):worker.validate_result(bad,'opaque-a',item())

    def test_source_truncation_is_explicit_but_pixels_may_settle_blur(self):
        source=item();source['source_context']['caption_texts']=['x'*5000]
        prompt=worker.build_prompt([source])
        self.assertNotIn('SECRET_PRIOR_VERDICT',prompt)
        self.assertIn('"prompt_context_truncated":true',prompt)
        worker.validate_result(blurred(),'opaque-a',source)
        worker.validate_result(result(),'opaque-a',source)

    def test_fixed_cli_runtime_real_attachment_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def prepare(source,directory,index,**kw):
                directory.mkdir(parents=True,exist_ok=True);path=directory/f'{index:02}.png'
                Image.new('RGB',(16,16),(40,60,90)).save(path)
                with Image.open(path) as im:im.load()
                return {'path':str(path),'sent_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':16,'height':16,'decoded':True}
            fake=MockCLI({'items':[blurred()]})
            with mock.patch('corpus_pipeline.codex_review.inputs.prepare_image',side_effect=prepare),mock.patch.object(worker.subprocess,'Popen',side_effect=fake):
                output=worker.run_batch([item()],root/'batch',{'codex_cli_version':'test'})
            self.assertEqual(output['results'],[blurred()]);self.assertEqual(output['errors'],{})
            self.assertIn('gpt-5.6-sol',fake.command);self.assertIn('model_reasoning_effort="low"',fake.command)
            self.assertEqual(fake.stdin_bytes.decode(),worker.build_prompt([item()]))
            for artifact in output['metadata']['artifacts'].values():
                self.assertEqual(worker.file_sha(artifact['path']),artifact['sha256'])


if __name__=='__main__':unittest.main()

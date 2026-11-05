"""Selected generation readouts preserve understanding and exact CQ gradients."""
from training.loss import compose_losses
import copy
import unittest

import torch

from data.batch import fake_batch
from data.planning import build_plan
from models.modeling_umm import JointModel
import tests.test_qwen35_vision as vision_tests


class SelectedCQContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vision_tests.NativeVisionContracts.setUpClass()
        cls.fixture=vision_tests.NativeVisionContracts()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        vision_tests.NativeVisionContracts.tearDownClass()

    def compare_selected(self,selected,preserve_documents):
        cfg=self.fixture.cfg()
        full=fake_batch(cfg,64,images_per_sequence=2)
        # Exercise both CPU mask builders against the same already packed
        # document; the lightweight renderer fixture does not need whole-doc
        # packing or MOSS to verify backbone readout and gradient equivalence.
        plan_cfg=copy.deepcopy(cfg)
        plan_cfg.preserve_documents=preserve_documents
        # Baseline computes every CQ but consumes exactly the selected losses.
        full.generation_image_indices=None
        full.mainline=build_plan(full,plan_cfg)
        count=len(full.image_indices)
        chosen=torch.tensor(selected,dtype=torch.long)
        full.generation_image_indices=chosen
        full.latents=full.latents[chosen]
        full.noise=full.noise.reshape(cfg.flow_draws,count,cfg.image_tokens,cfg.latent_channels)[:,chosen].reshape(
            cfg.flow_draws*len(chosen),cfg.image_tokens,cfg.latent_channels)
        full.times=full.times.reshape(cfg.flow_draws,count)[:,chosen].flatten()
        lean=copy.deepcopy(full)
        lean.mainline=build_plan(lean,plan_cfg)

        torch.testing.assert_close(lean.mainline.condition_indices,full.mainline.condition_indices)
        self.assertEqual(lean.mainline.output_size,full.mainline.output_size)
        self.assertEqual(len(full.mainline.query_positions)-len(lean.mainline.query_positions),
                         (count-len(chosen))*cfg.image_condition_queries)
        self.assertEqual(int(lean.mainline.query_image.sum()),len(chosen)*cfg.image_condition_queries)
        torch.testing.assert_close(lean.image_indices,full.image_indices)
        torch.testing.assert_close(lean.vision.content_indices,full.vision.content_indices)
        torch.testing.assert_close(lean.mainline.delta_plan.content_order,full.mainline.delta_plan.content_order)

        full.vision.pixel_values.requires_grad_()
        lean.vision.pixel_values.requires_grad_()
        torch.manual_seed(37)
        expected_model=JointModel(cfg)
        torch.nn.init.normal_(expected_model.renderer.output.weight,std=.02)
        actual_model=copy.deepcopy(expected_model)
        expected=compose_losses(expected_model(full, return_hidden=True), expected_model.cfg)
        actual=compose_losses(actual_model(lean, return_hidden=True), actual_model.cfg)
        for key in ('loss','image_loss','text_loss'):
            torch.testing.assert_close(actual[key],expected[key],atol=3e-6,rtol=2e-4,msg=key)
        torch.testing.assert_close(actual['hidden'][lean.text_indices],expected['hidden'][full.text_indices],
                                   atol=3e-6,rtol=2e-4)
        indices=full.mainline.condition_indices[chosen]
        torch.testing.assert_close(actual['hidden'][indices],expected['hidden'][indices],atol=3e-6,rtol=2e-4)
        expected['loss'].backward()
        actual['loss'].backward()
        for (name,p),(other_name,other) in zip(actual_model.named_parameters(),expected_model.named_parameters()):
            self.assertEqual(name,other_name)
            self.assertEqual(p.grad is None,other.grad is None,name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,other.grad,atol=4e-6,rtol=4e-4,msg=name)
        torch.testing.assert_close(lean.vision.pixel_values.grad,full.vision.pixel_values.grad,
                                   atol=4e-6,rtol=4e-4)
        # The first image remains a real understanding input even if its CQ is
        # omitted. Caption/text gradients must still pass through that image.
        if 0 not in selected:
            self.assertGreater(lean.vision.pixel_values.grad[:lean.vision.lengths[0]].abs().sum().item(),0)

    def test_selected_cq_text_losses_and_every_gradient_match(self):
        for selected in ([1],[],[1,0]):
            for preserve_documents in (False,True):
                with self.subTest(selected=selected,preserve_documents=preserve_documents):
                    self.compare_selected(selected,preserve_documents)

    def test_none_preserves_all_readouts_and_bi_keeps_its_write_stream(self):
        for method in ('CQ_IT','Bi_I+CQ_T'):
            with self.subTest(method=method):
                cfg=self.fixture.cfg(method=method)
                batch=fake_batch(cfg,64,images_per_sequence=2)
                batch.generation_image_indices=None
                original=build_plan(batch,cfg)
                batch.generation_image_indices=torch.arange(2)
                selected=build_plan(batch,cfg)
                for name in ('query_positions','query_rope','output_positions','condition_indices',
                             'generation_positions','generation_condition','rope_cos','rope_sin'):
                    torch.testing.assert_close(getattr(original,name),getattr(selected,name),atol=0,rtol=0)
                if method=='Bi_I+CQ_T':
                    batch.generation_image_indices=torch.empty(0,dtype=torch.long)
                    empty=build_plan(batch,cfg)
                    torch.testing.assert_close(empty.generation_positions,original.generation_positions)
                    torch.testing.assert_close(empty.generation_condition,original.generation_condition)

    def test_invalid_readout_indices_fail_during_cpu_planning(self):
        cfg=self.fixture.cfg()
        batch=fake_batch(cfg,64,images_per_sequence=2)
        for selected in (torch.tensor([2]),torch.tensor([-1]),torch.tensor([0,0]),
                         torch.tensor([0.]),torch.tensor([[0]])):
            with self.subTest(selected=selected.tolist()):
                batch.generation_image_indices=selected
                with self.assertRaises(ValueError):build_plan(batch,cfg)


if __name__=='__main__':unittest.main()

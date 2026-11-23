"""Selected generation readouts preserve understanding and exact CQ gradients."""

from training.loss import compose_losses
import copy
import unittest

import torch

from data.batch import fake_batch
from data.batch_layout import build_plan
from models.modeling_umm import JointModel
import tests.test_qwen35_vision as vision_tests


class SelectedCQContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vision_tests.NativeVisionContracts.setUpClass()
        cls.fixture = vision_tests.NativeVisionContracts()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        vision_tests.NativeVisionContracts.tearDownClass()

    def test_selected_queries_match_all_readouts_and_gradients(self):
        cfg = self.fixture.cfg()
        for selected in ([1], [], [1, 0]):
            torch.manual_seed(37)
            full = fake_batch(cfg, 64, images_per_sequence=2)
            lean = copy.deepcopy(full)
            chosen = torch.tensor(selected, dtype=torch.long)
            lean.generation_image_indices = chosen
            lean.mainline = build_plan(lean, cfg)
            full.vision.pixel_values.requires_grad_()
            lean.vision.pixel_values.requires_grad_()
            reference = JointModel(cfg)
            candidate = copy.deepcopy(reference)

            def objective(model, batch):
                hidden = model.backbone(batch)
                text = hidden[batch.text_indices]
                visual = batch.mainline.gather_image_conditions(hidden, chosen)
                return torch.cat((text, visual.reshape(-1, cfg.width)))

            expected = objective(reference, full)
            actual = objective(candidate, lean)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-4)
            grad = torch.randn_like(expected) / expected.numel()
            expected.backward(grad)
            actual.backward(grad)
            for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=5e-8, rtol=4e-4, msg=lambda m: name + " " + m)
            torch.testing.assert_close(
                full.vision.pixel_values.grad, lean.vision.pixel_values.grad, atol=5e-8, rtol=4e-4
            )
            self.assertGreater(lean.vision.pixel_values.grad[: lean.vision.lengths[0]].abs().sum().item(), 0)

    def test_invalid_readout_indices_fail_during_cpu_planning(self):
        cfg = self.fixture.cfg()
        batch = fake_batch(cfg, 64, images_per_sequence=2)
        for selected in (
            torch.tensor([2]),
            torch.tensor([-1]),
            torch.tensor([0, 0]),
            torch.tensor([0.0]),
            torch.tensor([[0]]),
        ):
            with self.subTest(selected=selected.tolist()):
                batch.generation_image_indices = selected
                with self.assertRaises(ValueError):
                    build_plan(batch, cfg)


if __name__ == "__main__":
    unittest.main()

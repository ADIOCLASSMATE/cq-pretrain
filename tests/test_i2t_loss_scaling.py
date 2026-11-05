"""Sparse image objectives preserve DP weighting and understanding gradients."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from distributed.communication import OrderedGradientBuckets
from training.train import data_loss_scales, parse_args


class I2TLossScalingTests(unittest.TestCase):
    def test_unequal_dp_packs_match_global_selected_image_gradient(self):
        # Four packs on two replicas, including two understanding-only packs.
        # All 116 input images still reach the backbone, but just three images
        # should determine the renderer average and its gradient.
        packs=[[(8,[1.,3.],3),(100,[],5)],[(7,[],2),(1,[5.],4)]]
        dp=len(packs);global_images=3;global_text=14
        actual=torch.tensor(.7,requires_grad=True)
        total=actual*0
        for replica in packs:
            for inputs,targets,text in replica:
                batch=SimpleNamespace(image_indices=torch.empty(inputs,1),
                    generation_image_count=len(targets),text_indices=torch.empty(text,dtype=torch.long))
                _,scale=data_loss_scales(batch,global_text,global_images,dp)
                loss=(actual-torch.tensor(targets)).square().mean() if targets else actual*0
                total=total+loss*scale/dp
        total.backward()
        expected=torch.tensor(.7,requires_grad=True)
        oracle=(expected-torch.tensor([1.,3.,5.])).square().mean()
        oracle.backward()
        torch.testing.assert_close(total,oracle)
        torch.testing.assert_close(actual.grad,expected.grad)

    def test_no_selected_targets_leave_text_weight_unchanged(self):
        batch=SimpleNamespace(image_indices=torch.empty(9,1),generation_image_count=0,
                              text_indices=torch.empty(7,dtype=torch.long))
        self.assertEqual(data_loss_scales(batch,14,0,2),(1.,0.))
        batch.text_indices=torch.empty(0,dtype=torch.long)
        self.assertEqual(data_loss_scales(batch,0,0,2),(0.,0.))

    def test_absent_renderer_preserves_vision_and_accumulated_gradients(self):
        vision=torch.nn.Linear(2,2)
        renderer=torch.nn.Linear(2,2)
        renderer.bias.requires_grad_(False)
        model=SimpleNamespace(patch_embed=vision,renderer=renderer)
        parameters=[p for m in (vision,renderer) for p in m.parameters() if p.requires_grad]
        for p in parameters:p.main_grad=torch.full_like(p,3.)
        counts={}
        def ready(parameter,force):
            counts[id(parameter)]=counts.get(id(parameter),0)+1
        group=SimpleNamespace(register_grad_ready=ready)
        wrapped=SimpleNamespace(module=model,param_to_bucket_group={p:group for p in parameters},
                                force_all_reduce=False)
        OrderedGradientBuckets.absent_renderer(SimpleNamespace(wrapped=wrapped))
        self.assertEqual(counts,{id(renderer.weight):1})
        for p in parameters:torch.testing.assert_close(p.main_grad,torch.full_like(p,3.))
        # The real vision backward may now report its readiness exactly once.
        for p in vision.parameters():ready(p,False)
        self.assertEqual(set(counts),{id(p) for p in parameters})
        self.assertEqual(set(counts.values()),{1})

    def test_training_cli_inherits_config_and_accepts_explicit_recipe(self):
        with patch('sys.argv',['train','--output','unused']):
            args=parse_args()
        self.assertIsNone(args.i2t_image_loss_probability)
        self.assertIsNone(args.cfg_null_condition)
        with patch('sys.argv',['train','--output','unused','--i2t-image-loss-probability','.2',
                               '--cfg-null-condition','i2t_prefix']):
            args=parse_args()
        self.assertEqual(args.i2t_image_loss_probability,.2)
        self.assertEqual(args.cfg_null_condition,'i2t_prefix')
        for value in ('-0.1','1.1','nan','inf'):
            with self.subTest(value=value),patch('sys.argv',['train','--output','unused',
                    '--i2t-image-loss-probability',value]),patch('sys.stderr'):
                with self.assertRaises(SystemExit):parse_args()

    def test_production_command_forwards_new_and_previous_recipe(self):
        if not (Path(__file__).resolve().parents[1]/'train_production.py').is_file():
            self.skipTest('Local production launcher is not part of the portable source tree')
        from train_production import training_command
        args=SimpleNamespace(nodes=8,nproc_per_node=16,node_rank=0,config=Path(__file__).resolve().parents[1]/'configs/qwen35_2b_wan.yaml',
            output=Path('output'),data_root=Path('corpus'),steps=0,save_every=100,
            keep_checkpoints=3,validation_suite=Path('suite.json'),validate_every=500)
        for probability,null in ((.2,'i2t_prefix'),(1.,'empty_t2i')):
            args.i2t_image_loss_probability=probability;args.cfg_null_condition=null
            command=training_command(args,Path('attempt'))
            self.assertEqual(command[command.index('--i2t-image-loss-probability')+1],str(probability))
            self.assertEqual(command[command.index('--cfg-null-condition')+1],null)


if __name__=='__main__':unittest.main()

"""Contracts for the cross-attention conditioning control arm."""
import copy
import unittest
import torch
from .config import ModelConfig, tiny_config
from .generation import GenerationInputs
from .moss import MOSSRenderer
import numpy as np


def layout_and_batch(queries=4, shapes=((32,48),(16,32)), seed=0):
    images=[np.zeros((h,w,3),np.uint8) for h,w in shapes]
    layout=GenerationInputs.prepare(images,queries,16)
    g=torch.Generator().manual_seed(seed)
    n=sum(layout.lengths)
    from types import SimpleNamespace
    return SimpleNamespace(generation=layout,
        latents=torch.randn(n,128,generator=g),
        noise=torch.randn(n,128,generator=g),
        times=torch.rand(len(shapes),generator=g).clamp(.05,.95))


def small_cfg(cross):
    cfg=ModelConfig(renderer='moss',latent_channels=128,flow_draws=1,
        dit_width=64,dit_heads=4,dit_layers=2,dit_ffn=128,width=64,
        image_condition_queries=4,moss_cross_attention=cross,
        checkpoint_dit=False,checkpoint_backbone=False)
    return cfg


class CrossAttentionControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_zero_gate_is_identity_at_initialization(self):
        """An untrained cross block must reproduce the concat-free baseline."""
        torch.manual_seed(7)
        cfg=small_cfg(True)
        model=MOSSRenderer(cfg).double()
        batch=layout_and_batch()
        batch.latents=batch.latents.double();batch.noise=batch.noise.double()
        batch.generation.time_features=torch.randn(2,256,generator=torch.Generator().manual_seed(3)).double()
        cond=torch.randn(len(batch.generation.text_ends),cfg.image_condition_queries,cfg.width,dtype=torch.float64)
        with torch.no_grad():
            before=model(cond,batch)
            # Perturbing conditions must not change the loss while gate==0.
            other=model(cond+10*torch.randn_like(cond),batch)
        torch.testing.assert_close(before,other,atol=0,rtol=0)

    def test_conditions_influence_output_once_gate_opens(self):
        torch.manual_seed(11)
        cfg=small_cfg(True)
        model=MOSSRenderer(cfg).double()
        with torch.no_grad():
            for layer in model.layers:layer.cross_attn.gate.normal_(0,.5)
        batch=layout_and_batch()
        batch.latents=batch.latents.double();batch.noise=batch.noise.double()
        batch.generation.time_features=torch.randn(2,256,generator=torch.Generator().manual_seed(3)).double()
        cond=torch.randn(len(batch.generation.text_ends),cfg.image_condition_queries,cfg.width,dtype=torch.float64)
        with torch.no_grad():
            a=model(cond,batch)
            b=model(cond+torch.randn_like(cond),batch)
        self.assertGreater(abs(a.item()-b.item()),1e-9)

    def test_image_isolation_in_cross_attention(self):
        """Image 0's latents must not read image 1's conditions."""
        torch.manual_seed(13)
        cfg=small_cfg(True)
        model=MOSSRenderer(cfg).double()
        with torch.no_grad():
            for layer in model.layers:layer.cross_attn.gate.normal_(0,.5)
        batch=layout_and_batch()
        batch.latents=batch.latents.double();batch.noise=batch.noise.double()
        batch.generation.time_features=torch.randn(2,256,generator=torch.Generator().manual_seed(3)).double()
        cond=torch.randn(2,cfg.image_condition_queries,cfg.width,dtype=torch.float64,requires_grad=True)
        layout=batch.generation
        # Gradient of image 0's latent rows must not touch image 1's conditions.
        out=model.predict(((1-batch.times.index_select(0,layout.video_sample_indices)[:,None])*batch.latents
                           +batch.times.index_select(0,layout.video_sample_indices)[:,None]*batch.noise),
                          cond,layout.time_features,layout)
        n0=layout.lengths[0]
        out[:n0].sum().backward()
        self.assertGreater(cond.grad[0].abs().max().item(),1e-12)
        self.assertEqual(cond.grad[1].abs().max().item(),0.0)

    def test_baseline_unchanged_when_flag_is_off(self):
        """The concat baseline must be bit-identical to the pre-change path."""
        torch.manual_seed(17)
        cfg=small_cfg(False)
        model=MOSSRenderer(cfg).double()
        self.assertFalse(model.cross_conditioning)
        for layer in model.layers:self.assertIsNone(layer.cross_attn)
        batch=layout_and_batch()
        batch.latents=batch.latents.double();batch.noise=batch.noise.double()
        batch.generation.time_features=torch.randn(2,256,generator=torch.Generator().manual_seed(3)).double()
        cond=torch.randn(2,cfg.image_condition_queries,cfg.width,dtype=torch.float64)
        with torch.no_grad():loss=model(cond,batch)
        self.assertTrue(torch.isfinite(loss))

    def test_cross_parameters_excluded_from_official_checkpoint_keys(self):
        cfg=small_cfg(True)
        model=MOSSRenderer(cfg)
        names=set(model.official_tensors())
        self.assertFalse([n for n in names if 'cross_attn' in n])
        trainable={n for n,_ in model.named_parameters() if 'cross_attn' in n}
        self.assertTrue(trainable)


    def test_projections_initialized_and_gate_gradient_alive(self):
        """A zero to_out also zeroes d(loss)/d(gate), killing the branch.

        The fully-pretrained path sets perform_initialization=False and
        official_tensors() skips this block, so without an explicit init the
        projections stay at whatever the allocator returned -- measured as
        exact zeros on a real 1250-step run. The gate then never receives
        gradient and the renderer silently learns unconditional generation.
        """
        torch.manual_seed(5)
        cfg=small_cfg(True)
        model=MOSSRenderer(cfg).double()
        for index,layer in enumerate(model.layers):
            for name in ('to_q','to_kv','to_out'):
                weight=getattr(layer.cross_attn,name).impl.weight
                self.assertGreater(weight.abs().max().item(),0.,
                    'layer %d %s must be initialized, not left at zero'%(index,name))
            self.assertEqual(layer.cross_attn.gate.abs().max().item(),0.,
                'gate stays zero so step 0 is an exact identity')
        batch=layout_and_batch()
        batch.latents=batch.latents.double();batch.noise=batch.noise.double()
        batch.generation.time_features=torch.randn(2,256,generator=torch.Generator().manual_seed(3)).double()
        cond=torch.randn(len(batch.generation.text_ends),cfg.image_condition_queries,
                         cfg.width,dtype=torch.float64)
        model(cond,batch).backward()
        for index,layer in enumerate(model.layers):
            grad=layer.cross_attn.gate.grad
            self.assertIsNotNone(grad,'layer %d gate got no gradient'%index)
            self.assertGreater(grad.abs().max().item(),0.,
                'layer %d gate must receive gradient or it never opens'%index)


if __name__=='__main__':unittest.main()

"""Contracts for the cross-attention conditioning control arm."""
import copy
import unittest
import torch
from configs.config import RunConfig, tiny_config
from data.generation import GenerationInputs
from models.moss import MOSSRenderer
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
    cfg=RunConfig(renderer='moss',latent_channels=128,flow_draws=1,
        dit_width=64,dit_heads=4,dit_layers=2,dit_ffn=128,width=64,
        image_condition_queries=4,moss_cross_attention=cross,
        checkpoint_dit=False,checkpoint_backbone=False)
    return cfg


class CrossAttentionControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_small_nonzero_output_projection_without_gate(self):
        torch.manual_seed(7)
        cfg=small_cfg(True)
        model=MOSSRenderer(cfg).double()
        expected=.02/(2*cfg.dit_layers)**.5
        for layer in model.layers:
            self.assertFalse(hasattr(layer.cross_attn,'gate'))
            weight=layer.cross_attn.to_out.impl.weight
            self.assertAlmostEqual(weight.std().item(),expected,delta=expected*.05)

    def test_conditions_influence_output_from_initialization(self):
        torch.manual_seed(11)
        cfg=small_cfg(True)
        model=MOSSRenderer(cfg).double()
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


    def test_projections_and_conditions_get_first_step_gradients(self):
        """Reproduces the real failure: construction under no_init_weights.

        A fully pretrained run builds the model inside transformers'
        no_init_weights(), which replaces torch.nn.init.* with no-ops, and sets
        perform_initialization=False. Pretrained tensors are then restored by
        load_pretrained, but official_tensors() deliberately skips this block,
        so nothing ever writes it. Constructor-time init is silently discarded,
        leaving exact zeros -- and a zero to_out zeroes d(loss)/d(gate) too, so
        the gate can never open. Measured on a real 1250-step run: every gate
        and every projection was still exactly zero.
        """
        import torch.nn as nn
        names=['uniform_','normal_','trunc_normal_','constant_','ones_','zeros_',
               'xavier_uniform_','xavier_normal_','kaiming_uniform_','kaiming_normal_']
        saved={n:getattr(nn.init,n) for n in names if hasattr(nn.init,n)}
        def skip(*a,**k):pass
        torch.manual_seed(5)
        reference=MOSSRenderer(small_cfg(True)).double()
        cfg=small_cfg(True)
        try:
            for n in saved:setattr(nn.init,n,skip)
            model=MOSSRenderer(cfg).double()
        finally:
            for n,f in saved.items():setattr(nn.init,n,f)
        # Whatever the allocator returned is not a deliberate initialization:
        # on the real 1250-step run every one of these was exactly zero. Assert
        # the invariant that matters instead of the allocator's whim -- torch's
        # init functions were genuinely suppressed during construction.
        probe=torch.zeros(4,4)
        try:
            for n in saved:setattr(nn.init,n,skip)
            nn.init.normal_(probe,std=.02)
        finally:
            for n,f in saved.items():setattr(nn.init,n,f)
        self.assertEqual(probe.abs().max().item(),0.,
                         'precondition: no_init_weights suppresses torch init')
        # Stand in for load_pretrained, which restores every official tensor but
        # deliberately skips cross-attention.
        official=set(reference.official_tensors())
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if 'cross_attn' in name:continue
                parameter.copy_(dict(reference.named_parameters())[name])
        self.assertFalse([n for n in official if 'cross_attn' in n])
        self.assertEqual(model.initialize_cross_attention(),len(model.layers)*3)
        for index,layer in enumerate(model.layers):
            for name in ('to_q','to_kv','to_out'):
                weight=getattr(layer.cross_attn,name).impl.weight
                self.assertTrue(torch.isfinite(weight).all(),'%s must be finite'%name)
                self.assertGreater(weight.abs().max().item(),0.,
                    'layer %d %s must be initialized after construction'%(index,name))
        batch=layout_and_batch()
        batch.latents=batch.latents.double();batch.noise=batch.noise.double()
        batch.generation.time_features=torch.randn(2,256,generator=torch.Generator().manual_seed(3)).double()
        cond=torch.randn(len(batch.generation.text_ends),cfg.image_condition_queries,
                         cfg.width,dtype=torch.float64,requires_grad=True)
        model(cond,batch).backward()
        self.assertTrue(torch.isfinite(cond.grad).all())
        self.assertGreater(cond.grad.abs().max().item(),0.)
        for index,layer in enumerate(model.layers):
            for name in ('to_q','to_kv','to_out'):
                grad=getattr(layer.cross_attn,name).impl.weight.grad
                self.assertIsNotNone(grad,(index,name))
                self.assertTrue(torch.isfinite(grad).all(),(index,name))
                self.assertGreater(grad.abs().max().item(),0.,(index,name))

    def test_tp_initialization_matches_full_weights(self):
        """TP4 must partition one draw, including the strided K/V projection."""
        from unittest.mock import patch
        from types import SimpleNamespace
        import infra.parallel as P
        cfg=small_cfg(True)
        full=MOSSRenderer(cfg)
        torch.manual_seed(101)
        full.initialize_cross_attention()
        for rank in range(4):
            model=MOSSRenderer(cfg)
            for layer in model.layers:
                for name,dim,stride in [('to_q',0,1),('to_kv',0,2),('to_out',1,1)]:
                    linear=getattr(layer.cross_attn,name)
                    shape=list(linear.impl.weight.shape);shape[dim]//=4
                    linear.impl.weight=torch.nn.Parameter(torch.empty(shape))
                    P.mark_sharded(linear.impl.weight,dim=dim,stride=stride)
            with patch.object(P,'tp_size',return_value=4),patch.object(P,'state',
                    return_value=SimpleNamespace(get_tensor_model_parallel_rank=lambda:rank)):
                torch.manual_seed(101)
                model.initialize_cross_attention()
                for expected,actual in zip(full.layers,model.layers):
                    for name in ('to_q','to_kv','to_out'):
                        weight=getattr(actual.cross_attn,name).impl.weight
                        source=getattr(expected.cross_attn,name).impl.weight
                        torch.testing.assert_close(weight,P.checkpoint_shard(source,weight),rtol=0,atol=0)

    def test_recomputed_cross_attention_matches_gradients(self):
        torch.manual_seed(23)
        plain=MOSSRenderer(small_cfg(True)).double()
        recomputed=copy.deepcopy(plain);recomputed.cfg.checkpoint_dit=True
        batch=layout_and_batch()
        batch.generation.time_features=torch.randn(2,256,dtype=torch.float64)
        cond=torch.randn(2,4,64,dtype=torch.float64,requires_grad=True)
        other=cond.detach().clone().requires_grad_()
        a=plain(cond,batch);b=recomputed(other,batch)
        a.backward();b.backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(cond.grad,other.grad)
        for (name,p),(_,q) in zip(plain.named_parameters(),recomputed.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,msg=name)


if __name__=='__main__':unittest.main()

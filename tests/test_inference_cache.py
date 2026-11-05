import numpy as np
import pytest
import torch
from evaluations.sampling import make_batch
from models.cq import backbone
from models.inference import CQDecoder
from models.modeling_umm import JointModel
from data.vision import NativeImageProcessor
from tests import test_qwen35_vision as vision_fixture
from tests.test_wan_unified import small_config
from models.generator_wan21 import Wan21Generator
from data.generation import WanGenerationInputs


@pytest.mark.parametrize('visibility',['visual_block','causal'])
@pytest.mark.parametrize('fused_projections',[False,True])
@pytest.mark.parametrize('prefix_kind',['text','image','video'])
def test_cached_cq_matches_full_prefix_and_keeps_storage(fused_projections,prefix_kind,visibility):
    NativeVisionContracts=vision_fixture.NativeVisionContracts
    NativeVisionContracts.setUpClass()
    try:
        fixture=NativeVisionContracts()
        cfg=fixture.cfg(layers=8,renderer='wan',latent_channels=16,flow_draws=1,
            dit_width=64,dit_heads=2,dit_ffn=128,dit_layers=2,wan_freq_dim=32,
            qwen_fused_projections=fused_projections,linear_value_heads=4,content_visibility=visibility)
        torch.manual_seed(71);model=JointModel(cfg).eval().double()
        processed=NativeImageProcessor(cfg,128)(np.zeros((64,96,3),np.uint8)) if prefix_kind!='text' else None
        sequence=None
        if prefix_kind=='video':
            from data.sequence import image_sequence
            rgb,pixels,(_,h,w)=processed
            processed=(np.stack([rgb]*5),pixels.repeat(2,1),(2,h,w))
            block=image_sequence((1,h,w),fixture.tokens);n=len(block['ids'])
            # Two native visual time blocks separated by timestamp text.
            sequence=dict(ids=np.concatenate((block['ids'],[17,18],block['ids'])),
                visual=np.concatenate((block['visual'],block['visual']+n+2)),
                sigma=np.concatenate((block['sigma'],[n,n+1],block['sigma']+n+2)),
                positions=np.concatenate((block['positions'],np.full((3,2,1),block['positions'].max()+1),
                    block['positions']+block['positions'].max()+4),axis=1),text_indices=np.empty(0,dtype=np.int64))
        batch=make_batch(cfg,fixture.tokens,[],processed,'I2T',sequence=sequence)
        features=torch.randn(len(batch.vision.content_indices),cfg.width,dtype=torch.float64) if processed is not None else None
        with torch.no_grad():
            expected=backbone(model,batch,vision_features=features)[batch.mainline.output_positions]
            decoder=CQDecoder(model,batch,7,vision_features=features)
            torch.testing.assert_close(decoder.hidden,expected,atol=1e-10,rtol=1e-9)
            pointers=[{k:v.data_ptr() for k,v in c.items() if isinstance(v,torch.Tensor)} for c in decoder.layers]
            tokens=[]
            for token in [7,9,12,5,11,6,13]:
                tokens.append(token)
                fresh=make_batch(cfg,fixture.tokens,tokens,processed,'I2T',sequence=sequence)
                expected=backbone(model,fresh,vision_features=features)[fresh.mainline.output_positions]
                torch.testing.assert_close(decoder.advance(token),expected,atol=1e-10,rtol=1e-9)
            assert pointers==[{k:v.data_ptr() for k,v in c.items() if isinstance(v,torch.Tensor)} for c in decoder.layers]
            with pytest.raises(ValueError,match='capacity'):decoder.advance(7)
    finally:NativeVisionContracts.tearDownClass()


def test_wan_cached_condition_kv_matches_steps_and_skips_projections():
    torch.manual_seed(72);cfg=small_config();model=Wan21Generator(cfg).eval()
    layout=WanGenerationInputs.prepare([np.zeros((32,48,3),np.uint8),
        np.zeros((5,48,32,3),np.uint8)],[3,7],32)
    condition=torch.randn(2,7,32);x=torch.randn(sum(layout.lengths),64)
    with torch.no_grad():
        context=model.project_conditions(condition,layout)
        cached=model.cache_condition_kv(context)
        for time in [torch.randn(2,32) for _ in range(3)]:
            expected=model.predict(x,condition,time,layout)
            calls=[]
            hooks=[b.cross_attn.kv.register_forward_hook(lambda *args:calls.append(True)) for b in model.blocks]
            actual=model.predict(x,None,time,layout,context,cached)
            for hook in hooks:hook.remove()
            assert calls==[]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            x=actual
    with pytest.raises(ValueError,match='inference-only'):
        model.predict(x,None,time,layout,context,cached)

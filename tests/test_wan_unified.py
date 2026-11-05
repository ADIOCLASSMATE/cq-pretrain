import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from configs.config import tiny_config
from data.generation import WanGenerationInputs, patch_latent, unpatch_latent
from models.generator_wan21 import Wan21Generator


def small_config(**kwargs):
    return tiny_config(renderer='wan',width=32,dit_width=64,dit_heads=2,dit_layers=2,
        dit_ffn=128,latent_channels=16,wan_freq_dim=32,flow_draws=1,**kwargs)


def native_state(model):
    result={}
    for name,value in model.state_dict().items():
        name=name.replace('.impl.','.')
        if '.kv.' in name:
            for label,part in zip(('k','v'),value.chunk(2,0)):
                result[name.replace('.kv.',f'.{label}.')]=part.clone()
        else:result[name]=value.clone()
    return result


def test_training_cast_keeps_only_wan_time_layers_in_fp32():
    from models.modeling_umm import JointModel
    from models.backbone_qwen35 import cast_for_training
    for renderer,channels,dtype in [('wan',16,torch.float32),('moss',128,torch.bfloat16)]:
        cfg=replace(small_config(),renderer=renderer,latent_channels=channels)
        model=cast_for_training(JointModel(cfg))
        assert model.renderer.time_embedding[0].weight.dtype==dtype
        assert model.embedding.weight.dtype==torch.bfloat16


def test_packed_wan_matches_diffusers_forward_and_backward():
    # Independent upstream implementation and its official checkpoint converter.
    from unittest.mock import patch
    with patch('diffusers.utils.import_utils._torch_npu_available',False), \
         patch('transformers.utils.import_utils.is_torch_npu_available',return_value=False), \
         patch('transformers.utils.is_torch_npu_available',return_value=False):
        from diffusers import WanTransformer3DModel
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
    torch.manual_seed(41)
    cfg=small_config(); model=Wan21Generator(cfg)
    reference=WanTransformer3DModel(num_attention_heads=2,attention_head_dim=32,in_channels=16,
        out_channels=16,text_dim=32,freq_dim=32,ffn_dim=128,num_layers=2)
    reference.load_state_dict(convert_wan_transformer_to_diffusers(native_state(model)),strict=True)
    layout=WanGenerationInputs.prepare([np.zeros((32,48,3),np.uint8),
        np.zeros((5,48,32,3),np.uint8)],[3,7],32)
    x=torch.randn(sum(layout.lengths),64,requires_grad=True)
    condition=torch.randn(2,7,32,requires_grad=True)
    times=torch.tensor([37.,853.]);phase=times.double()[:,None]*10000**(-torch.arange(16,dtype=torch.float64)/16)
    features=torch.cat((phase.cos(),phase.sin()),-1).float()
    actual=model.predict(x,condition,features,layout)
    xr=x.detach().clone().requires_grad_();cr=condition.detach().clone().requires_grad_()
    expected=[];offset=0
    for i,(grid,n) in enumerate(zip(layout.grids,layout.lengths)):
        z=unpatch_latent(xr[offset:offset+n],grid)
        y=reference(z[None],times[i:i+1],cr[i:i+1,:layout.query_counts[i]],return_dict=False)[0][0]
        expected.append(patch_latent(y));offset+=n
    expected=torch.cat(expected)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-6)
    upstream=torch.randn_like(actual)
    (actual*upstream).sum().backward();(expected*upstream).sum().backward()
    torch.testing.assert_close(x.grad,xr.grad,rtol=2e-4,atol=5e-6)
    torch.testing.assert_close(condition.grad,cr.grad,rtol=3e-4,atol=5e-6)
    torch.testing.assert_close(model.blocks[0].self_attn.norm_q.weight.grad,
        reference.blocks[0].attn1.norm_q.weight.grad,rtol=2e-4,atol=5e-6)
    assert condition.grad[0,3:].count_nonzero()==0


def test_wan_native_checkpoint_loading(tmp_path):
    from safetensors.torch import save_file
    cfg=small_config();model=Wan21Generator(cfg);state=native_state(model)
    (tmp_path/'config.json').write_text(json.dumps(dict(dim=64,ffn_dim=128,num_heads=2,num_layers=2,in_dim=16)))
    save_file(state,str(tmp_path/'diffusion_pytorch_model.safetensors'))
    other=Wan21Generator(cfg);other.load_pretrained(tmp_path)
    for key,value in native_state(other).items():torch.testing.assert_close(value,state[key],rtol=0,atol=0)


@pytest.mark.parametrize('recompute',[False,True])
def test_wan_chunks_preserve_loss_and_every_gradient(recompute):
    import copy
    from types import SimpleNamespace
    from training.renderer_schedule import backward_deferred_renderer
    torch.manual_seed(52)
    cfg=small_config(checkpoint_dit=recompute)
    bridge=torch.nn.Linear(32,32);renderer=Wan21Generator(cfg)
    other_bridge=copy.deepcopy(bridge);other_renderer=copy.deepcopy(renderer)
    layout=WanGenerationInputs.prepare([np.zeros((32,48,3),np.uint8),
        np.zeros((5,48,32,3),np.uint8)],[3,7],32,max_images=1)
    layout.time_features=torch.randn(2,32)
    batch=SimpleNamespace(generation=layout,latents=torch.randn(sum(layout.lengths),64),
        noise=torch.randn(sum(layout.lengths),64),times=torch.tensor([.2,.8]))
    inputs=torch.randn(2,7,32);condition=bridge(inputs)
    expected=condition.square().mean()+.3*cfg.image_weight*renderer(condition,batch)
    (expected*.4).backward()
    actual_condition=other_bridge(inputs);text_loss=actual_condition.square().mean()
    losses=dict(loss=text_loss,text_loss=text_loss,image_conditions=actual_condition)
    visits=[];actual_condition.register_hook(lambda gradient:visits.append(True))
    backward_deferred_renderer(SimpleNamespace(cfg=cfg,renderer=other_renderer),batch,losses,.4,image_factor=cfg.image_weight*.3)
    assert visits==[True]
    torch.testing.assert_close(losses['loss'],expected)
    for reference,actual in ((bridge,other_bridge),(renderer,other_renderer)):
        for (name,p),(_,q) in zip(reference.named_parameters(),actual.named_parameters()):
            torch.testing.assert_close(q.grad,p.grad,rtol=3e-4,atol=3e-6,msg=name)


def test_cache_identity_atomic_resume_and_tamper(tmp_path):
    from data.cache import VisualCache
    cache=VisualCache(tmp_path,'weights-sha',dict(size=32))
    key=cache.identity('image-sha',dict(kind='image'))
    value=torch.randn(4,32,dtype=torch.bfloat16)
    path=cache.save(key,features=value)
    cache.save(key,features=torch.zeros_like(value))
    torch.testing.assert_close(cache.load(key)['features'],value)
    assert cache.load(cache.identity('another-image',dict(kind='image'))) is None
    assert VisualCache(tmp_path,'changed-weights',dict(size=32)).load(
        dict(key,encoder='changed-weights')) is None
    payload=torch.load(path,weights_only=True);payload['tensors']['features'][0,0]=100
    torch.save(payload,path)
    with pytest.raises(ValueError,match='checksum'):cache.load(key)


@pytest.mark.parametrize("visibility", ["visual_block", "causal"])
def test_video_blocks_and_understanding_queries(visibility):
    from data.batch import Batch
    from data.planning import build_visual_plan
    from types import SimpleNamespace
    cfg=small_config(visual_query_mode='native',content_visibility=visibility)
    s=16; spans=torch.tensor([[2,3,4,5,7,8,9,10],[12,13,-1,-1,-1,-1,-1,-1]])
    sigma=torch.arange(s)[None];sigma[0,2:6]=2;sigma[0,7:11]=7;sigma[0,12:14]=12
    empty=torch.empty(0)
    batch=Batch(torch.zeros(1,s,dtype=torch.long),empty,empty,spans,torch.tensor([1,11,15]),
        torch.zeros(1,s,dtype=torch.bool),sigma,torch.zeros(1,s,dtype=torch.long),
        torch.arange(s).float()[None,:,None].expand(3,-1,-1).clone(),empty,empty,empty,empty,
        vision=SimpleNamespace(grid_thw=((2,4,4),(1,2,4))),generation_image_indices=torch.tensor([1]))
    original_sigma=batch.sigma.clone()
    plan=build_visual_plan(batch,cfg)
    torch.testing.assert_close(batch.sigma,original_sigma,rtol=0,atol=0)
    assert plan.query_image.sum()==2
    assert plan.condition_indices.shape==(2,2) # An understanding video cannot increase target Q padding.
    assert plan.output_size==s+2
    assert bool(plan.cq_mask[0,0,2,5]) == (visibility=='causal')
    assert plan.cq_mask[0,0,2,7] # no future block
    assert not plan.cq_mask[0,0,7,2] # past block visible
    assert plan.cq_mask[0,0,s+len(batch.text_indices):,12:].all() # target content is invisible to its queries
    # Exercise the complete three-GDN/one-attention stack, beyond mask inspection.
    from models.backbone_qwen35 import QwenBlock
    torch.manual_seed(3)
    blocks=torch.nn.ModuleList([QwenBlock(cfg,i) for i in range(4)])
    content=torch.randn(s+len(plan.query_positions),1,cfg.width)
    def forward(value):
        for block in blocks:value=block(value,(plan.rope_cos,plan.rope_sin),plan,s)
        return value
    with torch.no_grad():
        expected=forward(content)
        changed=content.clone();changed[7:11]+=torch.randn_like(changed[7:11])
        torch.testing.assert_close(forward(changed)[2:6],expected[2:6],rtol=0,atol=0)
        changed=content.clone();changed[5]+=torch.randn_like(changed[5])
        actual=forward(changed)
        if visibility=='visual_block':assert not torch.allclose(actual[2],expected[2])
        else:torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
        assert not torch.allclose(actual[7],expected[7]) # earlier block is visible
        changed=content.clone();changed[12:14]+=torch.randn_like(changed[12:14])
        torch.testing.assert_close(forward(changed)[s+len(batch.text_indices):],
                                   expected[s+len(batch.text_indices):],rtol=0,atol=0)
    batch.generation_image_indices=torch.empty(0,dtype=torch.long)
    plan=build_visual_plan(batch,cfg)
    assert not plan.query_image.any() and plan.output_size==s
    assert plan.condition_indices.shape==(2,0)
    # A video target uses the same flag as an image, across all native time blocks.
    batch.generation_image_indices=torch.tensor([0,1])
    batch.generation_unconditional=torch.tensor([True,False])
    plan=build_visual_plan(batch,cfg)
    assert plan.query_unconditional.sum()==8
    content=torch.randn(s+len(plan.query_positions),1,cfg.width)
    with torch.no_grad():
        expected=forward(content)
        changed=content.clone();changed[:s]+=torch.randn_like(changed[:s])
        actual=forward(changed)
        null_rows=s+plan.query_unconditional.nonzero().flatten()
        normal_rows=s+(plan.query_image[0]&~plan.query_unconditional).nonzero().flatten()
        torch.testing.assert_close(actual[null_rows],expected[null_rows],rtol=0,atol=0)
        assert not torch.allclose(actual[normal_rows],expected[normal_rows])


def test_native_video_serialization_and_mixed_source_resume(tmp_path):
    from pathlib import Path
    from PIL import Image
    from data.vision import native_processor
    from data.sequence import video_sequence, native_video_metadata
    from data.video import prepare_qwen_inputs
    from data.stream import DocumentStream
    from data.packing import Packer, token_contract
    assets=Path(__file__).resolve().parents[1]/'assets/Qwen3.5-0.8B'
    if not (assets/'tokenizer.json').exists():pytest.skip('Local native tokenizer fixture required')
    processor=native_processor(assets)
    rgb=torch.zeros(5,3,64,96,dtype=torch.uint8)
    clip=dict(timestamps=[0.,.12,.2,.31,.43],fps=10)
    settings=dict(mode='native',qwen_num_frames=4)
    pixels,grid,times=prepare_qwen_inputs(rgb,clip,processor,settings)
    sequence=video_sequence(processor,tuple(grid.tolist()),times)
    selected=torch.tensor([0,1,3,4])
    official=processor(text=[processor.video_token],videos=[rgb[selected]],do_sample_frames=False,
        do_resize=False,video_metadata=[native_video_metadata(times)],return_tensors='pt')
    np.testing.assert_array_equal(sequence['ids'],official['input_ids'][0].numpy())
    np.save(tmp_path/'frames.npy',rgb.permute(0,2,3,1).numpy())
    Image.fromarray(np.zeros((64,96,3),np.uint8)).save(tmp_path/'image.png')
    rows=[dict(task='v2t',caption='A dark video.',frames_path='frames.npy',fps=10),
          dict(task='t2v',caption='Another video.',frames_path='frames.npy',fps=10),
          dict(task='t2i',caption='A dark image.',image_path='image.png')]
    (tmp_path/'items.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    import yaml
    source=tmp_path/'sources.yaml'
    source.write_text(yaml.safe_dump(dict(sources=[dict(path='items.jsonl',video=settings)])))
    cfg=small_config(vision_encoder='qwen35',vision_assets=str(assets),text_assets=str(assets),
        vocab_size=248320,vision_min_pixels=1024,vision_max_pixels=6144,generation_max_pixels=6144,
        preserve_documents=True,image_condition_queries=7)
    def stream(state=None):
        return DocumentStream(source,'A',0,1,42,cfg=cfg,native_images=True,sequence_length=512,
            workers=1,prefetch=1,image_threads=1,block_rows=1,tokenize_rows=1,state=state)
    docs=stream()
    try:
        first=docs.peek();docs.advance();state=docs.state_dict()
        following=docs.peek()['id']
    finally:docs.close()
    docs=stream(state)
    try:assert docs.peek()['id']==following
    finally:docs.close()
    docs=stream()
    try:
        batch,receipt=Packer(docs,cfg,512,token_contract(source,cfg)).next()
        assert len(batch.image_indices)==3 and batch.generation_image_count==2
        assert batch.mainline.query_image.sum()==14
        assert sorted(batch.generation.grids)==[(1,4,6),(2,4,6)]
        assert len(batch.vision.ends)==5 # 2 + 2 native video blocks, one image.
        from data.planning import build_visual_plan
        native=build_visual_plan(batch,replace(cfg,visual_query_mode='native'))
        expected=torch.cat([batch.row_col[:,batch.image_indices[i][batch.image_indices[i]>=0]]
                            for i in batch.generation_image_indices.tolist()],1)
        torch.testing.assert_close(native.query_rope[:,native.query_image[0]],expected,rtol=0,atol=0)
        from data.codec import DevicePreparer
        receipt['signature']=docs.signature
        batch=DevicePreparer(cfg,device='cpu',encoder='fake')(batch,receipt)
        assert batch.latents.shape==(72,64)
        assert torch.isfinite(batch.latents).all()
    finally:docs.close()


def test_adapted_native_checkpoint_keeps_bias_and_second_layer(tmp_path):
    from safetensors.torch import save_file
    cfg=small_config()
    native=Wan21Generator(replace(cfg,width=96))
    state=native_state(native)
    (tmp_path/'config.json').write_text(json.dumps(dict(dim=64,ffn_dim=128,
        num_heads=2,num_layers=2,in_dim=16)))
    save_file(state,str(tmp_path/'diffusion_pytorch_model.safetensors'))
    adapted=Wan21Generator(cfg)
    report=adapted.load_pretrained(tmp_path)
    assert report['initialized']==['text_embedding.0.weight']
    assert adapted.text_embedding[0].weight.shape==(64,32)
    assert adapted.text_embedding[2].weight.shape==(64,64)
    for name,value in native_state(adapted).items():
        if name!='text_embedding.0.weight':
            torch.testing.assert_close(value,state[name],rtol=0,atol=0)
    assert not any(name.startswith('condition_proj') for name,_ in adapted.named_parameters())

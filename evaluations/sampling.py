"""Small deterministic CQ validation, sharded by DP with TP groups intact.

No reference captions enter generation. This is a qualitative regression suite,
not an official GenEval/COCO score, nor a claim of decontaminated held-out data.
MOSS schedule/latent decode adapted from OpenMOSS-Team/MOSS-Image-Base
1ceede80badddd6a7812069619edaa5633587192 (Apache-2.0).
"""
from dataclasses import replace
import hashlib
import html
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from PIL import Image, ImageOps

import distributed.megatron as P
from distributed.accel import accel_type, device_module
from models.cq_backbone import backbone
from data.batch_layout import build_plan, content_view, finalize_plan
from data.packing import Packer, token_contract
import data.chat as chat


def video_input(cfg,case):
    from data.video import prepare_video, prepare_qwen_inputs
    from data.sequence import video_sequence
    from data.vision import native_processor
    settings=case.get('video_settings',{'mode':'native'})
    if case['task']=='T2V':
        n=case['num_frames'];fps=case.get('fps',16)
        if n<5 or (n-1)%4:raise ValueError('T2V frames must be 4k+1 and at least 5')
        rgb=torch.zeros(n,3,case['height'],case['width'],dtype=torch.uint8)
        clip=dict(timestamps=(np.arange(n)/fps).tolist(),fps=fps)
    else:
        video=case.get('video',dict(path=case.get('video_path')))
        rgb,clip=prepare_video(video,settings)
        if case.get('video_sha256') and clip['source_sha256']!=case['video_sha256']:
            raise ValueError('Validation video changed')
    processor=native_processor(cfg.vision_assets)
    pixels,grid,times=prepare_qwen_inputs(rgb,clip,processor,settings)
    grid=tuple(grid.tolist())
    return (rgb.permute(0,2,3,1).numpy(),pixels,grid),video_sequence(processor,grid,times)


def save_visual(frames,path,fps):
    if len(frames)==1:
        Image.fromarray(frames[0]).save(path);return
    import av
    from fractions import Fraction
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=Fraction(str(fps)))
        stream.width=frames.shape[2];stream.height=frames.shape[1];stream.pix_fmt='yuv420p'
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


def moss_sigmas(steps,shift=5.,total=1000):
    if steps<1 or shift<=0:raise ValueError('Positive steps and shift required')
    def grid(count,amount):
        values=torch.linspace(1,amount/(total+amount-1),count)
        return amount*values/(1+(amount-1)*values)
    values=grid(steps,shift)
    training=grid(total,1.).flip(0)
    upper=torch.searchsorted(training,values).clamp(max=total-1)
    lower=(upper-1).clamp(min=0)
    nearest=torch.where((values-training[lower]).abs()<=(training[upper]-values).abs(),lower,upper)
    return torch.cat((torch.unique_consecutive(training[nearest]),torch.zeros(1)))


def prune_queries(batch,cfg,keep):
    """CQ never writes KV/GDN state: retain only the requested readouts."""
    plan=batch.mainline;s=batch.ids.shape[1]
    keep=keep.nonzero().flatten() if keep.dtype==torch.bool else keep
    rows=torch.cat((torch.arange(s),s+keep))
    same_rows = len(keep) == len(plan.query_positions) and torch.equal(keep, torch.arange(len(keep)))
    plan=replace(plan,query_positions=plan.query_positions[keep],query_image=plan.query_image[:,keep],
        query_rope=plan.query_rope[:,keep],output_positions=plan.output_positions[keep],
        query_unconditional=(plan.query_unconditional[keep] if plan.query_unconditional is not None else None),
        query_read_sigma=(plan.query_read_sigma[keep] if plan.query_read_sigma is not None else None),
        # Pruned readouts historically use the dense/tiled kernel. The full
        # predicate has different row indices and must not be reused here.
        cq_predicate=plan.cq_predicate if same_rows else None,
        cq_mask=plan.cq_mask[:,:,rows].contiguous(),cq_valid=plan.cq_valid[:,rows],cq_plan=None)
    batch.mainline=finalize_plan(content_view(batch,cfg),cfg,plan)
    return batch


class OneDocument:
    images_preprocessed=True
    def __init__(self,elements,processed=None):
        self.doc=dict(id='validation',elements=elements);self.processed=processed
    def peek(self):return self.doc
    def advance(self):self.doc=None
    def image(self,doc,element):
        # Editing cases carry one prepared image per image element.
        return self.processed[element] if isinstance(self.processed,dict) else self.processed
    def prefetch_images(self,*args):pass
    def state_dict(self):return {}


def encode_spans(tokenizer,pieces):
    """Tokenize scaffolding exactly as the training pipeline does."""
    return [np.asarray(tokenizer.encode(piece,add_special_tokens=False).ids,dtype=np.int32)
            for piece in pieces]

def context_image(path, checksum, processor):
    source = Path(path)
    if checksum is not None:
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError('Validation image changed')
        source = io.BytesIO(payload)
    with Image.open(source) as image:
        return processor(np.asarray(ImageOps.exif_transpose(image).convert('RGB')))


def encode_history(case, tokenizer, processor):
    """Earlier turns use native role/turn/kind segments and indices into case.images."""
    segments = []
    for segment in case.get('history', ()):
        if segment.get('role') not in ('system','user','assistant','tool') or 'turn' not in segment:
            raise ValueError('History segments need an explicit role and turn')
        if segment.get('kind') not in ('text','image'):
            raise ValueError('History segments must contain text or an image')
        segments.append(dict(segment, supervised=True, image_loss_allowed=False))
    elements, decoded = [], {}
    for span in chat.causal_spans(segments):
        if span['kind'] == 'image':
            index = span['image_index']
            images = case.get('images', ())
            if type(index) is not int or not 0 <= index < len(images):
                raise ValueError('History image_index is outside case.images')
            if index not in decoded:
                item = images[index]
                decoded[index] = context_image(item['path'], item.get('sha256'), processor)
            elements.append(dict(type='image', generation_target=False, image_loss_allowed=False, prepared=decoded[index]))
        else:
            ids, = encode_spans(tokenizer, [span['text']])
            if len(ids):
                elements.append(dict(type='text', input_ids=ids, supervised=span['supervised'],
                                     context_text=not span['synthetic'] and bool(span['text'].strip()),
                                     instruction_start=span.get('instruction_start', False),
                                     instruction_text=span.get('instruction_text', False)))
    return elements


def make_batch(cfg,tokens,text_ids,processed,task,prune=True,tokenizer=None,instruction=None,sequence=None,non_cfg=False,context=None,history=()):
    """Build one validation pack under the Qwen3.5 chat template.

    Training wraps every caption in user/assistant messages, so evaluation must
    use the same scaffolding or the model is probed off-distribution. The
    instruction is fixed rather than hashed: validation must stay comparable
    across steps.
    """
    body=np.asarray(text_ids,dtype=np.int32)
    def element(ids,supervised=False,instruction_start=False):
        return dict(type='text',input_ids=ids,supervised=supervised,
                    context_text=supervised,instruction_start=instruction_start)
    picture=dict(type='image',generation_target=task in ('T2I','T2V'),non_cfg=non_cfg)
    if sequence is not None:picture['video_sequence']=sequence
    user_header = (chat.IM_START if history else '') + chat.ROLE_USER
    if context is not None or history:
        if tokenizer is None or task not in ('T2I','T2V') or processed is None:
            raise ValueError('Contextual generation requires a templated generation target')
        head,end,user,assistant=encode_spans(tokenizer,[user_header,chat.TURN_END,
            chat.IM_START+chat.ROLE_USER,chat.IM_START+chat.ROLE_ASSISTANT])
        closing=dict(element(end),supervised=True)
        elements=[element(head)]
        if context is not None:
            source=dict(type='image',generation_target=False,image_loss_allowed=False,prepared=context)
            elements.extend([source,closing,element(user)])
        elements.extend([element(body,True,instruction_start=True),closing,element(assistant),picture,closing])
    elif processed is None:
        elements=[element(body,True)]
    elif tokenizer is None:
        # Legacy untemplated layout, retained only for direct unit tests.
        elements=[element(body,True),picture] if task in ('T2I','T2V') else [picture,element(body,True)]
    elif task in ('T2I','T2V'):
        head,tail=encode_spans(tokenizer,[
            chat.ROLE_USER+(instruction or ('' if task=='T2V' else chat.T2I_INSTRUCTIONS[0])),
            chat.TURN_END+chat.IM_START+chat.ROLE_ASSISTANT])
        elements=[element(head),element(body,True),element(tail),picture]
    else:
        head,tail=encode_spans(tokenizer,[
            chat.ROLE_USER,
            (instruction or chat.I2T_INSTRUCTIONS[0])+chat.TURN_END+
            chat.IM_START+chat.ROLE_ASSISTANT])
        elements=[element(head),picture,element(tail),element(body,True)]
    elements=[e for e in (*history, *elements) if e['type']!='text' or len(e['input_ids'])]
    if context is not None or history:
        processed={i:processed if e is picture else e['prepared'] for i,e in enumerate(elements) if e['type']=='image'}
    docs=OneDocument(elements,processed)
    docs.doc['direction']=task
    # Fits the fixed validation examples and does not consume training data.
    image_tokens=0
    for i,e in enumerate(elements):
        if e['type']=='image':
            visual=processed[i] if isinstance(processed,dict) else processed
            image_tokens+=len(e['video_sequence']['ids']) if 'video_sequence' in e else math.prod(visual[2])//4+2
    scaffold=sum(len(e['input_ids']) for e in elements if e['type']=='text')
    length=max(scaffold+image_tokens+2,cfg.vision_min_pixels//1024+4)
    packer=Packer(docs,replace(cfg,visual_cache='',visual_uncond_probability=0.,visual_instruction_dropout_probability=0.),
                  length,tokens,padding_multiple=0)
    batch,receipt=packer.next()
    if not receipt['documents'][0]['completed']:raise ValueError('Validation example exceeds its pack')
    used=receipt['occupied']
    # Remove the unused physical tail before constructing the inference plan.
    batch.ids=batch.ids[:,:used];batch.segments=batch.segments[:,:used];batch.sigma=batch.sigma[:,:used]
    batch.image_queries=batch.image_queries[:,:used];batch.row_col=batch.row_col[:,:used]
    batch.mainline=build_plan(batch,cfg)
    if not prune:return batch
    if (context is not None or history) and batch.generation_image_count!=1:
        raise ValueError('A contextual generation case must have exactly one generation target')
    if task in ('T2I','T2V'):keep=batch.mainline.query_image[0]
    else:keep=batch.mainline.query_positions==used-1
    return prune_queries(batch,cfg,keep)


def make_null_batch(cfg,tokens,processed,tokenizer,sequence=None):
    """Match training: empty visual-Q reads, or an explicit legacy template."""
    mode=cfg.cfg_null_condition
    if mode=='empty':
        return make_batch(cfg,tokens,[],processed,'T2I',tokenizer=tokenizer,sequence=sequence,non_cfg=True)
    if mode=='empty_t2i':
        return make_batch(cfg,tokens,[],processed,'T2I',tokenizer=tokenizer,sequence=sequence)
    if mode=='i2t_prefix':
        batch=make_batch(cfg,tokens,[],processed,'I2T',prune=False,tokenizer=tokenizer)
        return prune_queries(batch,cfg,batch.mainline.query_image[0])
    raise ValueError(f'Unknown CFG null condition: {mode}')


def make_history_batch(full, cfg):
    """Move the visual CQ reads in the positive pack; share tokens, pixels and positions."""
    if full.generation_instruction_starts is None or not full.generation_image_count:
        raise ValueError('History guidance needs a generation target and a known instruction boundary')
    dropped = full.generation_instruction_starts.ge(0)
    selected = full.generation_image_indices
    if not (dropped if selected is None else dropped[selected]).all():
        raise ValueError('History guidance needs an instruction with a nonempty retained prefix')
    batch = replace(full, generation_instruction_dropped=dropped)
    batch.mainline = build_plan(batch, cfg)
    return prune_queries(batch, cfg, batch.mainline.query_image[0])


@torch.no_grad()
def score_reference(model,case,tokenizer,tokens,processor,vae):
    """Fixed teacher-forced caption NLL and one fixed-noise conditional flow loss."""
    if not case.get('references'):return {}
    cfg=model.cfg
    processed=context_image(case['image'],case.get('image_sha256'),processor)
    ids=tokenizer.encode(case['references'][0],add_special_tokens=False).ids
    batch=make_batch(cfg,tokens,ids,processed,'I2T',prune=False,tokenizer=tokenizer).to(accel_type())
    hidden=backbone(model,batch)
    indices=batch.text_indices
    indices=indices[(indices>batch.image_indices.max()+1)&(indices<batch.ids.shape[1]-1)]
    targets=batch.ids.flatten()[indices]
    from ops.cross_entropy import linear_cross_entropy
    weight=model.lm_head.weight if model.lm_head is not None else model.embedding.weight
    nll=linear_cross_entropy(hidden[indices],weight,targets,cfg.ce_chunk)
    if cfg.renderer=='wan':return dict(reference_text_nll=float(nll),reference_text_tokens=len(targets))
    batch=make_batch(cfg,tokens,ids,processed,'T2I',tokenizer=tokenizer).to(accel_type())
    empty=torch.zeros(len(batch.vision.content_indices),cfg.width,device=accel_type(),dtype=model.embedding.weight.dtype)
    hidden=backbone(model,batch,vision_features=empty)
    condition=batch.mainline.gather_image_conditions(hidden,batch.generation_image_indices)
    pixels=torch.from_numpy(processed[0].transpose(2,0,1).copy())[None].to(accel_type()).float().div(127.5).sub(1)
    mean,std=vae.encode(pixels)
    rng=torch.Generator(device='cpu').manual_seed(case.get('seed',1234))
    target=mean.flatten(2).transpose(1,2)[0]+std.flatten(2).transpose(1,2)[0]*torch.randn(sum(batch.generation.lengths),cfg.latent_channels,generator=rng).to(accel_type())
    batch.latents=target
    batch.noise=torch.randn(target.shape,generator=rng).to(accel_type())
    batch.times=torch.tensor([.5],device=accel_type())
    phase=500.*10000**(-torch.arange(128,dtype=torch.float64)/128)
    batch.generation.time_features=torch.cat((phase.cos(),phase.sin()))[None].to(device=accel_type(),dtype=torch.float32 if cfg.renderer=='wan' else torch.bfloat16)
    loss=model.renderer(condition,batch)
    return dict(reference_text_nll=nll.item(),reference_text_tokens=len(targets),conditional_flow_mse=loss.item(),
                fixed_flow_time=.5,fixed_noise_seed=case.get('seed',1234))


def greedy_token(model,hidden):
    weight=model.lm_head.weight if model.lm_head is not None else model.embedding.weight
    logits=F.linear(hidden,weight).float().flatten()
    local=logits.argmax()
    tp_rank=P.state().get_tensor_model_parallel_rank() if P.state() else 0
    pair=torch.stack((logits[local],(local+tp_rank*weight.shape[0]).float()))
    if P.tp_size()>1:
        pairs=[torch.empty_like(pair) for _ in range(P.tp_size())]
        dist.all_gather(pairs,pair,group=P.state().get_tensor_model_parallel_group())
    else:pairs=[pair]
    values=[p.tolist() for p in pairs]
    return int(min(values,key=lambda x:(-x[0],x[1]))[1])


@torch.no_grad()
def caption(model,case,tokenizer,tokens,processor):
    cfg=model.cfg
    if case['task']=='V2T':
        processed,sequence=video_input(cfg,case)
        video=case.get('video',{})
        source=video.get('frames_path') or video.get('path') or case.get('video_path','')
    else:
        path=Path(case['image'])
        if not case.get('image_sha256'):raise ValueError('Validation caption image requires its SHA-256')
        processed=context_image(path,case['image_sha256'],processor)
        sequence=None;source=case['image']
    emitted=[]
    # A suite prompt is a user instruction, not the start of the answer. Feeding
    # it as an assistant prefix made the model read its own instruction back and
    # immediately emit <|im_end|>, yielding an empty caption.
    instruction=case.get('prompt','').strip() or None
    reason='max_new_tokens'
    stops={tokens['eos'],tokens['boi']}
    end=tokenizer.token_to_id('<|im_end|>')
    if end is not None:stops.add(end)
    from models.decoding import CQDecoder
    limit=case.get('max_new_tokens',48)
    cached=cfg.method=='CQ_IT';features=None
    if limit and cached:
        batch=make_batch(cfg,tokens,[],processed,'I2T',tokenizer=tokenizer,
                         instruction=instruction,sequence=sequence)
        decoder=CQDecoder(model,batch,limit)
    for step in range(limit):
        if cached:hidden=decoder.hidden
        else:
            batch=make_batch(cfg,tokens,emitted,processed,'I2T',tokenizer=tokenizer,
                instruction=instruction,sequence=sequence).to(accel_type())
            if features is None:features=model.patch_embed(batch.vision)
            hidden=backbone(model,batch,vision_features=features)[batch.mainline.output_positions]
        token=greedy_token(model,hidden)
        if token in stops:
            reason=f'stop_token_{token}';break
        emitted.append(token)
        if cached and step+1<limit:decoder.advance(token)
    return dict(text=tokenizer.decode(emitted,skip_special_tokens=False),token_ids=emitted,
                stop_reason=reason,reference_captions=case.get('references',[]),
                image=source,prompt=case.get('prompt',''))


@torch.no_grad()
def generate(model,case,tokenizer,tokens,processor,vae,output):
    """Wan cases can set s_ctx and s_ins independently; each defaults to guidance."""
    cfg=model.cfg;context=None
    if case['task']=='EDIT':
        context=context_image(case['image'],case.get('image_sha256'),processor)
        # Training targets share the input's processed size in this release.
        height,width=context[0].shape[:2]
    else:height,width=case['height'],case['width']
    task='T2I' if context is not None else case['task']
    if min(height,width)<16 or height%16 or width%16 or height*width>cfg.generation_max_pixels:
        raise ValueError('Validation shape violates the training pixel/alignment cap')
    processed,sequence=(video_input(cfg,case) if case['task']=='T2V' else
                        (processor(np.zeros((height,width,3),np.uint8)),None))
    guidance=case.get('guidance',1. if cfg.renderer=='wan' else 3.)
    s_ctx, s_ins = case.get('s_ctx', guidance), case.get('s_ins', guidance)
    limit = torch.finfo(torch.float32).max  # guidance is applied to FP32 velocities
    if not all(type(s) in (int, float) and -limit <= s <= limit for s in (guidance, s_ctx, s_ins)):
        raise ValueError('Guidance scales must be finite FP32 numbers')
    split = s_ctx != s_ins
    if split and (cfg.renderer != 'wan' or cfg.cfg_null_condition != 'empty'):
        raise ValueError('Independent guidance scales require Wan with an empty null condition')
    prefix = encode_history(case, tokenizer, processor)
    prompt_ids = tokenizer.encode(case['prompt'], add_special_tokens=False).ids
    full = make_batch(cfg, tokens, prompt_ids, processed, task, tokenizer=tokenizer,
                      sequence=sequence, context=context, history=prefix)
    # With no history v_hist = v_null; with no instruction v_hist = v_full.
    scale = s_ins
    history_batch = None
    has_history = context is not None or any(e['type']=='image' or e.get('context_text') for e in prefix)
    if split and has_history:
        if case['prompt'].strip() and len(prompt_ids):
            history_batch = make_history_batch(full, cfg)
        else:
            scale = s_ctx

    def condition(batch, features=None):
        if features is None:
            features = model.embedding.weight.new_zeros((len(batch.vision.content_indices), cfg.width))
        hidden = backbone(model, batch, vision_features=features)
        raw = batch.mainline.gather_image_conditions(hidden, batch.generation_image_indices)
        projected = model.renderer.project_conditions(raw, batch.generation)
        kv = dict(cached_kv=model.renderer.cache_condition_kv(projected)) if cfg.renderer == 'wan' else {}
        return raw, projected, kv

    full = full.to(accel_type())
    layout = full.generation
    # Both positive branches see the very same source features, computed once.
    features = model.patch_embed(full.vision) if len(full.image_indices) > full.generation_image_count else None
    positive = condition(full, features)
    negative = None
    if history_batch is not None or scale != 1.:
        negative = condition(make_null_batch(cfg,tokens,processed,tokenizer,sequence=sequence).to(accel_type()))
    history = condition(history_batch.to(accel_type()), features) if history_batch is not None else None
    del full, history_batch, features

    def predict(branch, latent, time):
        raw, projected, kv = branch
        return model.renderer.predict(latent, raw, time, layout, projected, **kv).float()

    generator=torch.Generator(device='cpu').manual_seed(case['seed'])
    latent=torch.randn(sum(layout.lengths),cfg.latent_channels*(4 if cfg.renderer=='wan' else 1),generator=generator).to(accel_type())
    sigmas=moss_sigmas(case.get('steps',30),case.get('shift',5.))
    if cfg.renderer=='wan':
        base=torch.linspace(1.,0.,case.get('steps',30)+1);shift=case.get('shift',5.)
        sigmas=shift*base/(1+(shift-1)*base)
    half=cfg.wan_freq_dim//2 if cfg.renderer=='wan' else 128
    phase=(sigmas[:-1]*1000).double()[:,None]*10000**(-torch.arange(half,dtype=torch.float64)/half)
    time_features=torch.cat((phase.cos(),phase.sin()),-1).to(device=accel_type(),dtype=torch.float32 if cfg.renderer=='wan' else torch.bfloat16)
    for i,(start,end) in enumerate(zip(sigmas[:-1],sigmas[1:])):
        time = time_features[i:i+1]
        velocity = predict(positive, latent, time)
        if negative is not None:
            uncond = predict(negative, latent, time)
            if history is None:
                velocity = uncond + scale * (velocity - uncond)
            else:
                hist = predict(history, latent, time)
                velocity = uncond + s_ctx * (hist - uncond) + s_ins * (velocity - hist)
        latent=latent+(float(end)-float(start))*velocity
    if not torch.isfinite(latent).all().item():raise FloatingPointError('Nonfinite generated latent')
    if cfg.renderer=='wan':
        from data.generation import unpatch_latent
        decoded=vae.decode([unpatch_latent(latent,layout.grids[0])])[0]
        if not torch.isfinite(decoded).all().item():raise FloatingPointError('Nonfinite decoded visual')
        frames=decoded.clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8).permute(1,2,3,0).cpu().numpy()
        suffix='png' if len(frames)==1 else 'mp4';name=f"{case['id']}.{suffix}"
        if P.state().get_tensor_model_parallel_rank()==0:
            save_visual(frames,output/name,case.get('fps',16))
        return dict(image=name,prompt=case['prompt'],height=height,width=width,num_frames=len(frames),
            seed=case['seed'],steps=len(sigmas)-1,guidance=guidance,latent_std=latent.std().item(),
            **(dict(s_ctx=s_ctx, s_ins=s_ins) if 's_ctx' in case or 's_ins' in case else {}),
            **({'input':case['image']} if context is not None else {}))
    h,w=layout.grids[0]
    z=latent.T.reshape(1,cfg.latent_channels,h,w).float()
    bn=vae.bn;eps=vae.config.batch_norm_eps
    z=z*(bn.running_var+eps).sqrt().view(1,-1,1,1)+bn.running_mean.view(1,-1,1,1)
    z=z.reshape(1,cfg.latent_channels//4,2,2,h,w).permute(0,1,4,2,5,3).reshape(1,cfg.latent_channels//4,2*h,2*w)
    vae.decoder.to(accel_type());vae.post_quant_conv.to(accel_type())
    try:
        pixels=vae.decode(z.to(vae.dtype),return_dict=False)[0]
        if not torch.isfinite(pixels).all().item():raise FloatingPointError('Nonfinite decoded pixels')
        pixels=(pixels.clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8)[0].permute(1,2,0).cpu().numpy())
    finally:
        device_module().synchronize()
        vae.decoder.to('cpu');vae.post_quant_conv.to('cpu')
    if P.state().get_tensor_model_parallel_rank()==0:Image.fromarray(pixels).save(output/f"{case['id']}.png")
    return dict(image=f"{case['id']}.png",prompt=case['prompt'],height=height,width=width,
                seed=case['seed'],steps=len(sigmas)-1,guidance=guidance,shift=case.get('shift',5.),
                cfg_null_condition=cfg.cfg_null_condition,
                latent_std=latent.std().item(),latent_mean=latent.mean().item())


def report_html(output,rows,weight_sources=None):
    parts=['<!doctype html><meta charset="utf-8"><title>CQ validation</title>',
           '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}section{display:inline-block;width:48%;vertical-align:top;padding:1%}img{max-width:100%;max-height:400px}</style>',
           '<h1>Fixed CQ validation</h1><p>Qualitative examples; not benchmark scores.</p>']
    if weight_sources:parts.append('<p>Weights: '+html.escape(json.dumps(weight_sources))+'</p>')
    for row in rows:
        parts.append('<section><h2>'+html.escape(row['id'])+'</h2>')
        source=row['image']
        if row.get('input'):
            import shutil
            name=row['id']+'_input'+Path(row['input']).suffix
            if Path(row['input']).resolve()!=(output/name).resolve():shutil.copyfile(row['input'],output/name)
            parts.append('<img src="'+html.escape(name,quote=True)+'" style="max-height:200px"><p>input</p>')
        if row['task'] in ('I2T','V2T'):
            import shutil
            name=row['id']+Path(source).suffix
            if Path(source).resolve()!=(output/name).resolve():shutil.copyfile(source,output/name)
            source=name
        escaped=html.escape(source,quote=True)
        media=('<video controls src="'+escaped+'"></video>' if row['task'].endswith('V') or row['task']=='V2T'
               else '<img src="'+escaped+'">')
        parts.append(media+'<p>'+html.escape(row.get('prompt',''))+'</p>')
        if 'text' in row:parts.append('<p>'+html.escape(row['text'])+'</p>')
        if row.get('reference_captions'):parts.append('<details><summary>References (not model input)</summary>'+html.escape(' | '.join(row['reference_captions']))+'</details>')
        parts.append('</section>')
    (output/'index.html').write_text(''.join(parts))


@torch.no_grad()
def run_suite(model,suite_path,output,preparer,weight_sources=None):
    from tokenizers import Tokenizer
    from data.vision import NativeImageProcessor
    if model.cfg.method!='CQ_IT':raise ValueError('This validator implements CQ_IT only')
    if preparer is None or not hasattr(preparer.vae,'vae'):raise ValueError('Frozen generation VAE preparer required')
    suite=json.loads(Path(suite_path).read_text());output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    was_training=model.training
    device_module().synchronize();dist.barrier();begin=time.perf_counter()
    tokens=token_contract(model.cfg.text_assets,model.cfg)
    tokenizer=Tokenizer.from_file(str(Path(model.cfg.text_assets)/'tokenizer.json'))
    processor=NativeImageProcessor(model.cfg,2048)
    model.eval();local=[]
    try:
        with torch.random.fork_rng(devices=[device_module().current_device()],device_type=accel_type()):
            for index,case in enumerate(suite['cases']):
                if index%P.dp_size()!=P.dp_rank():continue
                device_module().reset_peak_memory_stats();started=time.perf_counter()
                result=(generate(model,case,tokenizer,tokens,processor,preparer.vae if model.cfg.renderer=='wan' else preparer.vae.vae,output)
                    if case['task'] in ('T2I','T2V','EDIT') else caption(model,case,tokenizer,tokens,processor))
                if case['task']=='V2T' and Path(result['image']).suffix=='.npy':
                    path=output/f"{case['id']}.mp4"
                    if P.state().get_tensor_model_parallel_rank()==0:
                        save_visual(np.load(result['image'],allow_pickle=False),path,case['video']['fps'])
                    result['image']=str(path)
                if case['task']=='I2T' and case.get('score_reference',True):
                    result.update(score_reference(model,case,tokenizer,tokens,processor,preparer.vae))
                device_module().synchronize()
                result.update(id=case['id'],task=case['task'],seconds=time.perf_counter()-started,
                    peak_memory_gib=device_module().max_memory_allocated()/2**30,dp_rank=P.dp_rank(),
                    source=case.get('source',{}))
                local.append(result)
        group=P.state().get_data_parallel_group();rows=[None]*P.dp_size()
        dist.all_gather_object(rows,local,group=group)
        dist.barrier()
        event=dict(seconds=time.perf_counter()-begin,examples=sum(map(len,rows)),path=str(output),
                   suite_sha256=hashlib.sha256(Path(suite_path).read_bytes()).hexdigest(),tp=P.tp_size())
        if weight_sources:event['weight_sources']=weight_sources
        if dist.get_rank()==0:
            flat=[r for batch in rows for r in batch]
            (output/'results.json').write_text(json.dumps(dict(**event,suite=suite,results=flat),indent=2,ensure_ascii=False))
            report_html(output,flat,weight_sources)
        return event
    finally:
        model.train(was_training)

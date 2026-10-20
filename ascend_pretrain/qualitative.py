"""Small deterministic CQ validation, sharded by DP with TP groups intact.

No reference captions enter generation. This is a qualitative regression suite,
not an official GenEval/COCO score, nor a claim of decontaminated held-out data.
MOSS schedule/latent decode adapted from OpenMOSS-Team/MOSS-Image-Base
1ceede80badddd6a7812069619edaa5633587192 (Apache-2.0).
"""
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from PIL import Image

from . import parallel as P
from .mainline import backbone,finalize_plan
from .input_pipeline.packing import Packer,token_contract
from .input_pipeline import chat


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
    plan=replace(plan,query_positions=plan.query_positions[keep],query_image=plan.query_image[:,keep],
        query_rope=plan.query_rope[:,keep],output_positions=plan.output_positions[keep],
        cq_mask=plan.cq_mask[:,:,rows].contiguous(),cq_valid=plan.cq_valid[:,rows],cq_plan=None)
    batch.mainline=finalize_plan(batch,cfg,plan)
    return batch


class OneDocument:
    images_preprocessed=True
    def __init__(self,elements,processed=None):
        self.doc=dict(id='validation',elements=elements);self.processed=processed
    def peek(self):return self.doc
    def advance(self):self.doc=None
    def image(self,doc,element):return self.processed
    def prefetch_images(self,*args):pass
    def state_dict(self):return {}


def encode_spans(tokenizer,pieces):
    """Tokenize scaffolding exactly as the training pipeline does."""
    return [np.asarray(tokenizer.encode(piece,add_special_tokens=False).ids,dtype=np.int32)
            for piece in pieces]


def make_batch(cfg,tokens,text_ids,processed,task,prune=True,tokenizer=None,instruction=None):
    """Build one validation pack under the Qwen3.5 chat template.

    Training wraps every caption in user/assistant messages, so evaluation must
    use the same scaffolding or the model is probed off-distribution. The
    instruction is fixed rather than hashed: validation must stay comparable
    across steps.
    """
    body=np.asarray(text_ids,dtype=np.int32)
    def element(ids,supervised=False):
        return dict(type='text',input_ids=ids,supervised=supervised)
    picture=dict(type='image')
    if processed is None:
        elements=[element(body,True)]
    elif tokenizer is None:
        # Legacy untemplated layout, retained only for direct unit tests.
        elements=[element(body,True),picture] if task=='T2I' else [picture,element(body,True)]
    elif task=='T2I':
        head,tail=encode_spans(tokenizer,[
            chat.ROLE_USER+(instruction or chat.T2I_INSTRUCTIONS[0]),
            chat.TURN_END+chat.IM_START+chat.ROLE_ASSISTANT])
        elements=[element(head),element(body,True),element(tail),picture]
    else:
        head,tail=encode_spans(tokenizer,[
            chat.ROLE_USER,
            (instruction or chat.I2T_INSTRUCTIONS[0])+chat.TURN_END+
            chat.IM_START+chat.ROLE_ASSISTANT])
        elements=[element(head),picture,element(tail),element(body,True)]
    elements=[e for e in elements if e['type']!='text' or len(e['input_ids'])]
    docs=OneDocument(elements,processed)
    # Fits the fixed validation examples and does not consume training data.
    image_tokens=processed[2][1]*processed[2][2]//4 if processed is not None else 0
    scaffold=sum(len(e['input_ids']) for e in elements if e['type']=='text')
    length=max(scaffold+image_tokens+4,cfg.vision_min_pixels//1024+4)
    packer=Packer(docs,cfg,length,tokens,padding_multiple=0)
    batch,receipt=packer.next()
    if not receipt['documents'][0]['completed']:raise ValueError('Validation example exceeds its pack')
    used=receipt['occupied']
    # Remove the unused physical tail before constructing the inference plan.
    batch.ids=batch.ids[:,:used];batch.segments=batch.segments[:,:used];batch.sigma=batch.sigma[:,:used]
    batch.image_queries=batch.image_queries[:,:used];batch.row_col=batch.row_col[:,:used]
    from .mainline import build_plan
    batch.mainline=build_plan(batch,cfg)
    if not prune:return batch
    if task=='T2I':keep=batch.mainline.query_image[0]
    else:keep=batch.mainline.query_positions==used-1
    return prune_queries(batch,cfg,keep)


@torch.no_grad()
def score_reference(model,case,tokenizer,tokens,processor,vae):
    """Fixed teacher-forced caption NLL and one fixed-noise conditional flow loss."""
    if not case.get('references'):return {}
    cfg=model.cfg
    with Image.open(case['image']) as image:processed=processor(np.asarray(image.convert('RGB')))
    ids=tokenizer.encode(case['references'][0],add_special_tokens=False).ids
    batch=make_batch(cfg,tokens,ids,processed,'I2T',prune=False,tokenizer=tokenizer).to('npu')
    hidden=backbone(model,batch)
    indices=batch.text_indices
    indices=indices[(indices>batch.image_indices.max()+1)&(indices<batch.ids.shape[1]-1)]
    targets=batch.ids.flatten()[indices]
    nll=model.lm_head(hidden[indices],targets,cfg.ce_chunk)
    batch=make_batch(cfg,tokens,ids,processed,'T2I',tokenizer=tokenizer).to('npu')
    empty=torch.zeros(len(batch.vision.content_indices),cfg.width,device='npu',dtype=model.embedding.weight.dtype)
    condition=backbone(model,batch,vision_features=empty)[batch.mainline.condition_indices]
    pixels=torch.from_numpy(processed[0].transpose(2,0,1).copy())[None].to('npu').float().div(127.5).sub(1)
    mean,std=vae.encode(pixels)
    rng=torch.Generator(device='cpu').manual_seed(case.get('seed',1234))
    target=mean.flatten(2).transpose(1,2)[0]+std.flatten(2).transpose(1,2)[0]*torch.randn(sum(batch.generation.lengths),cfg.latent_channels,generator=rng).to('npu')
    batch.latents=target
    batch.noise=torch.randn(target.shape,generator=rng).to('npu')
    batch.times=torch.tensor([.5],device='npu')
    phase=500.*10000**(-torch.arange(128,dtype=torch.float64)/128)
    batch.generation.time_features=torch.cat((phase.cos(),phase.sin()))[None].to(device='npu',dtype=torch.bfloat16)
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
    path=Path(case['image'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=case['image_sha256']:raise ValueError('Validation image changed')
    with Image.open(path) as image:processed=processor(np.asarray(image.convert('RGB')))
    emitted=[]
    # A suite prompt is a user instruction, not the start of the answer. Feeding
    # it as an assistant prefix made the model read its own instruction back and
    # immediately emit <|im_end|>, yielding an empty caption.
    instruction=case.get('prompt','').strip() or None
    prefix=[]
    features=None;reason='max_new_tokens'
    stops={tokens['eos'],tokens['boi']}
    end=tokenizer.token_to_id('<|im_end|>')
    if end is not None:stops.add(end)
    for _ in range(case.get('max_new_tokens',48)):
        batch=make_batch(cfg,tokens,prefix+emitted,processed,'I2T',tokenizer=tokenizer,
                         instruction=instruction).to('npu')
        if features is None:features=model.patch_embed(batch.vision)
        hidden=backbone(model,batch,vision_features=features)
        token=greedy_token(model,hidden[batch.mainline.output_positions])
        if token in stops:
            reason=f'stop_token_{token}';break
        emitted.append(token)
    return dict(text=tokenizer.decode(emitted,skip_special_tokens=False),token_ids=emitted,
                stop_reason=reason,reference_captions=case.get('references',[]),
                image=case['image'],prompt=case.get('prompt',''))


@torch.no_grad()
def generate(model,case,tokenizer,tokens,processor,vae,output):
    cfg=model.cfg;height,width=case['height'],case['width']
    if min(height,width)<16 or height%16 or width%16 or height*width>cfg.generation_max_pixels:
        raise ValueError('Validation shape violates the training pixel/alignment cap')
    processed=processor(np.zeros((height,width,3),np.uint8))
    def conditions(prompt):
        batch=make_batch(cfg,tokens,tokenizer.encode(prompt,add_special_tokens=False).ids,processed,'T2I',tokenizer=tokenizer).to('npu')
        # Clean target pixels are strictly invisible to image CQ. Zero features
        # eliminate a useless ViT forward, tested against the full CQ plan.
        empty=torch.zeros(len(batch.vision.content_indices),cfg.width,device='npu',dtype=model.embedding.weight.dtype)
        hidden=backbone(model,batch,vision_features=empty)
        return hidden[batch.mainline.condition_indices],batch.generation
    positive,layout=conditions(case['prompt'])
    projected=model.renderer.project_conditions(positive,layout)
    guidance=case.get('guidance',3.)
    negative_projected=None
    if guidance!=1.:
        negative,other=conditions('')
        negative_projected=model.renderer.project_conditions(negative,other)
    generator=torch.Generator(device='cpu').manual_seed(case['seed'])
    latent=torch.randn(sum(layout.lengths),cfg.latent_channels,generator=generator).to('npu')
    sigmas=moss_sigmas(case.get('steps',30),case.get('shift',5.))
    for start,end in zip(sigmas[:-1],sigmas[1:]):
        phase=(start*1000).double()*10000**(-torch.arange(128,dtype=torch.float64)/128)
        time_features=torch.cat((phase.cos(),phase.sin()))[None].to(device='npu',dtype=torch.bfloat16)
        velocity=model.renderer.predict(latent,positive,time_features,layout,projected).float()
        if negative_projected is not None:
            uncond=model.renderer.predict(latent,positive,time_features,layout,negative_projected).float()
            velocity=uncond+guidance*(velocity-uncond)
        latent=latent+(float(end)-float(start))*velocity
    if not torch.isfinite(latent).all().item():raise FloatingPointError('Nonfinite generated latent')
    h,w=layout.grids[0]
    z=latent.T.reshape(1,cfg.latent_channels,h,w).float()
    bn=vae.bn;eps=vae.config.batch_norm_eps
    z=z*(bn.running_var+eps).sqrt().view(1,-1,1,1)+bn.running_mean.view(1,-1,1,1)
    z=z.reshape(1,cfg.latent_channels//4,2,2,h,w).permute(0,1,4,2,5,3).reshape(1,cfg.latent_channels//4,2*h,2*w)
    vae.decoder.to('npu');vae.post_quant_conv.to('npu')
    try:
        pixels=vae.decode(z.to(vae.dtype),return_dict=False)[0]
        if not torch.isfinite(pixels).all().item():raise FloatingPointError('Nonfinite decoded pixels')
        pixels=(pixels.clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8)[0].permute(1,2,0).cpu().numpy())
    finally:
        torch.npu.synchronize()
        vae.decoder.to('cpu');vae.post_quant_conv.to('cpu')
    if P.state().get_tensor_model_parallel_rank()==0:Image.fromarray(pixels).save(output/f"{case['id']}.png")
    return dict(image=f"{case['id']}.png",prompt=case['prompt'],height=height,width=width,
                seed=case['seed'],steps=len(sigmas)-1,guidance=guidance,shift=case.get('shift',5.),
                latent_std=latent.std().item(),latent_mean=latent.mean().item())


def report_html(output,rows,weight_sources=None):
    parts=['<!doctype html><meta charset="utf-8"><title>CQ validation</title>',
           '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}section{display:inline-block;width:48%;vertical-align:top;padding:1%}img{max-width:100%;max-height:400px}</style>',
           '<h1>Fixed CQ validation</h1><p>Qualitative examples; not benchmark scores.</p>']
    if weight_sources:parts.append('<p>Weights: '+html.escape(json.dumps(weight_sources))+'</p>')
    for row in rows:
        parts.append('<section><h2>'+html.escape(row['id'])+'</h2>')
        source=row['image']
        if row['task']=='I2T':
            import shutil
            name=row['id']+Path(source).suffix
            shutil.copyfile(source,output/name);source=name
        parts.append('<img src="'+html.escape(source,quote=True)+'"><p>'+html.escape(row.get('prompt',''))+'</p>')
        if 'text' in row:parts.append('<p>'+html.escape(row['text'])+'</p>')
        if row.get('reference_captions'):parts.append('<details><summary>References (not model input)</summary>'+html.escape(' | '.join(row['reference_captions']))+'</details>')
        parts.append('</section>')
    (output/'index.html').write_text(''.join(parts))


@torch.no_grad()
def run_suite(model,suite_path,output,preparer,weight_sources=None):
    from tokenizers import Tokenizer
    from .vision import NativeImageProcessor
    if model.cfg.method!='CQ_IT':raise ValueError('This validator implements CQ_IT only')
    if preparer is None or not hasattr(preparer.vae,'vae'):raise ValueError('MOSS VAE preparer required')
    suite=json.loads(Path(suite_path).read_text());output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    was_training=model.training
    torch.npu.synchronize();dist.barrier();begin=time.perf_counter()
    tokens=token_contract(model.cfg.text_assets,model.cfg)
    tokenizer=Tokenizer.from_file(str(Path(model.cfg.text_assets)/'tokenizer.json'))
    processor=NativeImageProcessor(model.cfg,2048)
    model.eval();local=[]
    try:
        with torch.random.fork_rng(devices=[torch.npu.current_device()],device_type='npu'):
            for index,case in enumerate(suite['cases']):
                if index%P.dp_size()!=P.dp_rank():continue
                torch.npu.reset_peak_memory_stats();started=time.perf_counter()
                result=(generate(model,case,tokenizer,tokens,processor,preparer.vae.vae,output)
                    if case['task']=='T2I' else caption(model,case,tokenizer,tokens,processor))
                if case['task']=='I2T' and case.get('score_reference',True):
                    result.update(score_reference(model,case,tokenizer,tokens,processor,preparer.vae))
                torch.npu.synchronize()
                result.update(id=case['id'],task=case['task'],seconds=time.perf_counter()-started,
                    peak_memory_gib=torch.npu.max_memory_allocated()/2**30,dp_rank=P.dp_rank(),
                    source=case['source'])
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

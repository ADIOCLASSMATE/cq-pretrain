"""Pinned CF224/384 + UniFD inference. Scores are uncalibrated p(fake), not policy decisions."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

# CPU machines may have torch_npu installed without an Ascend runtime.
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from torch import nn
import timm
import torchvision
from torchvision import transforms
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

ADAPTER_VERSION = '2026-10-22.v1'
EXPECTED_FC = '477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847'
EXPECTED = {
 'cf224': 'a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0',
 'cf384': 'b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387',
 'clip_vitl14': 'a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb',
}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8 << 20),b''):h.update(b)
 return h.hexdigest()

def atomic_json(path, obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

class FP32LayerNorm(nn.LayerNorm):
 def forward(self,x):
  # Matches OpenAI LayerNorm, including half precision execution.
  return torch.nn.functional.layer_norm(x.float(),self.normalized_shape,self.weight.float(),self.bias.float(),self.eps).to(x.dtype)

class QuickGELU(nn.Module):
 def forward(self,x):return x*torch.sigmoid(1.702*x)

class ResidualAttentionBlock(nn.Module):
 def __init__(self):
  super().__init__();self.attn=nn.MultiheadAttention(1024,16);self.ln_1=FP32LayerNorm(1024)
  self.mlp=nn.Sequential(OrderedDict([('c_fc',nn.Linear(1024,4096)),('gelu',QuickGELU()),('c_proj',nn.Linear(4096,1024))]));self.ln_2=FP32LayerNorm(1024)
 def forward(self,x):
  y=self.ln_1(x);x=x+self.attn(y,y,y,need_weights=False)[0]
  return x+self.mlp(self.ln_2(x))

class CLIPVision(nn.Module):
 """OpenAI ViT-L/14 visual tower; no L2 normalization after projection."""
 def __init__(self):
  super().__init__();self.conv1=nn.Conv2d(3,1024,14,14,bias=False)
  self.class_embedding=nn.Parameter(torch.empty(1024));self.positional_embedding=nn.Parameter(torch.empty(257,1024));self.ln_pre=FP32LayerNorm(1024)
  self.transformer=nn.Module();self.transformer.resblocks=nn.Sequential(*[ResidualAttentionBlock() for _ in range(24)])
  self.ln_post=FP32LayerNorm(1024);self.proj=nn.Parameter(torch.empty(1024,768))
 def forward(self,x):
  x=self.conv1(x).flatten(2).permute(0,2,1)
  cls=self.class_embedding.to(x.dtype)+torch.zeros(x.shape[0],1,x.shape[-1],device=x.device,dtype=x.dtype)
  x=self.ln_pre(torch.cat([cls,x],dim=1)+self.positional_embedding.to(x.dtype))
  x=self.transformer.resblocks(x.permute(1,0,2)).permute(1,0,2)
  return self.ln_post(x[:,0,:]) @ self.proj

class UniFD(nn.Module):
 def __init__(self):
  super().__init__();self.visual=CLIPVision();self.fc=nn.Linear(768,1)
 def forward(self,x):return self.fc(self.visual(x))

def hf_clip_visual_state(path):
 """Map ALL HF visual parameters to OpenAI names; reject unexpected visual keys."""
 mapped={};used=set()
 with safe_open(path,framework='pt',device='cpu') as f:
  all_keys=set(f.keys());visual_keys={k for k in all_keys if k.startswith('vision_model.') or k=='visual_projection.weight'}
  def take(key):used.add(key);return f.get_tensor(key)
  for dest,src in [('conv1.weight','embeddings.patch_embedding.weight'),('class_embedding','embeddings.class_embedding'),('positional_embedding','embeddings.position_embedding.weight'),('ln_pre.weight','pre_layrnorm.weight'),('ln_pre.bias','pre_layrnorm.bias'),('ln_post.weight','post_layernorm.weight'),('ln_post.bias','post_layernorm.bias')]:
   mapped[dest]=take('vision_model.'+src)
  mapped['proj']=take('visual_projection.weight').T.contiguous()
  for layer in range(24):
   src=f'vision_model.encoder.layers.{layer}.';dst=f'transformer.resblocks.{layer}.'
   for suffix in ['weight','bias']:
    mapped[dst+'attn.in_proj_'+suffix]=torch.cat([take(src+f'self_attn.{q}_proj.{suffix}') for q in ['q','k','v']],dim=0)
    for dest,key in [('attn.out_proj','self_attn.out_proj'),('ln_1','layer_norm1'),('ln_2','layer_norm2'),('mlp.c_fc','mlp.fc1'),('mlp.c_proj','mlp.fc2')]:mapped[dst+dest+'.'+suffix]=take(src+key+'.'+suffix)
  # Historical HF state_dicts may include a non-trainable position_ids buffer.
  position_buffer='vision_model.embeddings.position_ids'
  if position_buffer in visual_keys:
   assert torch.equal(take(position_buffer),torch.arange(257).unsqueeze(0))
  if visual_keys!=used:raise ValueError(f'Unmapped or missing vision parameters: {visual_keys ^ used}')
  excluded=all_keys-visual_keys
  if any(not(k.startswith('text_model.') or k in ['text_projection.weight','logit_scale']) for k in excluded):raise ValueError('Unexpected nonvisual parameter')
 return mapped,{'source_visual_keys':len(visual_keys),'mapped_visual_keys':len(mapped),'excluded_text_keys':len(excluded)}

def preprocessing(name):
 if name.startswith('cf'):
  size=int(name[2:]);resize=256 if size==224 else 440
  spec={'rgb':True,'resize_short_edge':resize,'resize_interpolation':'bilinear','antialias':True,'center_crop':size,'mean':[.485,.456,.406],'std':[.229,.224,.225],'exif_transpose':False}
  t=transforms.Compose([transforms.Resize(resize,interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),transforms.CenterCrop(size),transforms.ToTensor(),transforms.Normalize(spec['mean'],spec['std'])])
 else:
  spec={'rgb':True,'resize_short_edge':None,'center_crop':224,'small_image_padding':'torchvision CenterCrop zero padding','mean':[.48145466,.4578275,.40821073],'std':[.26862954,.26130258,.27577711],'exif_transpose':False,'feature_l2_normalization':False}
  t=transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(spec['mean'],spec['std'])])
 return t,spec

def load_model(name,root):
 root=Path(root);base=name if name.startswith('cf') else 'clip_vitl14';weights=root/base/'model.safetensors'
 digest=sha256(weights)
 if digest!=EXPECTED[base]:raise ValueError(f'Checkpoint digest mismatch: {weights}')
 meta={'adapter_version':ADAPTER_VERSION,'checkpoint':str(weights),'checkpoint_sha256':digest,'label_real':0,'label_fake':1,'score':'sigmoid(raw_logit); uncalibrated','dtype':'float32','state_load_strict':True}
 if name.startswith('cf'):
  size=int(name[2:]);model=timm.create_model(f'vit_small_patch16_{size}.augreg_in21k_ft_in1k',pretrained=False,num_classes=1)
  state=load_file(str(weights),device='cpu')
  if not all(k.startswith('vit.') for k in state):raise ValueError('Unexpected CF key')
  model.load_state_dict({k[4:]:v for k,v in state.items()},strict=True);del state
  meta.update(architecture=f'timm.vit_small_patch16_{size}.augreg_in21k_ft_in1k',source_revision='ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4')
 else:
  cfg=json.loads((root/'clip_vitl14/config.json').read_text())['vision_config']
  for k,v in {'hidden_size':1024,'num_hidden_layers':24,'num_attention_heads':16,'patch_size':14,'image_size':224,'hidden_act':'quick_gelu','projection_dim':768,'layer_norm_eps':1e-5}.items():
   if cfg[k]!=v:raise ValueError('Unexpected CLIP config '+k)
  model=UniFD();state,mapping=hf_clip_visual_state(weights);model.visual.load_state_dict(state,strict=True);del state
  fc=root/'official_source/unifd/pretrained_weights/fc_weights.pth'
  if sha256(fc)!=EXPECTED_FC:raise ValueError('UniFD detector head digest mismatch')
  state=torch.load(fc,map_location='cpu',weights_only=True)
  if set(state)!={'weight','bias'}:raise ValueError('Unexpected UniFD detector head')
  model.fc.load_state_dict(state,strict=True);del state
  meta.update(architecture='OpenAI CLIP ViT-L/14 vision (HF converted author weights) + official UniFD linear head',source_revision='030495aea3300a8b54c0ec37ec7fe1dd7e63c619',fc_sha256=sha256(fc),mapping=mapping)
 transform,spec=preprocessing(name);meta['preprocessing']=spec;meta['parameters']=sum(p.numel() for p in model.parameters())
 model.eval();model.requires_grad_(False)
 return model,transform,meta

def sync(device):
 if device.startswith('cuda'):torch.cuda.synchronize()
 elif device.startswith('npu'):torch.npu.synchronize()

def read_rows(path):
 if not Path(path).exists():return []
 rows=[]
 for line in Path(path).read_text().splitlines():
  if line.strip():rows.append(json.loads(line))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--models-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--models',nargs='+',choices=['cf224','cf384','unifd'],default=['cf224','cf384','unifd']);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=8);p.add_argument('--limit',type=int);p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32');p.add_argument('--retry-errors',action='store_true')
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 if args.device.startswith('npu'):import torch_npu
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 if args.device.startswith('cuda'):torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(0)
 dtype=getattr(torch,args.dtype);samples=read_rows(args.samples)
 if args.limit:samples=samples[:args.limit]
 ids=[s['image_sha256'] for s in samples]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate image_sha256 in samples')
 sample_hash=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest()
 run_info={'adapter_version':ADAPTER_VERSION,'code_sha256':sha256(__file__),'samples_file_sha256':sha256(args.samples),'selected_samples_sha256':sample_hash,'selected_count':len(samples),'device':args.device,'dtype':args.dtype,'batch_size':args.batch_size,'threads':args.threads,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__,'models':args.models}
 run_path=args.output/'inference_run.json'
 if run_path.exists() and json.loads(run_path.read_text())!=run_info:raise ValueError('Run signature changed; use a new output directory')
 atomic_json(run_path,run_info)
 for name in args.models:
  output=args.output/f'scores.{name}.jsonl';previous={r['image_sha256']:r for r in read_rows(output)}
  todo=[s for s in samples if s['image_sha256'] not in previous or (args.retry_errors and previous[s['image_sha256']]['status']!='ok')]
  if not todo:continue
  load_start=time.perf_counter();model,transform,meta=load_model(name,args.models_dir);model.to(device=args.device,dtype=dtype);sync(args.device)
  meta.update(device=args.device,dtype=args.dtype,load_s=time.perf_counter()-load_start)
  atomic_json(args.output/f'modelinfo.{name}.json',meta)
  print(json.dumps({'model':name,'todo':len(todo),'load_s':meta['load_s']}),flush=True)
  start=time.perf_counter();completed=0;failed=0;total_forward=0.;total_decode=0.
  with output.open('a',buffering=1) as out,torch.inference_mode():
   def emit(s,**extra):
    row={'image_sha256':s['image_sha256'],'image_path':s['image_path'],'model':name,**extra};out.write(json.dumps(row,ensure_ascii=False)+'\n')
   for offset in range(0,len(todo),args.batch_size):
    tensors=[];valid=[];decode_start=time.perf_counter()
    for s in todo[offset:offset+args.batch_size]:
     try:
      raw=Path(s['image_path']).read_bytes()
      if hashlib.sha256(raw).hexdigest()!=s['image_sha256']:raise ValueError('image bytes do not match sample SHA-256')
      with Image.open(io.BytesIO(raw)) as im:tensors.append(transform(im.convert('RGB')))
      valid.append(s)
     except Exception as e:emit(s,status='error',error=f'{type(e).__name__}: {str(e)[:400]}',stage='decode');failed+=1
    decode_s=time.perf_counter()-decode_start;total_decode+=decode_s
    if valid:
     batch=torch.stack(tensors).to(device=args.device,dtype=dtype);sync(args.device);t=time.perf_counter()
     try:
      logits=model(batch).float().flatten();sync(args.device);elapsed=time.perf_counter()-t;total_forward+=elapsed
      values=logits.cpu().tolist();probs=logits.sigmoid().cpu().tolist()
      if not all(math.isfinite(value) for value in values):raise ValueError('Nonfinite output')
      for s,logit,prob in zip(valid,values,probs):
       emit(s,status='ok',logit=logit,prob=prob,timing_s={'batch_forward':elapsed,'batch_size':len(valid),'batch_decode':decode_s});completed+=1
     except Exception as e:
      for s in valid:emit(s,status='error',error=f'{type(e).__name__}: {str(e)[:400]}',stage='forward');failed+=1
     del batch,tensors
    out.flush();os.fsync(out.fileno())
    if offset%(args.batch_size*8)==0:print(json.dumps({'model':name,'completed':completed,'failed':failed,'elapsed_s':time.perf_counter()-start}),flush=True)
  timing={'model':name,'new_completed':completed,'new_failed':failed,'elapsed_s':time.perf_counter()-start,'forward_s':total_forward,'decode_s':total_decode}
  atomic_json(args.output/f'timing.{name}.json',timing);print(json.dumps(timing),flush=True)
  del model;gc.collect()
  if args.device.startswith('cuda'):torch.cuda.empty_cache()
  elif args.device.startswith('npu'):torch.npu.empty_cache()
 by_model={name:{r['image_sha256']:r for r in read_rows(args.output/f'scores.{name}.jsonl')} for name in args.models}
 merged=args.output/'scores.jsonl';temp=merged.with_suffix('.tmp')
 with temp.open('w') as f:
  for s in samples:
   scores={name:by_model[name].get(s['image_sha256'],{'status':'missing'}) for name in args.models}
   row={'image_sha256':s['image_sha256'],'image_path':s['image_path'],'status':'ok' if all(v['status']=='ok' for v in scores.values()) else 'incomplete','scores':{name:{k:v for k,v in r.items() if k not in ['image_sha256','image_path','model']} for name,r in scores.items()}}
   f.write(json.dumps(row,ensure_ascii=False)+'\n')
 temp.replace(merged)
 atomic_json(args.output/'COMPLETE.json',{'success':all(all(s['image_sha256'] in by_model[n] and by_model[n][s['image_sha256']]['status']=='ok' for n in args.models) for s in samples),'records':len(samples),'models':args.models,'scores_sha256':sha256(merged)})

if __name__=='__main__':main()

"""CPU oracle / 16-rank TP2 validation of the benchmark's native Gemma path."""
import argparse
import json
from pathlib import Path
import torch
from unittest.mock import patch
import infra.parallel as P
from configs.config import RunConfig
from scripts.benchmark_gemma import GemmaDecoder, SoftcapCE, attention_tiles


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--npu', action='store_true'); args = p.parse_args()
    if args.npu: P.initialize(2, RunConfig(), initialize_weights=False)
    with patch('transformers.utils.import_utils.is_torch_npu_available', return_value=args.npu), \
         patch('transformers.utils.is_torch_npu_available', return_value=args.npu):
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    cfg = Gemma4TextConfig(vocab_size=256, vocab_size_per_layer_input=256,
        hidden_size=128, intermediate_size=256, num_hidden_layers=5,
        num_attention_heads=8, num_key_value_heads=1, head_dim=256, global_head_dim=512,
        hidden_size_per_layer_input=16, num_kv_shared_layers=2, use_double_wide_mlp=True,
        layer_types=['sliding_attention','full_attention','sliding_attention','sliding_attention','full_attention'],
        sliding_window=8, rope_parameters={'sliding_attention':{'rope_type':'default','rope_theta':10000.},
          'full_attention':{'rope_type':'proportional','rope_theta':1000000.,'partial_rotary_factor':.25}})
    torch.manual_seed(414)
    device = 'npu' if args.npu else 'cpu'; model = GemmaDecoder(cfg)
    model.config = P._config
    if args.npu:
        source = torch.load(args.output.with_suffix('.pt'), map_location='cpu', weights_only=False)
        with torch.no_grad():
            for name, value in model.checkpoint_targets().items():
                value.copy_(P.checkpoint_shard(source['weights'][name], value))
        model.to('npu', torch.bfloat16)
        wrapped, optimizer = P.make_optimizer(model, 1e-5, overlap_param_gather=False)
        wrapped.zero_grad_buffer()
        weight_error = max(float((v.float().cpu()-P.checkpoint_shard(source['weights'][k],v)).abs().max())
                           for k,v in model.checkpoint_targets().items())
        assert weight_error == 0, weight_error
    else:
        # Native E2B learns small K-normalization scales. Unit K scales at
        # D=512 saturate random softmax and make BF16/FP32 comparison ill-conditioned.
        from safetensors import safe_open
        asset = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Gemma4-E2B-Base/model.safetensors')
        with safe_open(str(asset), framework='pt') as native_weights, torch.no_grad():
            for layer in model.layers:
                native = 4 if layer.self_attn.kind == 'full_attention' else 0
                for norm in ('q_norm', 'k_norm'):
                    if hasattr(layer.self_attn, norm):
                        value = native_weights.get_tensor(f'model.language_model.layers.{native}.self_attn.{norm}.weight')
                        getattr(layer.self_attn, norm).weight.copy_(value)
        model.bfloat16().float()
        source = {'weights': {k:v.detach().clone() for k,v in model.checkpoint_targets().items()}}
        wrapped = model
    activations = {}
    model.per_layer_model_projection.register_forward_pre_hook(lambda module,inputs:
        activations.__setitem__('projection_input',inputs[0].detach().float().cpu().clone()))
    watched = {'embed_tokens', 'embed_tokens_per_layer', 'per_layer_model_projection',
               'layers.0.input_layernorm', 'layers.0.self_attn', 'layers.0', 'layers.1', 'layers.2', 'layers.3', 'layers.4'}
    for name, module in model.named_modules():
        if name in watched:
            module.register_forward_hook(lambda module, inputs, output, key=name:
                activations.__setitem__(key,(output[0] if isinstance(output,tuple) else output).detach().float().cpu()))
    s = 16; q = 4; n = s+q
    ids = torch.arange(n)[None] + 1; ids[:, s:] = 5; ids[:, 4] = 0
    positions = torch.cat((torch.arange(s), torch.tensor([3, 8, 12, 15])))[None]
    base = torch.ones(1,1,n,s,dtype=torch.bool)
    base[:, :, :s] = torch.arange(s)[None,:] > torch.arange(s)[:,None]
    base[:, :, 4:8,4:8] = False
    base[:, :, s:] = torch.arange(s)[None,:] >= positions[0,s:,None]
    masks = {'full_attention':base,'sliding_attention':base | (torch.arange(s)<positions[:,:,None]-7)[:,None]}
    rotary = Gemma4TextRotaryEmbedding(cfg)
    dtype = torch.bfloat16 if args.npu else torch.float32
    ropes = {kind:tuple(t.transpose(0,1).unsqueeze(2).to(device,dtype) for t in
             rotary(torch.empty(1,n,128),positions,layer_type=kind)) for kind in masks}
    tiles = {kind:attention_tiles(mask,4).bind(mask.to(device)) for kind,mask in masks.items()}
    upstream = torch.linspace(-.1,.1,q*128).reshape(q,1,128).to(device,dtype)
    y = wrapped(ids.to(device),torch.empty(0,dtype=torch.long,device=device),
        torch.empty(0,128,device=device,dtype=dtype),s,ropes,tiles,torch.ones(1,n,dtype=torch.bool,device=device))
    ce = SoftcapCE.apply(y[:, 0], model.embed_tokens.weight,
        torch.tensor([11,12,13,14], device=device), 2, 30.)
    ((y*upstream).sum() + ce*.1).backward()
    if args.npu:
        from scripts.benchmark_backbones import finish_gradients
        finish_gradients(wrapped, cfg)
        # Distributed Adam retains only each DP rank's reduced gradient shard.
        # Reconstruct those owned shards for comparison; the other buffer regions
        # still contain locally scaled contributions, not complete gradients.
        import torch.distributed as dist
        for buffer in wrapped.buffers:
            for bucket in buffer.buckets:
                full = torch.empty_like(bucket.grad_data)
                local = bucket.grad_data.chunk(P.dp_size())[P.dp_rank()].contiguous()
                dist.all_gather_into_tensor(full, local, group=P.state().get_data_parallel_group())
                bucket.grad_data.copy_(full)
    def native_grads():
        result = {}
        for name, param in model.named_parameters():
            grad = param.main_grad if args.npu else param.grad
            name = name.replace('.impl.', '.')
            if '.gate_up_proj.' in name:
                for key, value in zip(('gate_proj','up_proj'),grad.chunk(2,0)):
                    result[name.replace('gate_up_proj',key)] = value
            else:result[name] = grad
        return result
    if not args.npu:
        source.update(y=y.detach(), activations=activations, gradients={k:v.detach().clone() for k,v in native_grads().items()})
        torch.save(source,args.output.with_suffix('.pt')); print('CPU oracle written'); return
    def rms(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8))
    errors = {name:rms(value.cpu(),P.checkpoint_shard(source['gradients'][name],model.checkpoint_targets()[name]))
              for name,value in native_grads().items()}
    import torch.distributed as dist
    values = torch.tensor([rms(y.cpu(),source['y']),max(errors.values())],device='npu')
    dist.all_reduce(values,op=dist.ReduceOp.MAX)
    report = dict(world_size=dist.get_world_size(),tp=2,dp=P.dp_size(),output_relative_rms=float(values[0]),
                  max_parameter_gradient_relative_rms=float(values[1]),rank0_gradient_errors=errors,
                  activation_errors={k:rms(v,source['activations'][k]) for k,v in activations.items()},
                  passed=bool(values[0]<.03 and values[1]<.08))
    if dist.get_rank()==0:args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    assert report['passed'], report
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

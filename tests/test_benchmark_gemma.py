"""Check the benchmark adapter against the unmodified Transformers decoder."""
import torch
from unittest.mock import patch
import transformers.utils.import_utils
with patch('transformers.utils.import_utils.is_torch_npu_available', return_value=False), \
     patch('transformers.utils.is_torch_npu_available', return_value=False):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
from scripts.benchmark_gemma import GemmaDecoder, SoftcapCE, attention_tiles


def test_native_gemma_cq_forward_backward():
    torch.manual_seed(19)
    cfg = Gemma4TextConfig(vocab_size=128, vocab_size_per_layer_input=128,
        hidden_size=64, intermediate_size=128, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16, global_head_dim=32,
        hidden_size_per_layer_input=8, num_kv_shared_layers=2, use_double_wide_mlp=True,
        layer_types=['sliding_attention', 'full_attention', 'sliding_attention',
                     'sliding_attention', 'full_attention'], sliding_window=8,
        rope_parameters={'sliding_attention': {'rope_type':'default', 'rope_theta':10000.},
            'full_attention':{'rope_type':'proportional', 'rope_theta':1000000., 'partial_rotary_factor':.25}})
    cfg._attn_implementation = 'eager'
    reference = Gemma4TextModel(cfg).float()
    model = GemmaDecoder(cfg).float()
    targets = model.checkpoint_targets()
    with torch.no_grad():
        state = reference.state_dict()
        assert set(targets) == set(state)
        for name, tensor in targets.items(): tensor.copy_(state[name])
    s = 14; q = 4; n = s+q
    ids = torch.randint(1, 128, (1, n)); ids[:, -q:] = 5
    ids[:, 4] = 0
    positions = torch.cat((torch.arange(s), torch.tensor([3, 6, 10, 13])))[None]
    base = torch.ones(1, 1, n, s, dtype=torch.bool)
    base[:, :, :s] = torch.arange(s)[None, :] > torch.arange(s)[:, None]
    base[:, :, 4:7, 4:7] = False  # A bidirectional visual block.
    base[:, :, s:] = torch.arange(s)[None, :] >= positions[0, s:, None]
    masks = {'full_attention': base, 'sliding_attention': base | (torch.arange(s) < positions[:, :, None]-7)[:, None]}
    full_masks = {kind: torch.cat((mask, torch.ones(1, 1, n, q, dtype=torch.bool)), -1).float()*-1e9
                  for kind, mask in masks.items()}
    ropes = {kind: tuple(t.transpose(0, 1).unsqueeze(2) for t in reference.rotary_emb(
        torch.empty(1, n, 64), positions, layer_type=kind)) for kind in masks}
    tiles = {kind: attention_tiles(mask, 4) for kind, mask in masks.items()}
    ref = reference(input_ids=ids, attention_mask=full_masks, position_ids=positions,
                    use_cache=False).last_hidden_state[:, s:].transpose(0, 1)
    out = model(ids, torch.empty(0, dtype=torch.long), torch.empty(0, 64), s, ropes, tiles,
                torch.ones(1, n, dtype=torch.bool), recompute=True)
    torch.testing.assert_close(out, ref, rtol=3e-5, atol=3e-5)
    upstream = torch.randn_like(out)
    (out*upstream).sum().backward(); (ref*upstream).sum().backward()
    for name, expected in reference.named_parameters():
        if 'gate_proj' in name or 'up_proj' in name:
            # Fused projection views do not retain .grad; select its parent.
            layer = model.layers[int(name.split('.')[1])].mlp
            index = 0 if 'gate_proj' in name else 1
            actual = layer.gate_up_proj.weight.grad.chunk(2, 0)[index]
        else:
            clean = name.replace('.q_proj.', '.q_proj.impl.').replace('.o_proj.', '.o_proj.impl.').replace('.down_proj.', '.down_proj.impl.')
            actual = dict(model.named_parameters())[clean].grad
        relative = (actual-expected.grad).norm() / expected.grad.norm().clamp_min(1e-9)
        assert relative < 2e-5, (name, float(relative))


def test_softcap_ce():
    torch.manual_seed(12)
    h = torch.randn(13, 8, requires_grad=True); w = torch.randn(31, 8, requires_grad=True)
    labels = torch.randint(0, 31, (13,))
    actual = SoftcapCE.apply(h, w, labels, 4, 3.)
    grads = torch.autograd.grad(actual, (h, w), retain_graph=True)
    expected = torch.nn.functional.cross_entropy((h@w.T/3.).tanh()*3., labels)
    ref = torch.autograd.grad(expected, (h, w))
    torch.testing.assert_close(actual, expected)
    for a, b in zip(grads, ref): torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

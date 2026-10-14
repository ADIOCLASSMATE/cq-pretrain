"""Real local tensor/tokenizer files exercise strict checkpoint loading."""
import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from models.modeling_umm import UMM
from training.train import load_model, load_tokenizer
from utils.weights import check_source_architecture, convert_state_dict, load_into_model, read_weights
from test_model_core import config as tiny_config


def source_state(model):
    result = {}
    for name, value in model.state_dict().items():
        if name.startswith("backbone."):
            name = "model." + name[len("backbone."):]
        elif name.startswith("generator."):
            name = "image_flow_head.net." + name[len("generator."):]
        elif name.startswith("condition_projector."):
            name = "image_flow_condition_proj." + name[len("condition_projector."):]
        result[name] = value.clone()
    return result


def test_full_source_mapping_and_tied_head_round_trip(tmp_path):
    torch.manual_seed(8)
    expected = UMM(tiny_config())
    path = tmp_path / "source"
    path.mkdir()
    save_file(source_state(expected), str(path / "model.safetensors"))
    values = {**expected.config.to_dict(), "joint_dit_head_type": "z2", "flow_condition_contract": "backbone_xt_fixed"}
    (path / "config.json").write_text(json.dumps(values))
    loaded = load_model({"model": {"checkpoint": str(path)}})
    assert loaded.lm_head.weight is loaded.backbone.embed_tokens.weight
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_missing_duplicate_and_conflicting_weights_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_weights(tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        read_weights(tmp_path)
    model = UMM(tiny_config())
    values = source_state(model)
    with pytest.raises(ValueError, match="collision"):
        convert_state_dict({"model.norm.weight": values["model.norm.weight"], "backbone.norm.weight": values["model.norm.weight"]})
    values["lm_head.weight"] = values["lm_head.weight"] + 1
    with pytest.raises(ValueError, match="disagree"):
        load_into_model(model, values)
    save_file({"key": torch.ones(1)}, str(tmp_path / "part1.safetensors"))
    save_file({"key": torch.ones(1)}, str(tmp_path / "part2.safetensors"))
    with pytest.raises(ValueError, match="Duplicate"):
        read_weights(tmp_path)


def test_explicit_checkpoint_cannot_silently_downgrade_to_base(tmp_path):
    model = UMM(tiny_config())
    values = {key: value for key, value in source_state(model).items()
              if not key.startswith("image_flow_head.")}
    path = tmp_path / "incomplete.pt"
    torch.save({"model": values, "config": model.config.to_dict()}, path)
    with pytest.raises((RuntimeError, ValueError), match="[Mm]issing|[Ii]ncomplete|generator"):
        load_model({"model": {"checkpoint": str(path)}})


@pytest.mark.parametrize("unsupported", [
    {"joint_dit_head_type": "s2"},
    {"joint_dit_dynamic_xt": True},
    {"flow_condition_contract": "backbone_xt_dynamic"},
    {"flow_head_attention_contract": "causal"},
    {"dual_stream_attention_contract": "strict"},
    {"training_image_sigma_order": "random"},
    {"backbone_attention_output_gate": "per_head_identity_sigmoid"},
    {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
    {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000.0}},
])
def test_incompatible_source_math_is_rejected(unsupported):
    with pytest.raises(ValueError):
        check_source_architecture(unsupported)


def test_local_base_tokenizer_growth_preserves_pretrained_rows_and_binding(tmp_path):
    torch.manual_seed(10)
    config = tiny_config()
    original = UMM(config)
    vocab = {"[UNK]": 0, "</s>": 1, "[PAD]": 2, "a": 3, "b": 4}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>", pad_token="[PAD]")
    tokenizer.save_pretrained(tmp_path)
    base = {name: tensor.clone() for name, tensor in source_state(original).items()
            if name.startswith("model.") and not name.startswith("model.image_token_embedder.")}
    base["model.embed_tokens.weight"] = base["model.embed_tokens.weight"][:5].clone()
    # HF safetensors may omit the tied output alias.
    save_file(base, str(tmp_path / "model.safetensors"))
    base_config = {key: value for key, value in config.to_dict().items()
                   if key not in {"mask_token_id", "image_mask_token_id", "boi_token_id", "eoi_token_id"}}
    base_config.update(vocab_size=5, eos_token_id=1, pad_token_id=2, model_type="qwen3")
    (tmp_path / "config.json").write_text(json.dumps(base_config))
    requested = config.to_dict()
    requested.pop("vocab_size")
    requested["model_path"] = str(tmp_path)
    settings = {"model": requested}
    grown_tokenizer = load_tokenizer(settings)
    loaded = load_model(settings)
    assert len(grown_tokenizer) == 9 and loaded.config.vocab_size == 9
    assert len({loaded.config.mask_token_id, loaded.config.boi_token_id,
                loaded.config.eoi_token_id, loaded.config.image_mask_token_id}) == 4
    torch.testing.assert_close(loaded.backbone.embed_tokens.weight[:5], base["model.embed_tokens.weight"], rtol=0, atol=0)
    assert loaded.backbone.embed_tokens.weight[5:].abs().sum() > 0
    assert loaded.lm_head.weight is loaded.backbone.embed_tokens.weight


def test_base_missing_layer_and_wrong_shapes_are_not_accepted():
    model = UMM(tiny_config())
    base = {name: tensor for name, tensor in source_state(model).items()
            if name.startswith("model.") and not name.startswith("model.image_token_embedder.")}
    del base["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(ValueError, match="Incomplete"):
        load_into_model(model, base, backbone_only=True)
    base = source_state(model)
    base["model.embed_tokens.weight"] = torch.randn(35, 16)
    base.pop("lm_head.weight")
    with pytest.raises(ValueError, match="vocabulary"):
        load_into_model(model, base, backbone_only=True)


def test_checkpoint_conflicting_configuration_is_rejected(tmp_path):
    model = UMM(tiny_config())
    path = tmp_path / "weights.pt"
    torch.save({"model": source_state(model), "config": model.config.to_dict()}, path)
    with pytest.raises(ValueError, match="conflicts"):
        load_model({"model": {"checkpoint": str(path), "rope_theta": model.config.rope_theta * 2}})

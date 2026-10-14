"""Start from native HF Qwen weights; initialize only new image query tokens."""
from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from training.train import load_model, load_tokenizer
from utils.checkpoint import export_weights
from test_model_core import config as tiny_config


def base_checkpoint(path, *, vocab_size, tied):
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "<pad>": 2, "a": 3, "b": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>")
    tokenizer.save_pretrained(path)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        tie_word_embeddings=tied, pad_token_id=2, eos_token_id=1))
    base.save_pretrained(path, max_shard_size="5KB")
    values = tiny_config(tie_word_embeddings=tied).to_dict()
    values.pop("vocab_size")
    return base, {"model": {**values, "model_path": str(path)}}


@pytest.mark.parametrize("vocab_size", [5, 32])
@pytest.mark.parametrize("tied", [False, True])
def test_mask_choices_change_only_image_row_and_preserve_qwen_weights(tmp_path, vocab_size, tied):
    base, config = base_checkpoint(tmp_path, vocab_size=vocab_size, tied=tied)
    loaded = {}
    rng_states = []
    for mode in ("copy_text", "random"):
        settings = deepcopy(config)
        settings["model"]["image_mask_init"] = mode
        load_tokenizer(settings)
        torch.manual_seed(41)
        loaded[mode] = load_model(settings)
        rng_states.append(torch.get_rng_state())
        assert "_added_image_mask_token" not in settings["model"]
        assert settings["model"]["image_mask_init"] == mode
    copied, random = loaded["copy_text"], loaded["random"]
    text_id, image_id = copied.config.mask_token_id, copied.config.image_mask_token_id
    embedding = copied.backbone.embed_tokens.weight
    torch.testing.assert_close(embedding[image_id], embedding[text_id], rtol=0, atol=0)
    assert image_id != text_id and embedding[image_id].data_ptr() != embedding[text_id].data_ptr()
    assert not torch.equal(random.backbone.embed_tokens.weight[image_id], random.backbone.embed_tokens.weight[text_id])
    assert torch.equal(*rng_states)
    for key, value in copied.state_dict().items():
        other = random.state_dict()[key]
        if key == "backbone.embed_tokens.weight" or (tied and key == "lm_head.weight"):
            rows = torch.arange(value.shape[0]) != image_id
            torch.testing.assert_close(value[rows], other[rows], rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, other, rtol=0, atol=0)
    for key, expected in base.state_dict().items():
        target = "backbone." + key[len("model."):] if key.startswith("model.") else key
        actual = copied.state_dict()[target]
        if key in {"model.embed_tokens.weight", "lm_head.weight"}:
            actual = actual[:vocab_size]
            rows = torch.arange(vocab_size) != image_id if key.startswith("model.") or tied else torch.ones(vocab_size, dtype=torch.bool)
            actual, expected = actual[rows], expected[rows]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (copied.lm_head.weight is copied.backbone.embed_tokens.weight) == tied


@pytest.mark.parametrize("mode", ["copy_text", "random"])
def test_full_checkpoint_never_reinitializes_learned_mask_rows(tmp_path, mode):
    _, settings = base_checkpoint(tmp_path / "base", vocab_size=32, tied=True)
    load_tokenizer(settings)
    model = load_model(settings)
    with torch.no_grad():
        model.backbone.embed_tokens.weight[model.config.mask_token_id].fill_(2)
        model.backbone.embed_tokens.weight[model.config.image_mask_token_id].fill_(3)
    export = export_weights(tmp_path / "weights.pt", model=model, config=settings)
    request = {"model": {"checkpoint": str(export), "tokenizer_path": settings["model"]["model_path"],
                          "image_mask_init": mode}}
    # Even an old base tokenizer that newly adds the image token must not cause
    # the already-trained image embedding in the full checkpoint to be reset.
    load_tokenizer(request)
    restored = load_model(request)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_invalid_mask_initialization_is_rejected():
    with pytest.raises(ValueError, match="image_mask_init"):
        load_model({"model": {"image_mask_init": "shared"}})

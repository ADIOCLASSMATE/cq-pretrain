"""Small native Qwen/Wan fixtures; never used by production loading."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers import Qwen2VLImageProcessorFast
from configs.config import DEFAULTS, validate_config
from models.modeling_umm import UMM, UMMConfig
from data.cache import checkpoint_id, QWEN_FORMAT, WAN_FORMAT, save_cache
from data.dataset import prepare_raw_video, qwen_video_inputs


def make_fixture(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocab = {f"unused_{i}": i for i in range(64)}
    tokens = {
        0: "<pad>",
        1: "<unk>",
        2: "<eos>",
        3: "user",
        4: "assistant",
        5: "Describe",
        6: "video",
        7: "blue",
        8: "block",
        9: "moves",
        10: "red",
        11: "<|im_start|>",
        12: "<|im_end|>",
        57: "<|image_pad|>",
        58: "<|video_pad|>",
        59: "<|text_mask|>",
        60: "<|vision_mask|>",
        61: "<|vision_start|>",
        62: "<|vision_end|>",
    }
    for i, token in tokens.items():
        del vocab[f"unused_{i}"]
        vocab[token] = i
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        additional_special_tokens=[v for v in tokens.values() if v.startswith("<|")],
    )
    template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}{% for c in m['content'] %}{% if c['type'] == 'video' %}{{ '<|vision_start|><|video_pad|><|vision_end|>' }}{% else %}{{ c['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=2, temporal_patch_size=2, merge_size=2
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 256, "longest_edge": 256},
        ),
    )
    processor.save_pretrained(path / "processor")
    qwen = dict(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            tie_word_embeddings=True,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                partial_rotary_factor=0.5,
                mrope_section=[1, 1, 0],
            ),
            pad_token_id=0,
            eos_token_id=2,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=32,
            num_position_embeddings=16,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
        ),
        video_token_id=58,
        image_token_id=57,
        vision_start_token_id=61,
        vision_end_token_id=62,
    )
    wan = dict(
        in_dim=16,
        out_dim=16,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=48,
        num_heads=4,
        num_layers=1,
        text_len=3,
        attention_backend="sdpa",
        gradient_checkpointing=False,
    )
    config = deepcopy(DEFAULTS)
    # CPU regressions run in the same .venv-npu; they select the test runtime
    # explicitly rather than inheriting the production MindSpeed backend.
    config["distributed"].update(backend="accelerate", num_processes=1)
    config["model"].update(
        from_scratch=True,
        qwen_path=str(path / "processor"),
        wan_path=str(path / "processor"),
        processor_path=str(path / "processor"),
        gradient_checkpointing=False,
        video_uncond_prob=0.0,
        architecture=UMMConfig(qwen, 59, 60, wan).to_dict(),
        mask_token_id=59,
        video_mask_token_id=60,
    )
    config["data"].update(
        video=dict(
            num_frames=5,
            fps=4.0,
            height=16,
            width=16,
            qwen_num_frames=None,
            qwen_processor={},
        ),
        max_seq_length=256,
        representations=dict(qwen="cache", wan="cache"),
    )
    config["training"].update(
        device="cpu",
        mixed_precision="no",
        max_train_steps=2,
        gradient_accumulation_steps=2,
        use_ema=False,
        dataloader_workers=0,
        log_every=1,
        save_every=0,
        output_dir=str(path / "output"),
    )
    config["optimizer"].update(warmup_steps=0, decay_steps=0)
    validate_config(config)
    torch.manual_seed(123)
    model = UMM(UMMConfig.from_dict(config["model"]["architecture"])).eval()
    model.backbone.config._attn_implementation = (
        model.vision_encoder.visual.config._attn_implementation
    ) = "sdpa"
    with torch.no_grad():
        model.generator.head.head.weight.normal_(std=0.05)
    rgb = np.random.default_rng(42).integers(0, 256, (5, 16, 16, 3), dtype=np.uint8)
    np.save(path / "frames.npy", rgb)
    video = dict(id="v0", frames_path=str(path / "frames.npy"), fps=4.0)
    frames, clip = prepare_raw_video(video, config["data"]["video"])
    pixels, grid, times = qwen_video_inputs(
        frames, clip, processor, config["data"]["video"]
    )
    with torch.no_grad():
        features = model.vision_encoder(pixels, grid[None])[0]
    identity = checkpoint_id(path / "processor")
    metadata = dict(
        clip=clip, preprocessing=config["data"]["video"], checkpoint=identity
    )
    save_cache(
        path / "qwen.pt",
        dict(
            format=QWEN_FORMAT,
            features=features,
            grid_thw=grid,
            metadata=dict(metadata, qwen_timestamps=times),
        ),
    )
    latent = torch.randn(16, 2, 2, 2)
    save_cache(
        path / "wan.pt", dict(format=WAN_FORMAT, latent=latent, metadata=metadata)
    )
    video.update(qwen_cache=str(path / "qwen.pt"), wan_cache=str(path / "wan.pt"))
    rows = [
        dict(
            task=task,
            caption=f"{color} block moves",
            videos=[deepcopy(video)],
            sample_id=f"{task}-{i}",
        )
        for task in ("t2v", "v2t")
        for i, color in enumerate(("blue", "red"))
    ]
    manifest = path / "mixed.jsonl"
    manifest.write_text("".join(json.dumps(x) + "\n" for x in rows))
    config["data"]["sources"] = [{"path": str(manifest)}]
    return config, processor, model, rows, latent

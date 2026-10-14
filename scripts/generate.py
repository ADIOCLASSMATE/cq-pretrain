"""Generate one complete visual-token grid per prompt, with optional KL16 decode."""
import argparse
import json
from pathlib import Path

import torch

from configs.config import load_config
from data.sequence import make_image_sequence
from data.packing import collate_imagenet_flow_cache
from models.sampling import sample_visual_latents
from training.train import load_model, load_tokenizer


def prompt_batch(prompts, tokenizer, config):
    model = config["model"]
    prefix = config["data"].get("image", {}).get("caption_t2i_prefix", "Generate an image matching this description:")
    samples = []
    for index, prompt in enumerate(prompts):
        ids = torch.tensor(tokenizer.encode(f"{prefix} {prompt}", add_special_tokens=False), dtype=torch.long)
        sample = make_image_sequence(ids, torch.empty(0, dtype=torch.long),
            image_tokens=model["image_tokens_per_img"], boi_id=model["boi_token_id"],
            eoi_id=model["eoi_token_id"], mask_id=model["mask_token_id"], eos_id=tokenizer.eos_token_id)
        sample.update(image_latents=torch.zeros(model["image_tokens_per_img"], model["image_latent_dim"]),
                      image_loss_mask=sample["token_types"].eq(1), img_id=index,
                      sample_index=index, task_mode="t2i")
        samples.append(sample)
    return collate_imagenet_flow_cache(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", required=True, help="Directory for latents.pt, metadata and decoded PNGs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latents-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    config["model"]["checkpoint"] = args.checkpoint
    tokenizer = load_tokenizer(config)
    model = load_model(config).to(args.device).eval()
    batch = prompt_batch(args.prompt, tokenizer, config)
    batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    options = {key: value for key, value in config["sampling"].items()
               if key in {"steps", "solver", "cfg", "cfg_schedule", "temperature"}}
    seed = int(config["sampling"]["seed"])
    generator = torch.Generator(device=args.device).manual_seed(seed)
    noise = torch.randn(len(args.prompt), model.config.image_tokens_per_img, model.config.image_latent_dim,
                        device=args.device, generator=generator)
    with torch.no_grad(), torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16,
                                        enabled=config["training"]["mixed_precision"] == "bf16"):
        latents = sample_visual_latents(model, batch, initial_noise=noise, **options)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents.cpu(), "prompts": args.prompt, "seed": seed}, output / "latents.pt")
    if not args.latents_only:
        from PIL import Image
        from models.codec import KL16Codec
        codec = KL16Codec.from_pretrained(**config["codec"], device=args.device)
        for index, pixels in enumerate(codec.decode(latents)):
            array = pixels.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(array).save(output / f"image-{index:04d}.png")
    (output / "generation.json").write_text(json.dumps({"checkpoint": args.checkpoint,
        "prompts": args.prompt, "seed": seed, **options}, indent=2) + "\n")


if __name__ == "__main__":
    main()

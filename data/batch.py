"""Device-resident synthetic interleaved batches; no disk or decoder in timed steps."""

from dataclasses import dataclass, fields
import torch


@dataclass
class Batch:
    ids: torch.Tensor  # [B,S], content including clean images
    latents: torch.Tensor  # [total latent patches, 4C], selected targets only
    image_indices: torch.Tensor  # [images,max_input_tokens], -1 padding for native resolution
    text_indices: torch.Tensor  # supervised text positions in flattened [B,S]
    image_queries: torch.Tensor  # [B,S] boolean; query uses learned image mask
    sigma: torch.Tensor  # [B,S], image blocks have tied ranks
    segments: torch.Tensor  # [B,S], independent packed documents
    row_col: torch.Tensor  # [3,S,B], temporal/height/width RoPE
    noise: torch.Tensor  # [RF * total latent patches, 4C], draw-major
    times: torch.Tensor  # [RF * generation targets], draw-major
    generation_image_indices: torch.Tensor  # occurrence indices in this pack, in renderer order
    mainline: object = None
    generation: object = None
    vision: object = None  # Packed Qwen pixels + CPU-prepared spatial layout
    generation_unconditional: torch.Tensor | None = None  # [visuals], explicit non-CFG; None = normal C visibility
    generation_instruction_starts: torch.Tensor | None = (
        None  # [visuals], packed text boundary, document start = empty, -1 = unavailable
    )
    generation_instruction_dropped: torch.Tensor | None = None  # [visuals], read before the instruction start
    generation_null_positions: torch.Tensor | None = (
        None  # [visuals], first position after the document opener, -1 = none
    )

    @property
    def generation_image_count(self):
        return self.generation_image_indices.numel()

    def to(self, device, non_blocking=False):
        return Batch(
            **{
                f.name: value.to(device, non_blocking=non_blocking) if value is not None else None
                for f in fields(self)
                for value in (getattr(self, f.name),)
            }
        )

    def pin_memory(self, device="cuda"):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The argument 'device' of Tensor.pin_memory", category=DeprecationWarning
            )
            return Batch(
                **{
                    f.name: value.pin_memory(device=device)
                    if value is not None and (not isinstance(value, torch.Tensor) or value.numel())
                    else value
                    for f in fields(self)
                    for value in (getattr(self, f.name),)
                }
            )


def fake_batch(config, length=8192, batch_size=1, images_per_sequence=2, seed=42, device="cpu"):
    from data.synthetic import fake_vision_batch

    return fake_vision_batch(config, length, batch_size, images_per_sequence, seed, device)

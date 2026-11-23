"""Batched text-to-image sampling for the benchmarks.

Each distinct prompt runs the backbone once; the null condition, which reads no
prompt content, runs once per resolution. The conditional and null CFG branches
of a whole batch go through one DiT forward, same-size visuals share one SDPA
call per layer, and the VAE decodes in batches. Every image keeps its own seeded
initial latent, so an image depends only on (prompt, seed), not on batching.
"""

import os

import numpy as np
from PIL import Image
import torch

from data.generation import WanGenerationInputs, unpatch_latent
from evaluations.sampling import _dit_condition, make_batch, make_null_batch


def merge(layouts):
    """One packed generation layout for several single-target layouts."""
    lengths = tuple(x for lay in layouts for x in lay.lengths)
    queries = tuple(x for lay in layouts for x in lay.query_counts)
    width = max(queries)
    return WanGenerationInputs(
        pixels=(),
        grids=tuple(g for lay in layouts for g in lay.grids),
        lengths=lengths,
        query_counts=queries,
        video_ends=tuple(np.cumsum(lengths).tolist()),
        text_ends=tuple(np.cumsum(queries).tolist()),
        video_sample_indices=torch.cat([lay.video_sample_indices + i for i, lay in enumerate(layouts)]),
        condition_indices=torch.cat([lay.condition_indices + i * width for i, lay in enumerate(layouts)]),
        loss_weights=torch.empty(0),
        cos=torch.cat([lay.cos for lay in layouts]),
        sin=torch.cat([lay.sin for lay in layouts]),
        time_features=torch.empty(0),
    )


class Sampler:
    def __init__(self, runtime):
        self.runtime, self.model, self.cfg = runtime, runtime.model, runtime.cfg
        self.sampling = runtime.sampling
        self.device = next(self.model.parameters()).device
        self.conditions, self.null, self.checked = {}, None, 0
        s = self.sampling
        if s["steps"] < 1 or s["shift"] <= 0:
            raise ValueError("Positive steps and shift required")
        base = torch.linspace(1.0, 0.0, s["steps"] + 1)
        self.sigmas = s["shift"] * base / (1 + (s["shift"] - 1) * base)
        half = self.cfg.wan_freq_dim // 2
        phase = (self.sigmas[:-1] * 1000).double()[:, None] * 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
        self.time = torch.cat((phase.cos(), phase.sin()), -1).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def condition(self, prompt):
        if prompt in self.conditions:
            return self.conditions[prompt]
        r, size = self.runtime, self.sampling["resolution"]
        processed = r.processor(np.zeros((size, size, 3), np.uint8))
        ids = r.tokenizer.encode(prompt, add_special_tokens=False).ids
        full = make_batch(self.cfg, r.tokens, ids, processed, "T2I", tokenizer=r.tokenizer)
        null = make_null_batch(full, self.cfg) if self.sampling["guidance"] != 1.0 else None
        full = full.to(self.device)
        raw, projected, kv = _dit_condition(self.model, full)
        if null is not None and (self.null is None or self.checked < 2):
            # The null queries read no prompt content; check that on the first
            # prompts, then reuse one null condition for every image.
            value = _dit_condition(self.model, null.to(self.device))
            if self.null is None:
                self.null = value
            elif not torch.allclose(value[1], self.null[1], rtol=1e-3, atol=1e-3):
                raise ValueError("The null condition depends on the prompt; it cannot be shared")
            self.checked += 1
        if len(self.conditions) > 4096:
            self.conditions.clear()
        self.conditions[prompt] = (projected, kv["cached_kv"], full.generation)
        return self.conditions[prompt]

    @torch.no_grad()
    def __call__(self, specs):
        """specs: [(prompt, seed, destination png)], all at the configured resolution."""
        batch = int(self.sampling.get("batch", 32))
        for start in range(0, len(specs), batch):
            self._sample(specs[start : start + batch])

    def _sample(self, specs):
        cfg, guidance = self.cfg, float(self.sampling["guidance"])
        conditions = [self.condition(prompt) for prompt, _, _ in specs]
        branches = conditions + (
            [(self.null[1], self.null[2]["cached_kv"], conditions[0][2])] * len(specs) if guidance != 1.0 else []
        )
        layout = merge([c[2] for c in branches])
        projected = torch.cat([c[0] for c in branches])
        kv = tuple(tuple(torch.cat(t) for t in zip(*layer)) for layer in zip(*(c[1] for c in branches)))
        n, length = len(specs), conditions[0][2].lengths[0]
        if any(len(c[2].lengths) != 1 or c[2].lengths[0] != length for c in conditions):
            raise ValueError("Batched sampling needs one same-size target per prompt")
        latent = torch.cat(
            [
                torch.randn(length, cfg.latent_channels * 4, generator=torch.Generator(device="cpu").manual_seed(seed))
                for _, seed, _ in specs
            ]
        ).to(self.device)
        for i, (start, end) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            x = latent if guidance == 1.0 else torch.cat((latent, latent))
            time = self.time[i : i + 1].expand(len(branches), -1)
            velocity = self.model.renderer.predict(x, None, time, layout, projected, cached_kv=kv).float()
            if guidance != 1.0:
                cond, uncond = velocity.chunk(2)
                velocity = uncond + guidance * (cond - uncond)
            latent = latent + (float(end) - float(start)) * velocity
        if not torch.isfinite(latent).all().item():
            raise FloatingPointError("Nonfinite generated latent")
        grid = conditions[0][2].grids[0]
        pieces = [unpatch_latent(piece, grid) for piece in latent.split(length)]
        frames = self.runtime._preparer().vae.decode_batch(pieces)
        for (_, _, dest), video in zip(specs, frames):
            if not torch.isfinite(video).all().item():
                raise FloatingPointError("Nonfinite decoded visual")
            image = video.add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 3, 0)[0].cpu().numpy()
            dest.parent.mkdir(parents=True, exist_ok=True)
            writing = dest.with_name("." + dest.stem + ".writing.png")
            Image.fromarray(image).save(writing)
            os.replace(writing, dest)

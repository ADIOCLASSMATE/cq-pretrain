"""
Cache-backed ImageNet VAE-posterior dataset for unified image-flow training.

The cache stores scaled posterior mean/std rather than one frozen latent:
    {"posterior_stats": [N, image_tokens, 2 * latent_dim], "img_ids": [N], ...}

Training samples a fresh latent for every image and epoch. Validation uses one
fixed posterior sample per image so checkpoint metrics remain comparable.

This loader supports exactly two conditioning modes:
    class:    class_name <|boi|> image <|eoi|> <eos>
    caption:
        T2I: fixed_T2I_prefix caption <|boi|> image <|eoi|> <eos>
        I2T: fixed_I2T_prefix <|boi|> image <|eoi|> caption <eos>

Caption joint training uses disjoint targets. T2I rows supervise only image
flow, while I2T rows supervise only caption tokens (and their final EOS).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .cache import ImageNetSyntheticTextIndex, load_sharded_posterior
from .sequence import make_image_sequence
from .manifest import CaptionManifest, read_manifest, read_synset_names


DEFAULT_CAPTION_PREFIX = "Generate an image matching this description:"
DEFAULT_I2T_PREFIX = "Describe this image in one detailed caption:"
POSTERIOR_CACHE_FORMAT = "imagenet_kl16_scaled_posterior_v1"
POSTERIOR_STATS_LAYOUT = "scaled_mean_then_scaled_std"


class ImageNetFlowCacheDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        tokenizer,
        boi_token_id: int,
        eoi_token_id: int,
        mask_token_id: int,
        eos_token_id: int,
        image_tokens_per_img: int = 256,
        image_latent_dim: int = 16,
        manifest_jsonl: Optional[str] = None,
        expected_split: Optional[str] = None,
        expected_records: Optional[int] = None,
        synset_mapping_path: Optional[str] = None,
        conditioning_mode: Optional[str] = None,
        caption_jsonl: Optional[str | Sequence[str]] = None,
        caption_text_key: str = "recaption_short",
        caption_list_key: Optional[str] = "captions",
        caption_list_text_key: str = "text",
        caption_path_key: str = "path",
        caption_id_key: str = "id",
        caption_validation_index: int = 0,
        t2i_prompt_validation_index: int = 0,
        caption_sequence_modes: Optional[Sequence[str]] = None,
        synthetic_text_index_manifest: Optional[str] = None,
        caption_t2i_prefix: str = DEFAULT_CAPTION_PREFIX,
        caption_i2t_prefix: str = DEFAULT_I2T_PREFIX,
        caption_include_original: bool = True,
        cache_caption_tokens: bool = False,
        max_seq_length: Optional[int] = None,
        model_context_length: Optional[int] = None,
        max_samples: int = -1,
        seed: int = 42,
        image_sigma_order: str = "joint",
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(self.cache_path)

        # Memory-map the merged posterior cache so distributed ranks and
        # persistent workers reuse OS pages instead of copying the full
        # ImageNet tensor into every process.
        obj = load_sharded_posterior(self.cache_path) if self.cache_path.suffix == ".json" else torch.load(
            str(self.cache_path),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if "posterior_stats" not in obj:
            raise ValueError(
                f"{self.cache_path} is not a posterior cache: missing "
                "'posterior_stats'. Frozen 'latents' caches are no longer supported."
            )
        metadata = obj.get("metadata", {})
        if metadata.get("format") != POSTERIOR_CACHE_FORMAT:
            raise ValueError(
                f"Unsupported posterior cache format in {self.cache_path}: "
                f"{metadata.get('format')!r}; expected {POSTERIOR_CACHE_FORMAT!r}."
            )
        if metadata.get("stats_layout") != POSTERIOR_STATS_LAYOUT:
            raise ValueError(
                f"Unsupported posterior stats layout in {self.cache_path}: "
                f"{metadata.get('stats_layout')!r}; expected {POSTERIOR_STATS_LAYOUT!r}."
            )
        self.posterior_stats = obj["posterior_stats"]
        self.img_ids = obj.get("img_ids", torch.arange(self.posterior_stats.shape[0]))
        if max_samples is not None and max_samples > 0:
            self.posterior_stats = self.posterior_stats[:max_samples]
            self.img_ids = self.img_ids[:max_samples]

        expected_shape = (
            int(image_tokens_per_img),
            2 * int(image_latent_dim),
        )
        if (
            self.posterior_stats.ndim != 3
            or tuple(self.posterior_stats.shape[1:]) != expected_shape
        ):
            raise ValueError(
                "posterior_stats must have shape "
                f"[N, {expected_shape[0]}, {expected_shape[1]}], got "
                f"{tuple(self.posterior_stats.shape)}."
            )
        if (
            self.img_ids.ndim != 1
            or self.img_ids.shape[0] != self.posterior_stats.shape[0]
        ):
            raise ValueError(
                "img_ids must have one entry per posterior row, got "
                f"posterior_stats={tuple(self.posterior_stats.shape)}, "
                f"img_ids={tuple(self.img_ids.shape)}."
            )
        if not self.posterior_stats.is_floating_point():
            raise ValueError(
                f"posterior_stats must be floating point, got {self.posterior_stats.dtype}."
            )

        self.tokenizer = tokenizer
        self.boi_id = int(boi_token_id)
        self.eoi_id = int(eoi_token_id)
        self.mask_id = int(mask_token_id)
        self.eos_id = int(eos_token_id)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.image_latent_dim = int(image_latent_dim)
        self._empty_text_ids = torch.empty(0, dtype=torch.long)
        self.max_seq_length = int(max_seq_length) if max_seq_length else None
        self.model_context_length = (
            int(model_context_length) if model_context_length else 32768
        )
        self.seed = int(seed)
        self.image_sigma_order = str(image_sigma_order).strip().lower()
        if self.image_sigma_order != "joint":
            raise ValueError(
                f"Unknown image_sigma_order={image_sigma_order!r}; "
                "UMM v3 supports only 'joint'."
            )
        # DataLoader workers keep their own Dataset object, so a plain Python
        # integer would become stale when persistent_workers=True.  Tensor
        # storage remains shared after the Dataset is sent to workers, which
        # lets the training process advance the epoch without recreating them.
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
        self.training_index_mask: Optional[torch.Tensor] = None

        self.expected_split = (
            str(expected_split).strip().lower() if expected_split else None
        )
        if self.expected_split not in {None, "train", "val"}:
            raise ValueError(
                "expected_split must be 'train' or 'val', "
                f"got {expected_split!r}"
            )
        self.expected_records = (
            int(expected_records) if expected_records is not None else None
        )
        if self.expected_records is not None and self.expected_records <= 0:
            raise ValueError("expected_records must be positive")
        (
            self.synsets,
            self.source_paths,
            self.source_paths_full,
            self.manifest_splits,
        ) = read_manifest(manifest_jsonl)
        if self.expected_records is not None:
            if len(self) != self.expected_records:
                raise ValueError(
                    "posterior cache record count mismatch: "
                    f"{len(self)} != {self.expected_records}"
                )
            if len(self.synsets) != self.expected_records:
                raise ValueError(
                    "ImageNet manifest record count mismatch: "
                    f"{len(self.synsets)} != {self.expected_records}"
                )
        observed_splits = set(self.manifest_splits.values())
        if self.expected_split is not None and observed_splits != {
            self.expected_split
        }:
            raise ValueError(
                "ImageNet manifest split mismatch: "
                f"expected={self.expected_split!r}, "
                f"observed={sorted(observed_splits)!r}"
            )
        self.dataset_split = (
            self.expected_split
            if self.expected_split is not None
            else next(iter(observed_splits))
            if len(observed_splits) == 1
            else None
        )
        self.synset_names = read_synset_names(synset_mapping_path)
        self.caption_jsonl = caption_jsonl
        self.caption_text_key = str(caption_text_key)
        self.caption_list_key = str(caption_list_key) if caption_list_key else None
        self.caption_list_text_key = str(caption_list_text_key)
        self.caption_path_key = str(caption_path_key)
        self.caption_id_key = str(caption_id_key)
        self.caption_validation_index = int(caption_validation_index)
        if self.caption_validation_index < 0:
            raise ValueError("caption_validation_index must be non-negative")
        self.t2i_prompt_validation_index = int(t2i_prompt_validation_index)
        if self.t2i_prompt_validation_index < 0:
            raise ValueError("t2i_prompt_validation_index must be non-negative")
        self.caption_sequence_modes = self._normalize_caption_modes(
            caption_sequence_modes
        )
        self.synthetic_text_index_manifest = (
            str(synthetic_text_index_manifest)
            if synthetic_text_index_manifest
            else None
        )
        self.synthetic_text_index: Optional[ImageNetSyntheticTextIndex] = None
        self.caption_t2i_prefix = self._normalize_caption_prefix(
            caption_t2i_prefix,
            "caption_t2i_prefix",
        )
        self.caption_i2t_prefix = self._normalize_caption_prefix(
            caption_i2t_prefix,
            "caption_i2t_prefix",
        )
        self.caption_include_original = bool(caption_include_original)
        self.cache_caption_tokens = bool(cache_caption_tokens)
        self.text_cache: Dict[str, torch.Tensor] = {}
        self.sequence_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        if conditioning_mode is None:
            conditioning_mode = "caption" if caption_jsonl else "class"
        self.conditioning_mode = str(conditioning_mode).strip().lower()
        if self.conditioning_mode not in {"class", "caption"}:
            raise ValueError(
                f"Unknown conditioning_mode={conditioning_mode!r}; "
                "expected 'class' or 'caption'."
            )

        if not self.synsets:
            raise ValueError("class/caption conditioning requires manifest_jsonl")
        if self.conditioning_mode == "class" and not self.synset_names:
            raise ValueError("conditioning_mode='class' requires synset_mapping_path")
        self.caption_manifest = CaptionManifest(
            img_ids=self.img_ids, source_paths=self.source_paths,
            text_key=self.caption_text_key, list_key=self.caption_list_key,
            list_text_key=self.caption_list_text_key, path_key=self.caption_path_key,
            id_key=self.caption_id_key, include_original=self.caption_include_original,
        )
        self.captions: Dict[int, Tuple[str, ...]] = {}
        if self.conditioning_mode == "caption":
            if not caption_jsonl:
                raise ValueError("conditioning_mode='caption' requires caption_jsonl")
            if self.synthetic_text_index_manifest:
                self.synthetic_text_index = ImageNetSyntheticTextIndex(
                    self.synthetic_text_index_manifest
                )
                if (
                    self.dataset_split is not None
                    and self.synthetic_text_index.split != self.dataset_split
                ):
                    raise ValueError(
                        "posterior/manifest and synthetic-text splits differ: "
                        f"{self.dataset_split!r} != "
                        f"{self.synthetic_text_index.split!r}"
                    )
                if self.synthetic_text_index.row_count < len(self):
                    raise ValueError(
                        "synthetic text index is shorter than the posterior cache: "
                        f"{self.synthetic_text_index.row_count} < {len(self)}"
                    )
                expected_ids = torch.arange(
                    1,
                    len(self) + 1,
                    dtype=self.img_ids.dtype,
                )
                if not torch.equal(self.img_ids.cpu(), expected_ids):
                    raise ValueError(
                        "synthetic text seek index requires cache img_ids ordered "
                        "contiguously from 1"
                    )
            else:
                self.captions = self.caption_manifest.load(caption_jsonl)
                if not self.captions:
                    raise ValueError(
                        f"No captions matched cache img_ids from {caption_jsonl}."
                    )

    @staticmethod
    def _normalize_caption_modes(
        modes: Optional[Sequence[str]],
    ) -> Tuple[str, ...]:
        # Preserve the existing caption-conditioned T2I contract unless a
        # config explicitly opts into joint training.
        if modes is None:
            return ("t2i",)
        normalized = tuple(
            str(mode).strip().lower()
            for mode in modes
            if str(mode).strip()
        )
        if not normalized:
            raise ValueError("caption_sequence_modes must not be empty")
        invalid = sorted(set(normalized) - {"t2i", "i2t"})
        if invalid:
            raise ValueError(
                "caption_sequence_modes supports only 't2i' and 'i2t', "
                f"got {invalid}"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("caption_sequence_modes must not contain duplicates")
        return normalized

    @staticmethod
    def _normalize_caption_prefix(value: str, field: str) -> str:
        normalized = " ".join(str(value).split())
        if not normalized:
            raise ValueError(f"{field} must not be empty")
        return normalized


    def __len__(self) -> int:
        return int(self.posterior_stats.shape[0])

    @property
    def epoch(self) -> int:
        """Current epoch, shared with persistent DataLoader worker copies."""

        return int(self._epoch_state.item())

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._epoch_state.fill_(int(epoch))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_training_indices(self, indices: Sequence[int]) -> None:
        mask = torch.zeros(len(self), dtype=torch.bool)
        if len(indices) > 0:
            mask[torch.as_tensor(list(indices), dtype=torch.long)] = True
        self.training_index_mask = mask

    def _is_training_index(self, idx: int) -> bool:
        return self.training_index_mask is None or bool(
            self.training_index_mask[int(idx)]
        )

    def _sample_posterior(
        self,
        idx: int,
        sample_epoch: int,
    ) -> Tuple[torch.Tensor, int]:
        stats = self.posterior_stats[int(idx)]
        mean = stats[..., : self.image_latent_dim].float()
        std = stats[..., self.image_latent_dim :].float()
        posterior_seed = self._stable_sample_seed(
            int(idx), int(sample_epoch), "vae_posterior"
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(posterior_seed)
        noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32)
        latents = (mean + std * noise).to(dtype=stats.dtype)
        return self._apply_latent_layout(latents), posterior_seed

    def _stable_sample_seed(
        self,
        idx: int,
        current_epoch: int,
        purpose: str,
    ) -> int:
        purpose_offsets = {
            "vae_posterior": 11,
            "caption_task_offset": 23,
            "caption_validation_task": 37,
            "t2i_text_cycle_offset": 41,
            "i2t_text_cycle_offset": 43,
            "image_reveal_order": 53,
            "cfg_dropout": 67,
        }
        if purpose not in purpose_offsets:
            raise ValueError(
                f"missing arithmetic seed offset for purpose={purpose!r}"
            )
        return (
            int(self.seed)
            + 1_000_003 * int(current_epoch)
            + 97_409 * int(idx)
            + purpose_offsets[purpose]
        ) & ((1 << 63) - 1)

    def _apply_latent_layout(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.reshape(self.image_tokens_per_img, self.image_latent_dim)

    def _text_ids(self, text: str, cache: bool = True) -> torch.Tensor:
        text = text.strip()
        if not text:
            return torch.empty(0, dtype=torch.long)
        if not cache:
            return torch.tensor(
                self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long
            )
        cached = self.text_cache.get(text)
        if cached is None:
            cached = torch.tensor(
                self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long
            )
            self.text_cache[text] = cached
        return cached

    def _make_sequence_tensors(self, prefix_ids, suffix_ids, *, label_suffix=False):
        return make_image_sequence(
            prefix_ids, suffix_ids, image_tokens=self.image_tokens_per_img,
            boi_id=self.boi_id, eoi_id=self.eoi_id, mask_id=self.mask_id,
            eos_id=self.eos_id, max_seq_length=self.max_seq_length,
            label_suffix=label_suffix,
        )

    def _class_name_for_img_id(self, img_id: int) -> str:
        synset = self.synsets.get(int(img_id), "")
        class_name, _ = self.synset_names.get(synset, (synset, synset))
        class_name = str(class_name).strip()
        if not class_name:
            raise ValueError(f"No class name found for img_id={img_id}.")
        return class_name

    def _class_sequence_tensors(self, img_id: int) -> Dict[str, torch.Tensor]:
        class_name = self._class_name_for_img_id(img_id)
        cached = self.sequence_cache.get(class_name)
        if cached is None:
            cached = self._make_sequence_tensors(
                self._text_ids(class_name), self._empty_text_ids
            )
            self.sequence_cache[class_name] = cached
        return cached

    def _indexed_caption_texts(
        self,
        idx: int,
        img_id: int,
    ) -> Tuple[str, ...]:
        if self.synthetic_text_index is None:
            captions = self.captions.get(int(img_id))
            if captions is None:
                raise KeyError(f"No caption for img_id={img_id}")
            return captions
        row = self.synthetic_text_index.read_caption(int(idx))
        if int(row.get("manifest_index", -1)) != int(idx):
            raise ValueError(
                f"caption index identity mismatch at cache row {idx}: "
                f"manifest_index={row.get('manifest_index')!r}"
            )
        if int(row.get("img_id", -1)) != int(img_id):
            raise ValueError(
                f"caption img_id mismatch at cache row {idx}: "
                f"{row.get('img_id')!r} != {img_id}"
            )
        captions = self.caption_manifest.captions_from_row(
            row,
            self.caption_manifest.paths(self.caption_jsonl)[0],
            int(idx) + 1,
        )
        if not captions:
            raise ValueError(f"No selected synthetic captions for img_id={img_id}")
        return captions

    def _indexed_t2i_texts(
        self,
        idx: int,
        img_id: int,
    ) -> Tuple[str, ...]:
        if self.synthetic_text_index is None:
            return self._indexed_caption_texts(idx, img_id)
        row = self.synthetic_text_index.read_t2i(int(idx))
        expected_stem = Path(self.source_paths[int(img_id)]).stem
        expected_split = self.synthetic_text_index.split
        expected_image_id = f"{expected_split}/{expected_stem}"
        if row.get("image_id") != expected_image_id:
            raise ValueError(
                f"T2I image identity mismatch at cache row {idx}: "
                f"{row.get('image_id')!r} != {expected_image_id!r}"
            )
        prompt_rows = row.get("model_result", {}).get("prompts")
        if not isinstance(prompt_rows, list):
            raise ValueError(f"T2I prompts are missing for img_id={img_id}")
        prompts = tuple(
            str(prompt.get("prompt", "")).strip()
            for prompt in prompt_rows
            if str(prompt.get("prompt", "")).strip()
        )
        if len(prompts) != len(prompt_rows) or not prompts:
            raise ValueError(f"T2I prompts contain empty text for img_id={img_id}")
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"T2I prompts contain duplicates for img_id={img_id}")
        return prompts

    def _caption_sequence_tensors(
        self,
        img_id: int,
        idx: int,
        sample_epoch: int,
        is_training: bool,
    ) -> Tuple[Dict[str, torch.Tensor], int, int, str]:
        if is_training:
            task_count = len(self.caption_sequence_modes)
            task_offset = (
                self._stable_sample_seed(int(idx), 0, "caption_task_offset")
                % task_count
            )
            task_phase = task_offset + int(sample_epoch)
            task_index = task_phase % task_count
            # Count how many times this sample has visited the selected task.
            # For a single-mode T2I dataset this must advance every epoch;
            # otherwise its per-image prompt would remain frozen forever.
            task_occurrence = task_phase // task_count
        else:
            task_index = (
                self._stable_sample_seed(int(idx), 0, "caption_validation_task")
                % len(self.caption_sequence_modes)
            )
            task_occurrence = 0
        task_mode = self.caption_sequence_modes[task_index]
        texts = (
            self._indexed_t2i_texts(idx, img_id)
            if task_mode == "t2i"
            else self._indexed_caption_texts(idx, img_id)
        )
        caption_count = len(texts)
        if is_training and caption_count > 1:
            offset = (
                self._stable_sample_seed(
                    int(idx),
                    0,
                    f"{task_mode}_text_cycle_offset",
                )
                % caption_count
            )
            caption_index = (offset + task_occurrence) % caption_count
        else:
            caption_index = (
                self.t2i_prompt_validation_index
                if task_mode == "t2i"
                else self.caption_validation_index
            )
            if caption_index >= caption_count:
                raise ValueError(
                    f"{task_mode} validation index={caption_index} is out of "
                    f"range for img_id={img_id} with {caption_count} texts"
                )
        caption = texts[caption_index]
        if task_mode == "t2i":
            serialized_prompt = f"{self.caption_t2i_prefix} {caption}"
            prefix_ids = self._text_ids(
                serialized_prompt,
                cache=self.cache_caption_tokens,
            )
            suffix_ids = self._empty_text_ids
        else:
            serialized_prompt = f"{self.caption_i2t_prefix}\n{caption}"
            prefix_ids = self._text_ids(
                self.caption_i2t_prefix,
                cache=self.cache_caption_tokens,
            )
            suffix_ids = self._text_ids(
                caption,
                cache=self.cache_caption_tokens,
            )
        cached = (
            self.sequence_cache.get(serialized_prompt)
            if self.cache_caption_tokens
            else None
        )
        if cached is not None:
            return cached, caption_index, caption_count, task_mode
        sequence = self._make_sequence_tensors(
            prefix_ids,
            suffix_ids,
            label_suffix=task_mode == "i2t",
        )
        if self.cache_caption_tokens:
            self.sequence_cache[serialized_prompt] = sequence
        return sequence, caption_index, caption_count, task_mode

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Capture one shared epoch value so every stochastic choice for this
        # sample uses the same epoch even if the parent advances immediately.
        current_epoch = self.epoch
        is_training = self._is_training_index(int(idx))
        # A held-out sample must not change when the training epoch advances.
        sample_epoch = current_epoch if is_training else 0
        latents, posterior_seed = self._sample_posterior(int(idx), sample_epoch)
        img_id = int(self.img_ids[idx].item())
        reveal_seed = self._stable_sample_seed(
            int(idx), sample_epoch, "image_reveal_order"
        )
        cfg_dropout_seed = self._stable_sample_seed(
            int(idx), sample_epoch, "cfg_dropout"
        )
        if self.conditioning_mode == "class":
            sequence = self._class_sequence_tensors(img_id)
            caption_index = -1
            caption_count = 0
            task_mode = "class"
        else:
            (
                sequence,
                caption_index,
                caption_count,
                task_mode,
            ) = self._caption_sequence_tensors(
                img_id=img_id,
                idx=int(idx),
                sample_epoch=sample_epoch,
                is_training=is_training,
            )

        length = int(sequence["input_ids"].numel())
        if self.max_seq_length is not None and length > self.max_seq_length:
            raise ValueError(
                f"ImageNetFlowCacheDataset item length {length} exceeds max_seq_length={self.max_seq_length}. "
                "Increase max_seq_length."
            )
        if self.model_context_length is not None and length > self.model_context_length:
            raise ValueError(
                f"Serialized sample length {length} exceeds model context "
                f"window {self.model_context_length} for img_id={img_id}."
            )
        image_loss_mask = torch.zeros(length, dtype=torch.bool)
        if task_mode in {"class", "t2i"}:
            image_start = int(sequence["image_start"].item())
            image_loss_mask[
                image_start : image_start + self.image_tokens_per_img
            ] = True
        result = {
            "input_ids": sequence["input_ids"],
            "token_types": sequence["token_types"],
            "labels": sequence["labels"],
            "image_loss_mask": image_loss_mask,
            "image_latents": latents,
            "prompt_len": sequence["prompt_len"],
            "suffix_len": sequence["suffix_len"],
            "image_start": sequence["image_start"],
            "img_id": torch.tensor(img_id, dtype=torch.long),
            "synset": self.synsets.get(img_id, ""),
            "source_path": self.source_paths_full.get(img_id, ""),
            "sample_index": torch.tensor(int(idx), dtype=torch.long),
            "serialized_length": torch.tensor(length, dtype=torch.long),
            "caption_index": torch.tensor(caption_index, dtype=torch.long),
            "caption_count": torch.tensor(caption_count, dtype=torch.long),
            "task_mode": task_mode,
            "posterior_seed": torch.tensor(posterior_seed, dtype=torch.long),
            "reveal_seed": torch.tensor(reveal_seed, dtype=torch.long),
            "image_sigma_order": self.image_sigma_order,
            "cfg_dropout_seed": torch.tensor(cfg_dropout_seed, dtype=torch.long),
        }
        return result

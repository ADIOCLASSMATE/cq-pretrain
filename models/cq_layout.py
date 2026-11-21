"""Prepared CQ tensors shared by model execution and batch preparation."""

import copy
from dataclasses import dataclass, fields, replace
import torch
from ops.attention_layout import AttentionPlan
from ops.gated_delta_layout import DeltaPlan


@dataclass
class MainlinePlan:
    query_positions: torch.Tensor
    query_image: torch.Tensor
    query_rope: torch.Tensor
    output_positions: torch.Tensor
    condition_indices: torch.Tensor
    cq_mask: torch.Tensor
    cq_valid: torch.Tensor
    cq_plan: AttentionPlan | None = None
    output_size: int = 0
    delta_plan: DeltaPlan | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None
    query_unconditional: torch.Tensor | None = None  # [queries], no C reads; target positions stay intact
    cq_predicate: object = None  # CQPredicate: same mask, as vectors instead of [1,1,2S,S]
    query_read_sigma: torch.Tensor | None = None  # exclusive C cutoff; query_positions and RoPE stay at the target
    # Bidirectional visual Q: one (row_start, row_stop, key_start, key_stop) per
    # target. Rows index the [C ; Q] hidden sequence, keys are the contiguous C
    # prefix that target may read. Those rows read [prefix ; their own block].
    query_blocks: tuple = ()

    def gather_image_conditions(self, hidden, generation_image_indices):
        """Return one CQ row per generation target, in the renderer's sample order."""
        indices = self.condition_indices.index_select(0, generation_image_indices)
        # Selected rows can still have padding columns for unequal query counts.
        # The renderer consumes only the valid columns of each target.
        return hidden[indices.clamp_min(0)]

    def _map_tensors(self, fn):
        return replace(
            self,
            **{
                f.name: fn(value)
                for f in fields(self)
                if isinstance(value := getattr(self, f.name), torch.Tensor)
            },
        )

    def to(self, device, non_blocking=False):
        out = self._map_tensors(lambda t: t.to(device, non_blocking=non_blocking))
        if self.cq_plan is not None:
            out.cq_plan = self.cq_plan.bind(out.cq_mask)
        if self.delta_plan is not None:
            out.delta_plan = self.delta_plan.to(device, non_blocking=non_blocking)
        if self.cq_predicate is not None:
            out.cq_predicate = self.cq_predicate.to(device, non_blocking=non_blocking)
        return out

    def pin_memory(self, device):
        out = self._map_tensors(lambda t: t.pin_memory(device=device) if t.numel() else t)
        if self.cq_plan is not None:
            out.cq_plan = replace(self.cq_plan, owner=out.cq_mask, mask_pointer=out.cq_mask.data_ptr(),
                                  mask_version=out.cq_mask._version, tile_masks=())
        if self.delta_plan is not None:
            out.delta_plan = self.delta_plan.pin_memory(device)
        if self.cq_predicate is not None:
            out.cq_predicate = self.cq_predicate.pin_memory(device)
        return out

    def __deepcopy__(self, memo):
        out = self._map_tensors(lambda t: copy.deepcopy(t, memo))
        memo[id(self)] = out
        if self.cq_plan is not None:
            out.cq_plan = self.cq_plan.bind(out.cq_mask)
        out.delta_plan = copy.deepcopy(self.delta_plan, memo)
        out.cq_predicate = copy.deepcopy(self.cq_predicate, memo)
        return out

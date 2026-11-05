"""Prepared CQ tensors shared by model execution and batch preparation."""

import copy
from dataclasses import dataclass, fields, replace
import torch
from ops.attention_layout import AttentionPlan
from ops.gated_delta_layout import DeltaPlan


@dataclass
class MainlinePlan:
    method: str
    prefix: int
    query_positions: torch.Tensor
    query_image: torch.Tensor
    query_rope: torch.Tensor
    generation_positions: torch.Tensor
    generation_image: torch.Tensor
    generation_rope: torch.Tensor
    output_positions: torch.Tensor
    condition_indices: torch.Tensor
    generation_condition: torch.Tensor
    cq_mask: torch.Tensor
    cq_valid: torch.Tensor
    gen_mask: torch.Tensor
    gen_valid: torch.Tensor
    gen_suffix_mask: torch.Tensor
    cq_plan: AttentionPlan | None = None
    gen_plan: AttentionPlan | None = None
    output_size: int = 0
    delta_plan: DeltaPlan | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None
    query_unconditional: torch.Tensor | None = None  # [queries], no C reads; target positions stay intact

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
        for name in ("cq", "gen"):
            plan = getattr(self, name + "_plan")
            if plan is not None:
                setattr(out, name + "_plan", plan.bind(out.gen_suffix_mask if name == "gen" else out.cq_mask))
        if self.delta_plan is not None:
            out.delta_plan = self.delta_plan.to(device, non_blocking=non_blocking)
        return out

    def pin_memory(self, device):
        out = self._map_tensors(lambda t: t.pin_memory(device=device) if t.numel() else t)
        for name in ("cq", "gen"):
            plan = getattr(self, name + "_plan")
            if plan is not None:
                mask = out.gen_suffix_mask if name == "gen" else out.cq_mask
                setattr(
                    out,
                    name + "_plan",
                    replace(
                        plan,
                        mask_pointer=mask.data_ptr(),
                        mask_version=mask._version,
                        owner=mask,
                        tile_masks=(),
                    ),
                )
        if self.delta_plan is not None:
            out.delta_plan = self.delta_plan.pin_memory(device)
        return out

    def __deepcopy__(self, memo):
        out = self._map_tensors(lambda t: copy.deepcopy(t, memo))
        memo[id(self)] = out
        for name in ("cq", "gen"):
            plan = getattr(self, name + "_plan")
            if plan is not None:
                setattr(out, name + "_plan", plan.bind(out.gen_suffix_mask if name == "gen" else out.cq_mask))
        out.delta_plan = copy.deepcopy(self.delta_plan, memo)
        return out

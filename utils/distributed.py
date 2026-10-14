"""Small distributed operations shared by training and durable I/O."""
from __future__ import annotations

import torch.distributed as dist


def run_io_phase(accelerator, operation, *, description: str, main_process_only=False):
    """Propagate filesystem failures before ranks enter the next collective.

    Operations must not themselves contain unmatched collectives. Backend
    save/load collectives still depend on backend timeouts for killed ranks.
    """
    result = None
    error = None
    world_size = int(getattr(accelerator, "num_processes", 1))
    try:
        if not main_process_only or bool(getattr(accelerator, "is_main_process", True)):
            result = operation()
    except Exception as exc:
        if world_size == 1:
            raise
        error = f"rank {accelerator.process_index}: {type(exc).__name__}: {exc}"
    if world_size == 1:
        return result
    outcomes = [None] * world_size
    dist.all_gather_object(outcomes, {
        "error": error, "result": result if main_process_only else None,
    })
    errors = [item["error"] for item in outcomes if item["error"] is not None]
    if errors:
        raise RuntimeError(f"{description} failed: " + "; ".join(errors))
    return outcomes[0]["result"] if main_process_only else result

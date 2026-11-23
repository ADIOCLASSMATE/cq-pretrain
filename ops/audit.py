"""Which implementation actually ran, for every operator with more than one.

Each dispatch point reports the path it took: the vendor kernel it called, or
``reference`` for the portable PyTorch expression. Two uses:

- ``report()`` is written into summary.json and printed by the operator smoke
  tests, so a run can be audited after the fact instead of inferred from flags.
- ``PRETRAIN_REQUIRE_FUSED=1`` turns a reference path taken *on the training
  accelerator* into an immediate error. Ascend runs set it: a silent fallback
  to the eager expression is a correctness-shaped bug there, because it is both
  far slower and no longer the implementation the numbers were gated against.

CPU tensors never raise: the reference is the oracle the GPU gates compare
against, and the CPU correctness tests call it deliberately.
"""

import os
from collections import Counter

import distributed.accel as accel

CALLS = Counter()
STRICT = os.environ.get("PRETRAIN_REQUIRE_FUSED", "0") == "1"
REFERENCE = "reference"


def record(operator, path):
    CALLS[(operator, path)] += 1
    return path


def reference(operator, x, reason=""):
    """Record (and under STRICT reject) an accelerator tensor taking the reference path."""
    CALLS[(operator, REFERENCE)] += 1
    if STRICT and x is not None and accel.on_accelerator(x):
        detail = f" ({reason})" if reason else ""
        raise RuntimeError(
            f"PRETRAIN_REQUIRE_FUSED=1: {operator} fell back to its reference implementation "
            f"on {accel.BACKEND}{detail}. Fix the kernel wiring rather than training on the fallback."
        )
    return REFERENCE


def report():
    """{operator: {path: calls}} since process start."""
    paths = {}
    for (operator, path), count in sorted(CALLS.items()):
        paths.setdefault(operator, {})[path] = count
    return paths


def reset():
    CALLS.clear()

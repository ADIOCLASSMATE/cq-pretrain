"""Attribute every dispatched operator to the line of this repository that issued it.

The 910B profile of 2026-11-23 showed the device idle for about a third of each
step while the host issued ~351k kernel launches and ~1.9M torch ops per update
(evaluations/benchmarks/npu_910b_operator_profile_20261123.json). That profile
ran with ``with_stack=False``, so none of it could be traced to a call site and
every "optimization" after it would have been a guess.

This mode counts, it does not time: a ``TorchDispatchMode`` records
``(operator, innermost repository frame)`` for each dispatch. Counts are what
matters here, because a host-bound step is paid per launch, not per element.

Backward passes driven by C++ autograd nodes have no Python frame of ours on
the stack; those land under ``<engine>`` and are reported, not hidden. Python
autograd Functions in this repository (the GDN adapter, tiled attention, the
chunked CE, the activation checkpoints) do keep their frames, so the recompute
and custom-backward work stays attributable.

``SYNC_OPS`` are listed separately: each one moves a value from the device to
the host and therefore drains the launch queue, which costs far more on a
host-bound step than its own duration suggests.
"""

import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])

# The interpreter, the virtual environments and the vendor stacks all live
# under ROOT, so "starts with ROOT" alone would attribute every dispatch to
# torch's or MindSpeed's own frames and never reach the line that asked for it.
VENDORED = ("/.venv", "/site-packages/", "/.deps/", "/.python/", "/archive/")

# Dispatches that force the host to wait for the device.
SYNC_OPS = ("_local_scalar_dense", "item", "nonzero", "_assert_async", "equal", "allclose")


def _name(func):
    """`aten.empty_strided.default` rather than the full OpOverload repr."""
    text = str(func)
    return text.removeprefix("aten::")


class OperatorCensus:
    """Counts (operator, call site) pairs. Install with ``census(...)``."""

    def __init__(self):
        self.pairs = Counter()
        self.ops = Counter()
        self.sites = Counter()
        self.direct = Counter()
        self.syncs = Counter()
        self.steps = 0

    def record(self, name):
        site, direct = self._site()
        self.pairs[(name, site)] += 1
        self.ops[name] += 1
        self.sites[site] += 1
        if direct:
            self.direct[site] += 1
        if any(key in name for key in SYNC_OPS):
            self.syncs[(name, site)] += 1

    @staticmethod
    def _site():
        """Innermost frame inside this repository, and whether it issued the call itself.

        Walking frames directly costs a pointer chase per level; building a
        traceback object instead would format every frame of every dispatch and
        dominate the measurement it is meant to explain.

        The second value separates two very different things that resolve to the
        same line. A *direct* dispatch was written on that line. An *indirect*
        one was issued underneath it by library or autograd-engine frames, so
        the line is only where the work was entered -- a backward pass driven by
        C++ nodes on the calling thread resolves to whichever line called
        ``.backward()``, which would otherwise read as a precise culprit.
        """
        frame = sys._getframe(2)
        direct = True
        while frame is not None:
            path = frame.f_code.co_filename
            if path.endswith("ops/census.py"):
                # This module's own frames sit between every dispatch and its
                # caller; counting them as a hop would make nothing look direct.
                frame = frame.f_back
                continue
            if path.startswith(ROOT) and not any(part in path for part in VENDORED):
                return f"{path[len(ROOT) + 1 :]}:{frame.f_lineno}", direct
            direct = False
            frame = frame.f_back
        return "<engine>", False

    def report(self, top=60):
        return dict(
            steps=self.steps,
            total_dispatches=sum(self.ops.values()),
            dispatches_per_step=round(sum(self.ops.values()) / max(1, self.steps)),
            by_site=[
                dict(
                    site=site,
                    calls=count,
                    per_step=round(count / max(1, self.steps)),
                    written_on_this_line=self.direct[site],
                    issued_beneath_it=count - self.direct[site],
                )
                for site, count in self.sites.most_common(top)
            ],
            by_operator=[
                dict(operator=name, calls=count, per_step=round(count / max(1, self.steps)))
                for name, count in self.ops.most_common(top)
            ],
            by_site_and_operator=[
                dict(site=site, operator=name, calls=count, per_step=round(count / max(1, self.steps)))
                for (name, site), count in self.pairs.most_common(top)
            ],
            device_to_host_syncs=[
                dict(site=site, operator=name, calls=count, per_step=round(count / max(1, self.steps)))
                for (name, site), count in self.syncs.most_common(top)
            ],
            device_to_host_syncs_per_step=round(sum(self.syncs.values()) / max(1, self.steps)),
        )

    def write(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.report(), indent=2) + "\n")
        return path


def _mode_class():
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Mode(TorchDispatchMode):
        # FlexAttention is dispatched as a higher-order op. Without this a mode
        # that does not claim to handle one raises NotImplementedError and takes
        # the training step down with it, which is how the first L40S census
        # died: the backbone runs FlexAttention there and never did on Ascend.
        supports_higher_order_operators = True

        def __init__(self, counter):
            super().__init__()
            self.counter = counter

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.counter.record(_name(func))
            return func(*args, **(kwargs or {}))

    return _Mode


def census(counter, enabled):
    """Context manager counting this step's dispatches, or a no-op.

    A census that cannot be installed (an interposing dispatch mode from the
    vendor stack, say) must not end the run: the report says so instead.
    """
    if not enabled or counter is None:
        return nullcontext()
    try:
        mode = _mode_class()(counter)
    except Exception as error:  # the census is diagnostic; training is not
        counter.ops[f"<census unavailable: {type(error).__name__}>"] += 1
        return nullcontext()
    counter.steps += 1
    return mode

"""Sample from the original view distribution conditioned on release-time admission."""

from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
import hashlib
import math


_choice = ContextVar("length_choice", default=None)


@contextmanager
def choosing(fn):
    token = _choice.set(fn)
    try:
        yield
    finally:
        _choice.reset(token)


def index(n, original, key):
    fn = _choice.get()
    return fn(n, original, key) if fn is not None and n > 1 else original


def mass(tree):
    return Fraction(*tree["mass"]) if isinstance(tree, dict) else Fraction(tree)


def branch(children):
    if all(x == 1 for x in children):
        return 1
    if not any(children):
        return 0
    parts = [mass(c) for c in children]
    scale = math.lcm(*(p.denominator for p in parts))
    weights = [int(p * scale) for p in parts]
    total = sum(parts) / len(parts)
    return dict(mass=[total.numerator, total.denominator], weights=weights, children=children)


def pick(weights, key):
    position = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big") % sum(weights)
    for i, w in enumerate(weights):
        if position < w:
            return i
        position -= w
    raise ValueError("Empty admission weights")


def task_draw(row, seed, epoch, task, view_index, render):
    gate = row["stage1_views"]
    choices = [i for i, v in enumerate(row["views"]) if gate[i] and (task is None or v["task"] == task)]
    if not choices or view_index is not None and view_index not in choices:
        raise ValueError("Requested view is outside this training release")
    key = f"8k:{row['family_id']}:{seed}:{epoch}"
    if view_index is None:
        weights = branch([gate[i] for i in choices])
        weights = weights["weights"] if isinstance(weights, dict) else [1] * len(choices)
        view_index = choices[pick(weights, key + ":view")]
    tree = gate[view_index]
    if tree == 1:
        return render(row, seed, epoch, task, view_index)
    seen = {}

    def sample(n, original, slot):
        nonlocal tree
        if slot in seen:
            return seen[slot]
        if tree == 1:
            return original
        if len(tree["children"]) != n:
            raise ValueError("Training choices differ from the length index")
        i = pick(tree["weights"], key + ":" + str(view_index) + ":" + str(len(seen)))
        seen[slot] = i
        tree = tree["children"][i]
        return i

    with choosing(sample):
        result = render(row, seed, epoch, task, view_index)
    if tree != 1:
        raise ValueError("Incomplete training choice path")
    return result

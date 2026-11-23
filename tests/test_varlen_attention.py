"""Packed varlen attention means the same thing as the loop it replaces.

Every test here is the same question in a different place: does running all
segments in one call give exactly what running them one at a time gave? The
loop is written out explicitly in each case rather than imported, so these
stay oracles even if the implementation is rewritten again.

Device kernels are gated separately (scripts/validation/validate_varlen_npu.py);
this file pins the meaning in FP64 on CPU, where there is no kernel to blame.
"""

import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ops.attention import segment_lengths, varlen, varlen_reference


def loop(q, k, v, cu_q, cu_k):
    """One segment at a time: [Tq,H,D] x [Tk,H,D] -> [Tq,H,D]."""
    outputs = []
    start_q = start_k = 0
    for end_q, end_k in zip(cu_q, cu_k):
        query = q[start_q:end_q].transpose(0, 1)[None]
        key, value = (t[start_k:end_k].transpose(0, 1)[None] for t in (k, v))
        outputs.append(F.scaled_dot_product_attention(query, key, value)[0].transpose(0, 1))
        start_q, start_k = end_q, end_k
    return torch.cat(outputs)


def tensors(lengths_q, lengths_k, heads=2, dim=16, seed=0):
    torch.manual_seed(seed)
    make = lambda total: torch.randn(total, heads, dim, dtype=torch.float64)  # noqa: E731
    return make(sum(lengths_q)), make(sum(lengths_k)), make(sum(lengths_k))


def ends(lengths):
    total = 0
    out = []
    for length in lengths:
        total += length
        out.append(total)
    return tuple(out)


class SegmentTests(unittest.TestCase):
    def test_lengths_round_trip_through_cumulative_ends(self):
        for lengths in [(4,), (3, 7, 1), (5, 5, 5)]:
            self.assertEqual(segment_lengths(ends(lengths)), lengths)


class ReferenceTests(unittest.TestCase):
    def test_ragged_segments_match_the_loop(self):
        lengths_q = (6, 1, 9, 4)
        q, k, v = tensors(lengths_q, lengths_q)
        got = varlen_reference(q, k, v, ends(lengths_q), ends(lengths_q))
        torch.testing.assert_close(got, loop(q, k, v, ends(lengths_q), ends(lengths_q)), rtol=0, atol=0)

    def test_cross_segments_of_a_different_length_match_the_loop(self):
        lengths_q, lengths_k = (6, 1, 9), (3, 8, 2)
        q, k, v = tensors(lengths_q, lengths_k)
        got = varlen_reference(q, k, v, ends(lengths_q), ends(lengths_k))
        torch.testing.assert_close(got, loop(q, k, v, ends(lengths_q), ends(lengths_k)), rtol=0, atol=0)

    def test_the_batched_equal_length_path_agrees_with_the_loop(self):
        lengths_q, lengths_k = (5, 5, 5, 5), (7, 7, 7, 7)
        q, k, v = tensors(lengths_q, lengths_k)
        got = varlen_reference(q, k, v, ends(lengths_q), ends(lengths_k))
        torch.testing.assert_close(got, loop(q, k, v, ends(lengths_q), ends(lengths_k)), rtol=0, atol=1e-12)

    def test_a_single_segment_is_plain_attention(self):
        q, k, v = tensors((11,), (11,))
        got = varlen_reference(q, k, v, (11,), (11,))
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        )[0].transpose(0, 1)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_a_segment_cannot_read_another_segment(self):
        """Replacing one segment's keys leaves every other segment's output alone."""
        lengths = (4, 6, 3)
        q, k, v = tensors(lengths, lengths)
        before = varlen_reference(q, k, v, ends(lengths), ends(lengths))
        k, v = k.clone(), v.clone()
        k[4:10] += 100.0
        v[4:10] += 100.0
        after = varlen_reference(q, k, v, ends(lengths), ends(lengths))
        torch.testing.assert_close(after[:4], before[:4], rtol=0, atol=0)
        torch.testing.assert_close(after[10:], before[10:], rtol=0, atol=0)
        self.assertFalse(torch.allclose(after[4:10], before[4:10]))

    def test_the_dispatch_takes_the_reference_on_cpu(self):
        lengths = (3, 5)
        q, k, v = tensors(lengths, lengths)
        got = varlen(q, k, v, ends(lengths), ends(lengths))
        torch.testing.assert_close(got, loop(q, k, v, ends(lengths), ends(lengths)), rtol=0, atol=0)


class WanPackedAttentionTests(unittest.TestCase):
    """The DiT call site: [draws, patches, heads, dim] over per-visual segments."""

    def wan_loop(self, q, k, v, q_ends, k_ends):
        """What models.generator_wan21.packed_attention did before the varlen form."""
        result = []
        start_q = start_k = 0
        for end_q, end_k in zip(q_ends, k_ends):
            query = q[:, start_q:end_q]
            folded = query.reshape(1, -1, *query.shape[2:]) if k.shape[0] == 1 else query
            key, value = (t[:, start_k:end_k].transpose(1, 2) for t in (k, v))
            attended = F.scaled_dot_product_attention(folded.transpose(1, 2), key, value)
            result.append(attended.transpose(1, 2).reshape_as(query))
            start_q, start_k = end_q, end_k
        return torch.cat(result, dim=1).flatten(0, 1)

    def check(self, draws, lengths_q, lengths_k, shared):
        from models.generator_wan21 import packed_attention

        torch.manual_seed(1)
        rows = 1 if shared else draws
        q = torch.randn(draws, sum(lengths_q), 2, 16, dtype=torch.float64)
        k = torch.randn(rows, sum(lengths_k), 2, 16, dtype=torch.float64)
        v = torch.randn(rows, sum(lengths_k), 2, 16, dtype=torch.float64)
        got = packed_attention(q, k, v, ends(lengths_q), ends(lengths_k))
        expected = self.wan_loop(q, k, v, ends(lengths_q), ends(lengths_k))
        self.assertEqual(got.shape, expected.shape)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)

    def test_self_attention_over_native_resolution_visuals(self):
        self.check(draws=1, lengths_q=(6, 11, 3), lengths_k=(6, 11, 3), shared=False)

    def test_self_attention_with_several_draws(self):
        self.check(draws=3, lengths_q=(6, 11, 3), lengths_k=(6, 11, 3), shared=False)

    def test_cross_attention_reads_the_shared_condition(self):
        self.check(draws=1, lengths_q=(6, 11, 3), lengths_k=(4, 9, 2), shared=True)

    def test_cross_attention_repeats_the_condition_for_every_draw(self):
        self.check(draws=3, lengths_q=(6, 11, 3), lengths_k=(4, 9, 2), shared=True)

    def test_equal_sized_visuals_still_agree(self):
        self.check(draws=2, lengths_q=(8, 8, 8), lengths_k=(8, 8, 8), shared=False)

    def test_one_draw_is_independent_of_the_others(self):
        """Draw-major flattening: draw d's output must not depend on draw d+1."""
        from models.generator_wan21 import packed_attention

        torch.manual_seed(2)
        lengths = (5, 9)
        patches = sum(lengths)
        q = torch.randn(2, patches, 2, 16, dtype=torch.float64)
        k = torch.randn(2, patches, 2, 16, dtype=torch.float64)
        v = torch.randn(2, patches, 2, 16, dtype=torch.float64)
        both = packed_attention(q, k, v, ends(lengths), ends(lengths)).reshape(2, patches, 2, 16)
        for draw in range(2):
            alone = packed_attention(
                q[draw : draw + 1], k[draw : draw + 1], v[draw : draw + 1], ends(lengths), ends(lengths)
            )
            torch.testing.assert_close(both[draw], alone, rtol=0, atol=1e-12)


class VisionAttentionTests(unittest.TestCase):
    """The vision tower call site: one segment per image, [T, heads, dim]."""

    def test_the_tower_attention_matches_its_per_image_loop(self):
        from models.vision_encoder import VisionAttention, rotary

        torch.manual_seed(3)
        cfg = dict(hidden_size=64, num_heads=4)
        lengths = (4, 9, 2)
        total = sum(lengths)
        attention = VisionAttention(cfg).double()
        heads, dim = attention.heads, attention.dim
        layout = SimpleNamespace(
            lengths=lengths,
            ends=ends(lengths),
            cos=torch.randn(total, dim, dtype=torch.float64),
            sin=torch.randn(total, dim, dtype=torch.float64),
        )
        x = torch.randn(total, 64, dtype=torch.float64)
        q, k, v = attention.qkv(x).reshape(total, 3, heads, dim).unbind(1)
        # The same rotation the module applies; only the attending differs.
        q, k = rotary(q, layout.cos, layout.sin), rotary(k, layout.cos, layout.sin)
        expected = attention.proj(loop(q, k, v, layout.ends, layout.ends).reshape(total, -1))
        torch.testing.assert_close(attention(x, layout, True), expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

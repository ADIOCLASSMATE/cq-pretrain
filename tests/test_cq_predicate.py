"""Require the CQ predicate to reproduce the dense mask bitwise.

If this passes on real plan layouts -- multi-document packs, tied image blocks,
unconditional queries and invalid/padding rows -- then a kernel driven by the
predicate computes exactly the attention the dense mask describes.
"""
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from configs.config import tiny_config
from data.batch import fake_batch
from ops.attention_layout import document_cq_mask, cq_mask_vectors


ASSETS = Path(__file__).resolve().parents[1] / '.assets/Qwen3.5-2B-Base'


def wan_cfg(**kw):
    cfg = tiny_config(**kw)
    cfg.renderer = 'wan'
    cfg.latent_channels = 16
    cfg.flow_draws = 1
    cfg.vision_encoder = 'qwen35'
    cfg.vision_assets = str(ASSETS)
    cfg.text_assets = str(ASSETS)
    return cfg


class CQPredicateEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check(self, batch, query_positions, unconditional=None):
        mask, valid = document_cq_mask(batch, query_positions, unconditional)
        pred = cq_mask_vectors(batch, query_positions, valid, unconditional)
        rebuilt = pred.dense()
        self.assertEqual(rebuilt.shape, mask.shape, 'shape differs')
        self.assertTrue(torch.equal(rebuilt, mask),
                        f'predicate differs in {(rebuilt != mask).sum().item()} entries')
        return mask, valid

    def test_single_document(self):
        torch.manual_seed(3)
        s = 48
        batch = SimpleNamespace()
        batch.ids = torch.zeros(1, s, dtype=torch.long)
        batch.sigma = torch.arange(s)[None]
        batch.segments = torch.zeros(1, s, dtype=torch.long)
        qp = torch.tensor([5, 11, 29, 47])
        self.check(batch, qp)

    def test_multi_document_and_padding(self):
        torch.manual_seed(5)
        s = 64
        batch = SimpleNamespace()
        batch.ids = torch.zeros(1, s, dtype=torch.long)
        seg = torch.full((s,), -1, dtype=torch.long)
        seg[0:20] = 0
        seg[20:50] = 1
        # 50..64 stays -1: padding, must be fully forbidden + sentinel
        batch.segments = seg[None]
        sigma = torch.zeros(s, dtype=torch.long)
        sigma[0:20] = torch.arange(20)
        sigma[20:50] = torch.arange(30)
        batch.sigma = sigma[None]
        qp = torch.tensor([3, 19, 25, 49, 55])  # 55 is a padding row
        self.check(batch, qp)

    def test_tied_image_block_sigma(self):
        """Tied sigma inside an image group: content is bidirectional there."""
        s = 40
        batch = SimpleNamespace()
        batch.ids = torch.zeros(1, s, dtype=torch.long)
        batch.segments = torch.zeros(1, s, dtype=torch.long)
        sigma = torch.arange(s)
        sigma[10:20] = 10  # one image group shares a rank
        batch.sigma = sigma[None]
        qp = torch.tensor([5, 10, 15, 19, 25])
        mask, _ = self.check(batch, qp)
        # Inside the tied block, content rows see each other both ways.
        self.assertFalse(bool(mask[0, 0, 12, 17]))
        self.assertFalse(bool(mask[0, 0, 17, 12]))

    def test_unconditional_queries_read_nothing(self):
        s = 32
        batch = SimpleNamespace()
        batch.ids = torch.zeros(1, s, dtype=torch.long)
        batch.segments = torch.zeros(1, s, dtype=torch.long)
        batch.sigma = torch.arange(s)[None]
        qp = torch.tensor([4, 9, 21])
        unc = torch.tensor([False, True, False])
        mask, valid = self.check(batch, qp, unc)
        row = s + 1
        self.assertFalse(bool(valid[0, row]))
        # Only the sentinel column is open for an unconditional row.
        self.assertFalse(bool(mask[0, 0, row, 0]))
        self.assertTrue(bool(mask[0, 0, row, 1:].all()))

    def test_cutoffs_match_visible_keys_recurrent_writes_and_conv_history(self):
        from ops.gated_delta_layout import build_delta_plan
        rng = random.Random(315)
        for _ in range(32):
            docs, sigmas = [], []
            for doc in range(3):
                rank = rng.randrange(1, 10)
                for j in range(8):
                    rank += rng.choice([0,0,1,3])
                    docs.append(doc)
                    sigmas.append(rank)
            size = len(docs)
            positions = torch.tensor([rng.randrange(size) for _ in range(12)])
            cuts = torch.tensor([rng.randrange(sigmas[p]+1) for p in positions])
            null = torch.tensor([i%5==0 for i in range(len(positions))])
            batch = SimpleNamespace(ids=torch.zeros(1,size,dtype=torch.long),
                                    segments=torch.tensor([docs]),sigma=torch.tensor([sigmas]))
            plan = SimpleNamespace(generation_positions=torch.empty(0,dtype=torch.long),prefix=0,
                                   query_positions=positions,query_read_sigma=cuts,query_unconditional=null)
            kernel = rng.randrange(1,6)
            delta = build_delta_plan(batch,plan,SimpleNamespace(linear_conv_kernel=kernel))
            mask,valid = document_cq_mask(batch,positions,null,cuts)
            assert torch.equal(mask,cq_mask_vectors(batch,positions,valid,null,cuts).dense())
            for i,p in enumerate(positions.tolist()):
                visible = [k for k in range(size) if docs[k]==docs[p] and sigmas[k]<cuts[i] and not null[i]]
                event = int(delta.hidden_event[size+i])
                writes = delta.event_write[delta.event_bounds[docs[p]]:event]
                assert delta.content_order[writes[writes<size]].tolist() == visible
                row = (delta.query_source==size+i).nonzero().item()
                conv = delta.query_history[row]
                expected_conv = visible[-(kernel-1):] if kernel>1 else []
                assert delta.content_order[conv[conv<size]].tolist() == expected_conv
                allowed = ~mask[0,0,size+i] & valid[0,size+i]
                assert allowed.nonzero().flatten().tolist() == visible

    @unittest.skipUnless(ASSETS.is_dir(), 'local Qwen3.5-2B-Base assets required')
    def test_real_plan_layouts(self):
        """Layouts produced by the actual Wan plan builder."""
        for seed, length, images in ((11, 1024, 1), (12, 1536, 2), (13, 2048, 3)):
            cfg = wan_cfg(linear_heads=2, linear_head_dim=8)
            batch = fake_batch(cfg, length, 1, images, seed=seed)
            plan = batch.mainline
            self.check(batch, plan.query_positions, plan.query_unconditional)

    def test_randomized_layouts(self):
        g = torch.Generator().manual_seed(99)
        for _ in range(25):
            s = int(torch.randint(16, 96, (1,), generator=g))
            ndoc = int(torch.randint(1, 4, (1,), generator=g))
            batch = SimpleNamespace()
            batch.ids = torch.zeros(1, s, dtype=torch.long)
            seg = torch.full((s,), -1, dtype=torch.long)
            sigma = torch.zeros(s, dtype=torch.long)
            cursor = 0
            for d in range(ndoc):
                n = max(2, (s - cursor) // (ndoc - d))
                n = min(n, s - cursor)
                if n <= 0:
                    break
                seg[cursor:cursor + n] = d
                r = torch.arange(n)
                if bool(torch.randint(0, 2, (1,), generator=g)):  # tie a block
                    a = int(torch.randint(0, max(1, n - 1), (1,), generator=g))
                    r[a:min(n, a + 3)] = a
                sigma[cursor:cursor + n] = r
                cursor += n
            batch.segments = seg[None]
            batch.sigma = sigma[None]
            nq = int(torch.randint(1, 8, (1,), generator=g))
            qp = torch.randint(0, s, (nq,), generator=g)
            unc = torch.rand(nq, generator=g) < 0.3
            self.check(batch, qp, unc)


if __name__ == '__main__':
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Synthetic CPU tests only; no claim of real data/CUDA/Case026 qualification."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from model import HierarchicalVisibilityCDF
from evaluate import (FORMAT, FieldStats, HeadView, RankingStats, draw, freeze,
                      load_hierarchical, masked_union, output_gradients)


class EvalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)

    def test_head_adapters_and_q_gradients(self):
        m = freeze(HierarchicalVisibilityCDF(), torch.device('cpu'))
        x = torch.randn(6, 10)
        full, grad = output_gradients(m, x)
        u, gu = output_gradients(HeadView(m, 'union'), x)
        s, gs = output_gradients(HeadView(m, 'sensors'), x)
        torch.testing.assert_close(full[:, :1], u)
        torch.testing.assert_close(full[:, 1:], s)
        torch.testing.assert_close(grad[:, :1], gu)
        torch.testing.assert_close(grad[:, 1:], gs)
        self.assertEqual(tuple(gs.shape), (6, 8, 7))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.parameters()))

    def test_masked_max_matches_autograd(self):
        q = torch.randn(5, 7, requires_grad=True)
        pred = nn.Linear(7, 8)(q)
        mask = torch.ones_like(pred, dtype=torch.bool)
        mask[0] = False
        mask[1, pred[1].argmax()] = False
        grads = torch.stack([torch.autograd.grad(pred[:, s].sum(), q, retain_graph=True)[0]
                             for s in range(8)], dim=1)
        keep, value, grad = masked_union(pred, grads, mask)
        direct = pred[keep].masked_fill(~mask[keep], -torch.inf).max(dim=1).values
        dg = torch.autograd.grad(direct.sum(), q)[0][keep]
        torch.testing.assert_close(value, direct)
        torch.testing.assert_close(grad, dg)
        self.assertEqual(keep.tolist(), [False, True, True, True, True])

    def test_ranking_is_masked_and_fallback_not_visibility(self):
        p = torch.full((3, 8), 1000.)
        t = torch.zeros(3, 8)
        mask = torch.zeros(3, 8, dtype=torch.bool)
        mask[0, :3] = True
        p[0, :3] = torch.tensor([1., 3., 2.])
        t[0, :3] = torch.tensor([3., 2., 1.])
        mask[1, 4] = True
        p[1, 4] = 1.
        r = RankingStats()
        r.add(p, t, mask)
        result = r.result()
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['winner_top1_accuracy_or_recall'], .5)
        self.assertEqual(result['winner_top2_accuracy_or_recall'], .5)
        self.assertEqual(result['winner_top3_accuracy_or_recall'], 1.)
        self.assertEqual(result['fallback_count'], 1)
        self.assertEqual(result['fallback_accuracy_after_gt_winner_removed'], 1.)

    def test_field_metrics_and_chunks(self):
        p, t = torch.tensor([1., -1., 2.]), torch.tensor([0., -2., 2.])
        g = torch.eye(7)[:3]
        whole, chunks = FieldStats(), FieldStats()
        whole.add(p, g, t, g)
        for i in range(3):
            chunks.add(p[i:i+1], g[i:i+1], t[i:i+1], g[i:i+1])
        self.assertEqual(whole.result(), chunks.result())
        self.assertAlmostEqual(whole.result()['mae'], 2/3)
        self.assertEqual(whole.result()['gradient_cosine_mean'], 1.)
        self.assertIsNone(FieldStats().result()['mae'])

    def test_draws_do_not_depend_on_model_initialization(self):
        class Data:
            val_indices_cpu = torch.tensor([2, 3, 5, 8])
            x_cpu = torch.arange(30).reshape(10, 3).float()
            qlib_cpu = torch.zeros(10, 2, 7, 8)
            valid_cpu = torch.ones(10, 2, 8, dtype=torch.bool)
            def q_limits(self, device):
                return -torch.ones(7, device=device), torch.ones(7, device=device)
        a, b = {}, {}
        draw(Data(), 7, 11, np.random.default_rng(123), torch.device('cpu'), a, 'f')
        _ = HierarchicalVisibilityCDF()
        torch.rand(1000)
        draw(Data(), 7, 11, np.random.default_rng(123), torch.device('cpu'), b, 'f')
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(set(a['f_x_indices']).issubset({2, 3, 5, 8}))

    def test_completed_checkpoint_guard_and_serialization(self):
        m = HierarchicalVisibilityCDF()
        c = {'format': FORMAT, 'step': 50000, 'initialization': 'random_from_scratch', 'out_dim': 9,
             'args': {'seed': 0, 'val_count': 1000, 'horizontal_fov_deg': 50.,
                      'vertical_fov_deg': 66., 'z_min': .2, 'z_max': .7, 'delta': .01},
             'model_state': m.state_dict()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'final.pt'
            torch.save(c, path)
            restored, meta = load_hierarchical(str(path), torch.device('cpu'))
            inp = torch.randn(2, 10)
            torch.testing.assert_close(m(inp), restored(inp))
            self.assertEqual(meta['step'], 50000)
            c['step'] = 2
            torch.save(c, path)
            with self.assertRaises(ValueError):
                load_hierarchical(str(path), torch.device('cpu'))


if __name__ == '__main__':
    unittest.main(verbosity=2)

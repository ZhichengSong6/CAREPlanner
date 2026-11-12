"""CPU-only tests of Grid25 V4 pilot identity, sampling and frozen algorithm routing."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pilot import (
    prod, select_pilot_positions, upstream_module,
    save_json_exclusive, FORMAT, STAGES,
)


class FakeBank:
    def __init__(self):
        self.valid = np.ones((96, 4, 8), dtype=bool)
        # Expose both all-8 and partial-support x in each split.
        for i in (4, 6, 8, 50, 52, 54):
            self.valid[i, :, 0] = False


class PilotTests(unittest.TestCase):
    def test_select_new_grid_split_unchanged_and_reproducible(self):
        b = FakeBank()
        train = np.arange(0, 48, dtype=np.int64)
        val = np.arange(48, 96, dtype=np.int64)
        a = select_pilot_positions(b, train, val, ntrain=2, nval=2, seed=260927)
        c = select_pilot_positions(b, train, val, ntrain=2, nval=2, seed=260927)
        self.assertEqual(len(a), 4)
        for x, y in zip(a, c):
            self.assertEqual(x[0], y[0])
            self.assertEqual(x[1], y[1])
            np.testing.assert_array_equal(x[2], y[2])
        self.assertEqual([s for _, s, _ in a], [0, 0, 1, 1])
        self.assertTrue(all(x in train for x, s, _ in a if s == 0))
        self.assertTrue(all(x in val for x, s, _ in a if s == 1))
        self.assertEqual(len({x for x, _, _ in a}), 4)
        self.assertEqual(int((a[0][2] > 0).sum()), 8)
        self.assertEqual(int((a[2][2] > 0).sum()), 8)

    def test_selection_rejects_empty_split(self):
        bank = FakeBank()
        with self.assertRaisesRegex(RuntimeError, "selection sizes invalid"):
            select_pilot_positions(bank, [], np.arange(48, 96), ntrain=2,
                                   nval=2, seed=260927)

    def test_upstream_worker_implementation_is_reused(self):
        self.assertEqual(STAGES, ("global_pool", "v3_labels", "v4_tubes"))
        for stage in STAGES:
            mod = upstream_module(stage)
            self.assertEqual(mod.worker.__module__, stage)
            self.assertTrue(mod.__file__.endswith(
                f"hierarchical9_full_dataset_v4_v1/{stage}.py"))
        self.assertEqual(prod.Config(128, 4, 32, 1, 260927).v3_q_per_x, 4)
        self.assertEqual(FORMAT, "careplanner_grid25_v4_pilot_v1")

    def test_existing_full_v4_fps_anchor_selection(self):
        class Small:
            q = np.zeros((1, 4, 7, 8), dtype=np.float32)
            valid = np.ones((1, 4, 8), dtype=bool)
            masks = np.ones((8, 7), dtype=np.float32)
        bank = Small()
        for i in range(4):
            bank.q[0, i, 0, 0] = i * 0.1
        slots = prod.select_anchor_slots(bank, 0, 0, 2)
        self.assertEqual(slots.tolist(), [0, 3])

    def test_upstream_refine_anchor_exact_root(self):
        def geometry(q):
            h = np.asarray([q[0], 1, 1, 1, 1, 1], dtype=float)
            j = np.zeros((6, 7))
            j[0, 0] = 1.0
            return h, j
        q0 = np.zeros(7, dtype=float)
        q0[0] = 0.03
        mask = np.asarray([1, 0, 0, 0, 0, 0, 0], float)
        res = prod.refine_anchor(q0, mask, -np.ones(7), np.ones(7), geometry)
        self.assertTrue(res["success"])
        self.assertLess(abs(res["g"]), 1e-6)
        self.assertTrue(np.array_equal(np.asarray(res["q_star"])[1:], q0[1:]))

    def test_exclusive_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            save_json_exclusive(path, {"status": "FIRST"})
            with self.assertRaises(FileExistsError):
                save_json_exclusive(path, {"status": "SECOND"})
            self.assertEqual(json.loads(path.read_text()), {"status": "FIRST"})


if __name__ == "__main__":
    unittest.main()

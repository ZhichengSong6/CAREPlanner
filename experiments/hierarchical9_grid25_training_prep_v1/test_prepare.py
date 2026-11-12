"""CPU-only unit tests for fresh guards and spatial train/val disjointness."""
from __future__ import annotations
import unittest
import numpy as np
from prepare import fresh_guard, grid_buffer, split_masks


def lattice(shape, step=.025):
    ijk = np.stack(np.meshgrid(
        *[np.arange(n) for n in shape], indexing="ij"
    ), axis=-1).reshape(-1, 3)
    return np.asarray(ijk * step, dtype=np.float32)


class Grid25TrainingPrepTests(unittest.TestCase):
    def test_old_fresh_just_outside_excludes_near_grid(self):
        shape = (5, 5, 3)
        x = lattice(shape)
        outside_near = np.array([[.105, .01, .0]])
        guard = fresh_guard(x, shape, np.zeros(3), .025, outside_near, 25)
        self.assertGreater(int(guard.sum()), 0)
        far = np.array([[3., 0., 0.]])
        self.assertFalse(fresh_guard(x, shape, np.zeros(3), .025, far, 75).any())

    def test_grid_buffer_does_not_wrap_around(self):
        shape = (8, 8, 8)
        point = np.zeros(np.prod(shape), dtype=bool)
        point[0] = True
        buffered = grid_buffer(point, shape, 25, .025)
        self.assertTrue(buffered[0])
        self.assertTrue(buffered[np.ravel_multi_index((1, 0, 0), shape)])
        self.assertFalse(buffered[-1])
        self.assertFalse(buffered[np.ravel_multi_index((1, 1, 1), shape)])

    def test_grid_buffer_zero_radius_returns_original(self):
        shape = (5, 5, 5)
        a = np.zeros(np.prod(shape), dtype=bool)
        a[57] = True
        self.assertTrue(np.array_equal(grid_buffer(a, shape, 0, .025), a))

    def test_split_determinism_and_gap(self):
        shape = (30, 30, 18)
        x = lattice(shape)
        outside = np.array([[.73, .1, .1], [.01, .02, .02]])
        args = dict(fresh_mm=75, val_buffer_mm=75,
                    block_cells=8, val_fraction=.10, seed=20261112)
        a = split_masks(x, shape, np.zeros(3), .025, outside, **args)
        b = split_masks(x, shape, np.zeros(3), .025, outside, **args)
        for key in a:
            self.assertTrue(np.array_equal(a[key], b[key]), key)
        used = np.concatenate(list(a.values()))
        self.assertTrue(np.array_equal(np.sort(used), np.arange(len(x))))
        self.assertGreater(len(a["train"]), 0)
        self.assertGreater(len(a["val"]), 0)
        self.assertEqual(np.intersect1d(a["train"], a["fresh_guard"]).size, 0)
        self.assertEqual(np.intersect1d(a["train"], a["val_guard"]).size, 0)
        val = np.zeros(len(x), dtype=bool); val[a["val"]] = True
        self.assertFalse(grid_buffer(val, shape, 75, .025)[a["train"]].any())
        self.assertFalse(fresh_guard(x, shape, np.zeros(3), .025, outside, 75)[a["train"]].any())

    def test_invalid_fraction_rejected(self):
        shape = (5, 5, 5); x = lattice(shape)
        with self.assertRaisesRegex(RuntimeError, "Invalid split protocol"):
            split_masks(x, shape, np.zeros(3), .025, np.array([[0., 0., 0.]]),
                        val_fraction=0)


if __name__ == "__main__":
    unittest.main()

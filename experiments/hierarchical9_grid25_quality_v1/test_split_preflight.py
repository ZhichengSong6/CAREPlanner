"""CPU-only tests for Grid25 fresh holdout coordinate preflight."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from split_preflight import archive_metadata, load_fresh, spatial_coverage


class SplitPreflightTests(unittest.TestCase):
    def test_fresh_positions_not_inherited_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "starts.jsonl"
            rows = [
                {"x_index": 10, "x": [0.01, 0.015, 0.0], "group": "S0/in"},
                {"x_index": 10, "x": [0.01, 0.015, 0.0], "group": "S0/out"},
                {"x_index": 99, "x": [0.025, 0.01, 0.0], "group": "S1/out"},
            ]
            path.write_text("".join(json.dumps(x)+"\n" for x in rows))
            coords, num, old_idx, groups = load_fresh(path)
            self.assertEqual(coords.shape, (2, 3))
            self.assertEqual(num, 3)
            self.assertEqual(old_idx, 2)
            self.assertEqual(sum(groups.values()), 3)

    def test_missing_coordinate_is_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"starts.jsonl"
            path.write_text('{"x_index": 2}\n')
            with self.assertRaisesRegex(RuntimeError, "missing explicit"):
                load_fresh(path)

    def test_spatial_exclusion_uses_new_grid_coordinates(self):
        shape = (3, 3, 2)
        step = 0.025
        ijk = np.stack(np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
            indexing="ij"), axis=-1).reshape(-1, 3)
        x = (ijk*step).astype(np.float32)
        hold = np.asarray([[0.013, 0.015, 0.0]])
        report = spatial_coverage(x, shape, [0, 0, 0], step, hold, [25, 50])
        near = report["candidate_spatial_exclusions"]
        self.assertGreater(near["25mm"]["grid25_points_excluded"], 0)
        self.assertGreaterEqual(near["50mm"]["grid25_points_excluded"],
                                near["25mm"]["grid25_points_excluded"])
        self.assertLess(report["old_fresh_nearest_grid25_max_distance_mm"], 25)

    def test_grid_order_changed_is_rejected(self):
        shape = (2, 2, 2)
        coords = np.stack(np.meshgrid(np.arange(2), np.arange(2), np.arange(2),
                                      indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
        coords[[0, 7]] = coords[[7, 0]]
        with self.assertRaisesRegex(RuntimeError, "storage order"):
            spatial_coverage(coords, shape, [0, 0, 0], 1, [[0, 0, 0]], [25])

    def test_archive_metadata_no_bulk_decompression(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"bank.npz"
            np.savez_compressed(path,
                x=np.zeros((256, 3), np.float32),
                q=np.zeros((256, 4, 7, 8), np.float32),
                k=np.arange(256),
                valid_fov=np.ones((256, 4, 8), bool),
                sensor_chain_masks=np.ones((8, 7), np.float32),
                q_min=np.full((7,), -1, np.float32),
                q_max=np.full((7,), 1, np.float32))
            meta = archive_metadata(path)
            self.assertGreater(meta["uncompressed_npz_members_bytes"],
                               meta["compressed_npz_file_bytes"])
            self.assertGreater(meta["raw_q_member_bytes"], 0)


if __name__ == "__main__":
    unittest.main()

"""Small CPU-only tests for the read-only Grid25 audit helpers."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit import atomic_json, inspect_shard_values, sampled_ids


class QualityAuditTests(unittest.TestCase):
    def test_sampled_ids_are_deterministic_and_bounded(self):
        self.assertEqual(sampled_ids(8, 0), set())
        self.assertEqual(sampled_ids(8, 1), {0})
        self.assertEqual(sampled_ids(8, 2), {0, 7})
        self.assertEqual(len(sampled_ids(8, 9)), 8)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as temp:
            dst = Path(temp) / "analysis" / "report.json"
            atomic_json(dst, {"status": "PASS"})
            self.assertEqual(json.loads(dst.read_text()), {"status": "PASS"})
            atomic_json(dst, {"status": "UPDATED"})
            self.assertEqual(json.loads(dst.read_text()), {"status": "UPDATED"})

    def sample_archive(self, directory):
        n, k = 3, 2
        q = np.full((n, k, 7, 8), np.inf, dtype=np.float32)
        valid = np.zeros((n, k, 8), dtype=bool)
        q[0, 0, :, 0] = 0
        q[1, 1, :, 1] = 0
        valid[0, 0, 0] = valid[1, 1, 1] = True
        count = valid.sum(axis=1).astype(np.int32)
        g = np.full((n, k, 8), np.inf, dtype=np.float32)
        g[valid] = 0.001
        archive = Path(directory) / "sample.npz"
        np.savez_compressed(
            archive,
            q=q, valid_fov=valid, g=g,
            x=np.zeros((n, 3), dtype=np.float32),
            k=np.arange(3, dtype=np.int64),
            num_sensor_q0=count, valid_point_fov=valid.any(axis=(1, 2)),
        )
        job = {"cfg": {"q_min": [-2.0] * 7, "q_max": [2.0] * 7},
               "reference": {"epsilon": 0.003}}
        return archive, job, q, valid, g, count

    def test_valid_sample_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, job, *_ = self.sample_archive(temp)
            self.assertEqual(inspect_shard_values(archive, 0, np.arange(3), 2, job), 3)

    def test_sample_outside_epsilon_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, job, q, valid, g, count = self.sample_archive(temp)
            g[0, 0, 0] = 0.2
            np.savez_compressed(
                archive, q=q, valid_fov=valid, g=g, x=np.zeros((3, 3), np.float32),
                k=np.arange(3), num_sensor_q0=count,
                valid_point_fov=valid.any(axis=(1, 2)),
            )
            with self.assertRaisesRegex(RuntimeError, "epsilon"):
                inspect_shard_values(archive, 0, np.arange(3), 2, job)

    def test_sample_mask_inconsistent_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, job, q, valid, g, count = self.sample_archive(temp)
            valid[2, 0, 2] = True  # q is still +inf
            np.savez_compressed(
                archive, q=q, valid_fov=valid, g=g, x=np.zeros((3, 3), np.float32),
                k=np.arange(3), num_sensor_q0=count,
                valid_point_fov=valid.any(axis=(1, 2)),
            )
            with self.assertRaisesRegex(RuntimeError, "finite mask"):
                inspect_shard_values(archive, 0, np.arange(3), 2, job)


if __name__ == "__main__":
    unittest.main()

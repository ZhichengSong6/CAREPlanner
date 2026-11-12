"""CPU-only synthetic tests for Grid25 V4 full-production work budgeting."""
from __future__ import annotations

import math
import unittest

import numpy as np

from preflight import build_segments, build_shards, count_support, pilot_rates, project_cost


class Grid25ProductionPreflightTests(unittest.TestCase):
    def test_support_counts_are_exact_and_per_sensor(self):
        valid = np.zeros((6, 4, 8), dtype=bool)
        valid[0, :, 0] = True
        valid[0, :2, 5] = True
        valid[3, :1, 7] = True
        valid[5, :3, 2] = True
        ids = np.array([5, 0, 3], dtype=np.int64)
        counts = count_support(valid, ids, block=2)
        self.assertEqual(counts.shape, (3, 8))
        self.assertEqual(counts.dtype, np.uint16)
        self.assertEqual(counts[0, 2], 3)
        self.assertEqual(counts[1, 0], 4)
        self.assertEqual(counts[1, 5], 2)
        self.assertEqual(counts[2, 7], 1)
        self.assertEqual(int((counts>0).sum()), 4)

    def test_counts_reject_bad_grid_ids(self):
        valid = np.zeros((4, 10, 8), dtype=bool)
        with self.assertRaisesRegex(RuntimeError, "out of bank"):
            count_support(valid, [4], 8)
        with self.assertRaisesRegex(RuntimeError, "scan parameters"):
            count_support(valid, [0], 0)

    def test_shards_and_manual_segments_partition_exactly(self):
        counts = np.zeros((285, 8), dtype=np.uint16)
        counts[:, 0] = 500
        counts[:, 1] = 10
        counts[:200, 2] = 100
        shards = build_shards(285, counts, shard_x=128)
        self.assertEqual([x["x_count"] for x in shards], [128, 128, 29])
        self.assertEqual([x["v3_tasks"] for x in shards],
                         [128*3*4, (128*2+72)*4, 29*2*4])
        self.assertEqual(sum(x["v4_anchor_budget"] for x in shards),
                         int(np.minimum(counts, 32).sum()))
        segments = build_segments(shards, max_shards=2)
        self.assertEqual([x["x_count"] for x in segments], [256, 29])
        self.assertEqual(segments[0]["start"], 0)
        self.assertEqual(segments[-1]["stop"], 285)
        self.assertEqual(segments[0]["last_shard_exclusive"], 2)
        self.assertEqual(segments[1]["first_shard"], 2)

    def test_all_units_and_walltime_use_only_pilot_empirical_seconds(self):
        rates = dict(
            global_cpu_seconds_per_x=1.,
            v3_cpu_seconds_per_task=0.75,
            v4_cpu_seconds_per_anchor=0.01,
        )
        work = dict(x_count=100, v3_tasks=200, v4_anchor_budget=1000)
        cost = project_cost(work, rates, workers=4,
                            multiplier=2., overhead_minutes=10.)
        self.assertAlmostEqual(cost["estimated_worker_cpu_hours"],
                               260/3600, places=3)
        self.assertAlmostEqual(cost["conservative_wall_hours_at_workers"],
                               2*260/4/3600+10/60, places=3)

    def test_pilot_rates_validate_denominators(self):
        global_result = dict(worker_seconds=1.6)
        v3 = dict(tasks=16, valid_values=15, shard_worker_seconds=12.,
                  valid_fraction=.9375)
        v4 = dict(anchors=128, refined=126, shard_worker_seconds=5.,
                  refined_fraction=126/128, regular_fraction=.8)
        manifest = dict(counts=dict(selected_x=4))
        rates = pilot_rates(global_result, v3, v4, manifest)
        self.assertAlmostEqual(rates["v3_cpu_seconds_per_task"], .75)
        self.assertAlmostEqual(rates["v4_cpu_seconds_per_anchor"], 5/128)
        self.assertEqual(rates["pilot_positions"], 4)

    def test_project_cost_rejects_nonfinite_multiplier(self):
        rates = dict(global_cpu_seconds_per_x=1.,
                     v3_cpu_seconds_per_task=1.,
                     v4_cpu_seconds_per_anchor=1.)
        work = dict(x_count=1, v3_tasks=2, v4_anchor_budget=3)
        with self.assertRaisesRegex(RuntimeError, "resource-planning"):
            project_cost(work, rates, multiplier=float("nan"))
        with self.assertRaisesRegex(RuntimeError, "resource-planning"):
            project_cost(work, rates, workers=0)


if __name__ == "__main__":
    unittest.main()

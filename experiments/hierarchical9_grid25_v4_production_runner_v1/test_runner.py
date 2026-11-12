"""CPU-only synthetic tests for Grid25 V4 manual production segment gates."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import runner as r


class ProductionRunnerTests(unittest.TestCase):
    def test_slice_preserves_new_grid_indices_and_splits(self):
        plan = dict(
            x_index=np.arange(4000, 4300, dtype=np.int64),
            split=np.r_[np.zeros(200, np.uint8), np.ones(100, np.uint8)],
            valid_count=np.ones((300, 8), dtype=np.uint16)*10,
            support=np.ones((300, 8), dtype=bool),
            irrelevant=np.arange(300),
        )
        s = r.segment_slice(plan, dict(start=128, stop=256))
        self.assertEqual(list(s), ["x_index", "split", "valid_count", "support"])
        self.assertEqual(len(s["x_index"]), 128)
        self.assertEqual(s["x_index"][0], 4128)
        self.assertEqual(int(s["split"].sum()), 56)
        self.assertTrue(np.array_equal(s["support"], s["valid_count"] > 0))

    def test_frozen_stage_routing_reuses_original_algorithms(self):
        for stage in r.STAGES:
            module = r.original_module(stage)
            self.assertEqual(module.worker.__module__, stage)
            self.assertIn("hierarchical9_full_dataset_v4_v1", str(module.__file__))
            self.assertIs(module.open_run, r.open_segment)

    def test_prepares_one_new_segment_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "grid25"
            pre = root / "derived" / "grid25_v4_production_preflight_v1"
            pre.mkdir(parents=True)
            (pre / "report.json").write_text('{"status":"verified"}\n')
            idx = np.arange(10_000, 10_128, dtype=np.int64)
            counts = np.full((128, 8), 500, dtype=np.uint16)
            plan = dict(x_index=idx, split=np.zeros(128, np.uint8),
                        valid_count=counts, support=counts > 0)
            segment = dict(segment_id=0, start=0, stop=128, x_count=128,
                           supported_x_sensor_pairs=128*8,
                           v3_tasks=128*8*4,
                           v4_anchor_budget=128*8*32)
            report = dict(
                frozen_label_config=asdict(r.prod.Config(
                    128, 4, 32, 128, 260927)),
                budget_assumptions=dict(workers=4),
                provenance=dict(
                    bank_manifest_sha256="a"*64,
                    original_npz_sha256="b"*64),
                trainprep=str(root / "derived/grid25_trainprep_v1"),
                x_plan_sha256="c"*64,
                segments_sha256="d"*64,
            )
            with mock.patch.object(r, "frozen_source",
                                   return_value=(pre, report, plan, [segment])):
                out = r.prepare_one(root, 0)
                self.assertTrue((out / "manifest.json").is_file())
                self.assertTrue((out / "x_plan.npz").is_file())
                with np.load(out / "x_plan.npz", allow_pickle=False) as z:
                    self.assertEqual(z["x_index"][0], 10000)
                    self.assertEqual(int(z["support"].sum()), 1024)
                m = r.read(out / "manifest.json")
                self.assertFalse(m["training_ready"])
                self.assertEqual(m["counts"]["v3_tasks"], 4096)
                with self.assertRaisesRegex(RuntimeError, "already prepared"):
                    r.prepare_one(root, 0)

    def test_segment_id_boundaries(self):
        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            r.segment_out(Path("/tmp/ignore"), -1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pre = root / "derived" / "grid25_v4_production_preflight_v1"
            with mock.patch.object(r, "frozen_source",
                                   return_value=(pre, {}, {}, [])):
                with self.assertRaisesRegex(RuntimeError, "out of planned range"):
                    r.prepare_one(root, 0)

    def test_global_pool_row_attribution_and_sha(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            folder = out / "global_pool"
            folder.mkdir()
            idx = np.asarray([11, 33], dtype=np.int64)
            split = np.asarray([0, 1], dtype=np.uint8)
            support = np.ones((2, 8), dtype=bool)
            file = folder / "shard_0000.npz"
            np.savez_compressed(file, x_index=idx, split=split, support=support)
            (folder / "shard_0000.json").write_text(
                json.dumps({"sha256": r.sha(file)}) + "\n")
            # Upstream global_pool.merge always records the number of shards.
            # Verify both the fail-closed mismatch and the valid fixture.
            summary = out / "global_pool_summary.json"
            summary.write_text('{"status":"COMPLETE","shards":0}\n')
            m = dict(config=asdict(r.prod.Config(
                128, 4, 32, 128, 260927)))
            plan = dict(x_index=idx, split=split,
                        support=support,
                        valid_count=np.full((2, 8), 500, dtype=np.uint16))
            with mock.patch.object(r, "open_segment",
                                   return_value=(out, m, None, plan)):
                with self.assertRaisesRegex(RuntimeError, "Wrong global shard count"):
                    r.verify_one_stage(out, "global_pool")
                summary.write_text('{"status":"COMPLETE","shards":1}\n')
                result = r.verify_one_stage(out, "global_pool")
                self.assertEqual(result["status"], "COMPLETE")
                np.savez_compressed(file, x_index=idx, split=split,
                                    support=np.zeros_like(support))
                with self.assertRaisesRegex(RuntimeError, "bad checksum"):
                    r.verify_one_stage(out, "global_pool")

    def test_submission_files_do_not_request_gpu_or_automatic_next(self):
        script = (r.HERE / "submit.sh").read_text()
        self.assertIn("--cpus-per-task=4", script)
        self.assertNotIn("--gres", script)
        self.assertNotIn("--gpus", script)
        self.assertNotIn("--dependency", script)
        self.assertIn("squeue -h -u", script)
        self.assertIn("sbatch --test-only", script)


if __name__ == "__main__":
    unittest.main()

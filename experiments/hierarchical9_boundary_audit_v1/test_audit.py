#!/usr/bin/env python3
"""CPU numerical tests; --require-repo also requires real FK/runtime import tests."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core import (Config, SensorField, bank_label, cosine, distribution, field_metrics,
                  json_safe, qualify_boundary, rate, refine_boundary, tangent_seed, within)
from audit import Results, boundary_record, normal_profile, markdown
from runtime_probe import run_probe, analytic_ascent

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HAS_REPO = (REPO / "src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py").is_file()
REQUIRE_REPO = "--require-repo" in sys.argv
if REQUIRE_REPO: sys.argv.remove("--require-repo")


class PlaneOracle:
    """Interior q1>0; other planes inactive. All tensors retain autograd."""
    def planes(self, x, q, s):
        return torch.cat([q[:, :1], q[:, :1]*0 + torch.tensor([1., 2., 3., 4., 5.], device=q.device)], -1)
    def value(self, x, q, s): return float(self.planes(x, q.reshape(1, 7), s).min())
    def value_grad(self, x, q, s):
        v = q.detach().clone().requires_grad_(True)
        g = self.planes(x, v.reshape(1, 7), s).min(dim=-1).values.sum()
        return float(g.detach()), torch.autograd.grad(g, v)[0].detach()


class LinearSensors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Linear(10, 8, bias=False)
        with torch.no_grad():
            self.layers.weight.zero_()
            self.layers.weight[:, 3] = torch.arange(1., 9.)
    def forward(self, u): return self.layers(u)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg, self.oracle = Config(), PlaneOracle()
        self.x, self.q = torch.zeros(3), torch.zeros(7)
        self.mask = torch.tensor([1., 1., 0., 0., 0., 0., 0.])
        self.lo, self.hi = -torch.ones(7)*2, torch.ones(7)*2

    def test_discrete_label_jump_and_bad_normal(self):
        bank = torch.zeros(1, 7)
        q = torch.tensor([1e-4, .5, 0, 0, 0, 0, 0.])
        a = bank_label(q, bank, self.mask, 1)
        q[0] *= -1
        b = bank_label(q, bank, self.mask, -1)
        self.assertGreater(a["legacy_signed_value"] - b["legacy_signed_value"], .99)
        self.assertLess(cosine(a["gradient"], torch.tensor([1., 0, 0, 0, 0, 0, 0])), .001)

    def test_exact_bank_floor_has_no_normal(self):
        a = bank_label(self.q, self.q[None], self.mask, 1.)
        self.assertEqual(a["distance_rad"], 0.)
        self.assertAlmostEqual(a["legacy_signed_value"], 1e-4, places=8)
        self.assertEqual(float(a["gradient"].norm()), 0.)
        self.assertFalse(a["gradient_valid"])

    def test_bank_ignores_inactive_dofs(self):
        q = self.q.clone(); q[6] = 10
        self.assertEqual(bank_label(q, self.q[None], self.mask, 1.)["distance_rad"], 0.)

    def test_bank_first_index_tie_and_invalid_bank(self):
        bank = self.q.repeat(2, 1); bank[:, 0] = torch.tensor([-1., 1.])
        r = bank_label(self.q, bank, self.mask, 1.)
        self.assertEqual(r["nearest_slot_in_valid_bank"], 0)
        self.assertEqual(r["nearest_gap_rad"], 0.)
        self.assertFalse(r["gradient_valid"])
        with self.assertRaises(ValueError): bank_label(self.q, bank*float("nan"), self.mask, 1)

    def test_newton_root_and_inactive_invariance(self):
        q = self.q.clone(); q[0] = -.3; q[6] = .7
        r = refine_boundary(lambda q: self.oracle.value_grad(self.x, q, 0), q,
                            self.mask, self.lo, self.hi, self.cfg)
        self.assertTrue(r["accepted"])
        self.assertLess(abs(r["g"]), self.cfg.root_tol_m)
        self.assertAlmostEqual(float(r["q"][6]), .7)

    def test_refinement_degenerate_and_infeasible(self):
        r = refine_boundary(lambda q: (.1, torch.zeros(7)), self.q,
                            self.mask, self.lo, self.hi, self.cfg)
        self.assertEqual(r["reason"], "degenerate_geometry_gradient")
        r = refine_boundary(lambda q: (1., torch.ones(7)), self.q+3,
                            self.mask, self.lo, self.hi, self.cfg)
        self.assertEqual(r["reason"], "seed_out_of_bounds")

    def test_regular_boundary_fd(self):
        r = qualify_boundary(self.oracle, self.x, self.q, 0, self.mask, self.lo, self.hi, self.cfg)
        self.assertTrue(r["regular"])
        self.assertAlmostEqual(r["fd_cosine"], 1., places=5)

    def test_plane_tie_and_limit_rejection(self):
        class Tie(PlaneOracle):
            def planes(self, x, q, s):
                return torch.cat([q[:, :1], q[:, 1:2], torch.ones(len(q), 4)], -1)
        r = qualify_boundary(Tie(), self.x, self.q, 0, self.mask, self.lo, self.hi, self.cfg)
        self.assertIn("fov_plane_tie", r["reasons"])
        q = self.q.clone(); q[1] = self.hi[1]
        r = qualify_boundary(self.oracle, self.x, q, 0, self.mask, self.lo, self.hi, self.cfg)
        self.assertIn("near_joint_limit", r["reasons"])

    def test_nonfinite_geometry_rejected(self):
        class Bad(PlaneOracle):
            def value_grad(self, x, q, s): return float("nan"), torch.zeros(7)
        with self.assertRaises(RuntimeError):
            qualify_boundary(Bad(), self.x, self.q, 0, self.mask, self.lo, self.hi, self.cfg)

    def test_tangent_nonbank_distance(self):
        n = self.q.clone(); n[0] = 1
        q = tangent_seed(self.q, n, self.mask, np.random.default_rng(3), .1)
        self.assertEqual(float(q[0]), 0.)
        self.assertAlmostEqual(float(q.norm()), .1, places=6)
        self.assertGreater(bank_label(q, self.q[None], self.mask, 1)["distance_rad"], .09)
        self.assertFalse(bool(q[2:].any()))

    def test_sensor_mapping_and_raw_gradient(self):
        f = SensorField(LinearSensors())
        for s in range(8):
            q = self.q.clone(); q[0] = .1
            value, grad = f.value_grad(self.x, q, s)
            self.assertAlmostEqual(value, .1*(s+1), places=6)
            self.assertEqual(float(grad[0]), s+1)
        metrics = field_metrics(0., torch.ones(7), self.mask)
        self.assertGreater(metrics["inactive_gradient_norm"], 2.)

    def test_profiles_dont_clamp_or_zero_other_heads(self):
        fields = {"h9": SensorField(LinearSensors()), "old8": SensorField(LinearSensors())}
        n = self.q.clone(); n[0] = 1
        rows = normal_profile(self.oracle, fields, self.x, self.q, n, 0, self.q[None],
                              self.mask, self.lo, self.hi, self.cfg, {"origin": "test"})
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["expected_sign_ok"] for r in rows))
        self.assertTrue(all(r["models"]["h9"]["sign_correct"] for r in rows))
        lo, hi = self.lo.clone(), self.hi.clone(); hi[0] = .001
        bad = normal_profile(self.oracle, fields, self.x, self.q, n, 0, self.q[None],
                             self.mask, lo, hi, self.cfg, {})
        self.assertEqual(sum(not r["in_limits"] for r in bad), 4)

    def test_offbank_exclusion_and_boundary_records(self):
        fields = {"h9": SensorField(LinearSensors()), "old8": SensorField(LinearSensors())}
        r = boundary_record(self.oracle, fields, self.x, self.q, 0, self.q[None], self.mask,
                            self.lo, self.hi, self.cfg, {"origin": "offbank_refined"})
        self.assertFalse(r["regular"])
        self.assertIn("offbank_too_close_to_original_bank", r["reasons"])

    def test_empty_summary_is_missing_not_zero(self):
        r = Results(); r.boundaries["S7/offbank_refined"]; r.solves["S7/uniform_outside"]
        s = r.summary()
        self.assertIsNone(s["boundary"]["S7/offbank_refined"]["models"]["h9"]["abs_value"]["mean"])
        self.assertIsNone(s["solves"]["S7/uniform_outside"]["models"]["h9"]["fov_pass"]["rate"])
        self.assertIn("N/A", markdown(s))
        self.assertEqual(distribution([None, float("nan")])["count"], 0)
        self.assertIsNone(rate(0, 0)["rate"])
        self.assertNotIn("NaN", json.dumps(json_safe({"a": float("nan")}), allow_nan=False))

    def test_qzero_key_is_not_root_success(self):
        class FakeProbe:
            value_calls = gradient_calls = 0
            q_min, q_max = -torch.ones(7), torch.ones(7)
            def _optimize_branch(self, x, q, s):
                return {"q_zero": q[0].tolist(), "q_candidate": q[0].tolist(), "f_zero": -.4,
                        "root_source": "branch_root_not_found", "solution_mode": "branch_best_effort_ascent",
                        "projection_history": [], "ascent_history": []}
        q = self.q.clone(); q[0] = -.1
        r = run_probe(FakeProbe(), self.oracle, self.x, q, 0)
        self.assertFalse(r["predicted_bracket_found"])
        self.assertFalse(r["predicted_root_within_002"])
        self.assertFalse(r["fov_pass"])
        self.assertEqual(r["actual_seen"], "NOT_RUN")

    def test_analytic_control_is_bounded(self):
        q = self.q.clone(); q[0] = -.1
        r = analytic_ascent(self.oracle, self.x, q, 0, self.mask, self.lo, self.hi, steps=4)
        self.assertEqual(r["gradient_calls"], 4)
        self.assertTrue(r["fov_pass"])
        self.assertIn("NOT infeasibility", r["meaning"])

    def test_model_calls_do_not_consume_sampling_rng(self):
        a, b = np.random.default_rng(123), np.random.default_rng(123)
        field = SensorField(LinearSensors()); field.value_grad(self.x, self.q, 7)
        self.assertTrue(np.array_equal(a.normal(size=100), b.normal(size=100)))

    def test_invalid_config(self):
        with self.assertRaises(ValueError): replace(self.cfg, root_tol_m=-1).validate()
        with self.assertRaises(ValueError): replace(self.cfg, joint_margin_rad=1e-5).validate()


    def test_cli_pipeline_with_synthetic_dependencies(self):
        # Exercise all file/report/sample paths without claiming real data/FK.
        import audit
        from contextlib import redirect_stdout
        import io
        class Dataset:
            def __init__(self, *a, **kw):
                self.val_indices_cpu = torch.arange(1000)
                self.S, self.J = 8, 7
                self.x_cpu = torch.zeros(1000, 3)
                self.valid_cpu = torch.ones(1000, 2, 8, dtype=torch.bool)
                self.qlib_cpu = torch.zeros(1000, 2, 7, 8)
                self.qlib_cpu[:, 1, 1, :] = .5
            def q_limits(self, device): return (-torch.ones(7)*2).to(device), (torch.ones(7)*2).to(device)
            def sensor_masks(self, device): return torch.ones(8, 7, device=device)
        class FakeProbe:
            def __init__(self, model, masks, lo, hi):
                self.q_min, self.q_max, self.model = lo, hi, model
                self.value_calls = self.gradient_calls = 0
            def _optimize_branch(self, x, q, s):
                qz = q[0].clone(); qz[0] = 0.
                qc = qz.clone(); qc[0] = .05
                return {"q_zero": qz.tolist(), "q_candidate": qc.tolist(), "f_zero": 0.,
                    "root_source": "branch_sign_crossing_bisection", "solution_mode": "branch_projection_root_ascent",
                    "projection_history": [], "root_history": [], "ascent_history": []}
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for d in ("h9", "old8"):
                (tmp/d).mkdir(); (tmp/d/"final.pt").write_bytes(b"synthetic")
            (tmp/"data.npz").write_bytes(b"x")
            meta = {"step": 50000, "training_metadata": {"urdf_sha256": "synthetic", "data_bytes": 1}}
            fake_ev = types.ModuleType("evaluate")
            fake_ev.load_hierarchical = lambda *a: (LinearSensors(), meta)
            fake_ev.load_legacy = lambda *a: (LinearSensors(), meta)
            fake_ev.HeadView = lambda m, mode: m
            fake_train = types.ModuleType("train_signed_visibility_cdf_pairwise_replace")
            fake_train.VisibilityQ0Dataset = Dataset
            fake_train.DEFAULT_JOINT_NAMES = [str(i) for i in range(7)]
            fake_train.DEFAULT_SENSOR_FRAMES = [str(i) for i in range(8)]
            def fake_sha(p):
                if Path(p) == tmp/"h9/final.pt": return audit.V1
                if Path(p) == tmp/"old8/final.pt": return audit.OLD8
                return "synthetic"
            argv = ["audit.py", "--artifact-root", str(tmp), "--h9", "h9/final.pt",
                    "--old8", "old8/final.pt", "--data", "data.npz", "--output-dir", str(tmp/"out"),
                    "--device", "cpu", "--points-per-sensor", "1", "--anchors-per-point", "1",
                    "--random-starts-per-point", "1", "--analytic-control"]
            with patch.dict(sys.modules, {"evaluate": fake_ev,
                    "train_signed_visibility_cdf_pairwise_replace": fake_train}), \
                 patch.object(audit, "sha", fake_sha), patch.object(audit, "SensorOracle", lambda *a: PlaneOracle()), \
                 patch.object(audit, "preflight", lambda *a: {"status": "SYNTHETIC_ONLY"}), \
                 patch.object(audit, "make_probe", FakeProbe), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                audit.main()
            report = json.loads((tmp/"out/report.json").read_text())
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["boundary"]["S7/offbank_refined"]["regular"], 1)
            self.assertEqual(report["solves"]["S7/local_offbank_refined_r0.05"]["paired_starts"], 1)
            self.assertGreater((tmp/"out/generation.jsonl").stat().st_size, 0)
            self.assertGreater((tmp/"out/solves.jsonl").stat().st_size, 0)
            self.assertEqual(json.loads((tmp/"out/manifest.json").read_text())["status"], "COMPLETE")

    @unittest.skipUnless(HAS_REPO, "Real repository FK/URDF/runtime files not present in this local test harness")
    def test_real_repository_geometry_and_runtime(self):
        sys.path.insert(0, str(REPO/"src/care_visibility_cdf/scripts"))
        from train_signed_visibility_cdf_pairwise_replace import DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES
        from oracle import SensorOracle
        from runtime_probe import make_probe
        from audit import verify_runtime_parity
        oracle = SensorOracle(REPO/"src/arm_description/urdf/Arm.urdf", torch.device("cpu"),
                              DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES)
        qs = torch.from_numpy(np.random.default_rng(4).uniform(-.4, .4, (12, 7)).astype(np.float32))
        self.assertEqual(oracle.verify_against_upstream(torch.tensor([.1, .05, .15]), qs)["status"], "PASS")
        masks = torch.tensor([[1.,1.,0.,0.,0.,0.,0.]]*2 + [[1.,1.,1.,0.,0.,0.,0.]]*2 +
                             [[1.,1.,1.,1.,0.,0.,0.]]*2 + [[1.]*7]*2)
        probe = make_probe(LinearSensors().eval(), masks, self.lo, self.hi)
        r = verify_runtime_parity({"toy": probe}, self.x, -torch.ones(7)*.1)
        self.assertEqual(r["status"], "PASS")


if __name__ == "__main__":
    if REQUIRE_REPO and not HAS_REPO:
        raise SystemExit("--require-repo: real repository dependencies are missing; not a successful smoke")
    unittest.main(verbosity=2)

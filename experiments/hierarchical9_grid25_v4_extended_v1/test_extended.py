"""CPU-only fixed-stratum and verified-core FD tests for Grid25 extended audit."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest

import numpy as np

from gradient_audit import (
    FORMAT, STRATA, atomic_json, deterministic_strata, prod,
)


class ExtendedPilotTests(unittest.TestCase):
    def test_fixed_strata_independent_of_row_order(self):
        rows = []
        for split in (0, 1):
            for sensor in (0, 2):
                for sign in (-1, 1):
                    rows.extend([
                        (0, 100 + split, split, sensor, 0, 7, sign * .01),
                        (1, 200 + split, split, sensor, 1, 8, sign * .02),
                    ])
        a, missing = deterministic_strata(rows)
        b, reverse_missing = deterministic_strata(rows[::-1])
        self.assertEqual(a, b)
        self.assertEqual(missing, reverse_missing)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(missing), len(STRATA)-8)
        self.assertTrue(all(x["query_fov_sign"] in (-1, 1) for x in a))
        self.assertEqual([x["task_id"] for x in a], list(range(8)))
        self.assertEqual(FORMAT, "careplanner_grid25_v4_extended_fd_audit_v1")

    def test_sign_guard_records_missing_strata_no_replacement(self):
        rows = [(0, 17, 0, 2, 0, 3, -0.01),
                (0, 17, 0, 2, 1, 4, 0.00001)]
        chosen, missing = deterministic_strata(rows, sign_guard_m=0.0001)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["query_fov_sign"], -1)
        self.assertIn(dict(split=0, sensor=2, sign=1), missing)

    def test_bad_fov_margin_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Malformed pool"):
            deterministic_strata([(0, 1, 0, 0, 0, 7, float("nan"))])

    def test_original_v2_two_scale_reoptimization_fd_pass(self):
        core = prod.core
        q = np.zeros(7, dtype=np.float64)
        q[0] = 0.3
        mask = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
        grad = np.zeros(7, dtype=float)
        grad[0] = 1
        base = dict(
            gradient_candidate=grad,
            gradient_reasons=["DISTANCE_DERIVATIVE_NOT_VERIFIED"],
            distance_rad=.3, value=.3, value_valid=True, q_star=np.zeros(7),
            ambiguity=False, uncertain_competitor=False, selected={},
        )
        calls = []
        def probe(trial):
            calls.append(trial.copy())
            return dict(value=float(trial[0]), value_valid=True,
                        ambiguity=False, uncertain_competitor=False,
                        q_star=np.zeros(7), selected={})
        got = core.verify_gradient(base, q, mask, -np.ones(7), np.ones(7), probe)
        self.assertTrue(got["grad_valid"])
        self.assertEqual(got["verification"]["status"], "PASS")
        self.assertEqual(got["verification"]["expected_probes"], 4)
        self.assertEqual(len(calls), 4)

    def test_original_v2_gradient_mismatch_is_hold(self):
        core = prod.core
        q = np.zeros(7)
        q[0] = 0.3
        grad = np.zeros(7); grad[0] = -1
        base = dict(
            gradient_candidate=grad,
            gradient_reasons=["DISTANCE_DERIVATIVE_NOT_VERIFIED"],
            distance_rad=.3, value=.3, value_valid=True,
            q_star=np.zeros(7),
        )
        def probe(trial):
            return dict(value=float(trial[0]), value_valid=True,
                        ambiguity=False, uncertain_competitor=False,
                        q_star=np.zeros(7), selected={})
        got = core.verify_gradient(base, q, np.array([1, 0, 0, 0, 0, 0, 0]),
                                   -np.ones(7), np.ones(7), probe)
        self.assertFalse(got["grad_valid"])
        self.assertEqual(got["verification"]["status"], "HOLD")
        self.assertIn("DISTANCE_FD_MISMATCH", got["gradient_reasons"])

    def test_exclusive_audit_record_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fd_records" / "00.json"
            atomic_json(path, dict(fd_status="PASS"))
            with self.assertRaisesRegex(RuntimeError, "No overwrite"):
                atomic_json(path, dict(fd_status="HOLD"))
            self.assertIn("PASS", path.read_text())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Receipt-age counterexample and independent fixed-lease policy replays."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from phase_e_stopping_contract import qualify

ROOT = Path(__file__).resolve().parents[1]
ACCELERATION = [3., 3., 4., 4., 6., 6., 6.]


def budget(**overrides):
    args = dict(now_s=2.04, measured_s=2., applied=[1.5, 0., 0., 0., 0., 0., 0.],
                acceleration=ACCELERATION, velocity_limits=[2.] * 7,
                next_decision_s=.01, transport_bound_s=0., update_gap_s=.001,
                lease_end_s=2.08, lease_enforced=True, physical_error_rad=[0.] * 7,
                physical_settle_s=0., snapshot=None)
    args.update(overrides)
    return qualify(**args)


class ReceiptAgeTests(unittest.TestCase):
    def test_recorded_cpp_counterexample_is_enclosed(self):
        b = budget()['budgets']['withdraw_now']
        self.assertGreaterEqual(b['command_travel_rad'][0], .631473)
        self.assertGreaterEqual(b['command_stop_s'], .618)

    def test_one_physical_interval_has_one_absolute_enclosure(self):
        fresh = budget(now_s=2.)['budgets']
        delayed = budget()['budgets']
        for mode in fresh:
            np.testing.assert_allclose(fresh[mode]['command_travel_rad'], delayed[mode]['command_travel_rad'])
            self.assertAlmostEqual(fresh[mode]['required_until_s'], delayed[mode]['required_until_s'])
            self.assertAlmostEqual(fresh[mode]['command_stop_s'] - delayed[mode]['command_stop_s'], .04)

    def test_later_lease_adds_travel_and_physical_settling_extends_absolute_horizon(self):
        current = budget()['budgets']['withdraw_now']
        later = budget(lease_end_s=2.09, physical_error_rad=[.002] * 7,
                       physical_settle_s=.03)['budgets']['withdraw_now']
        self.assertTrue(np.all(np.asarray(later['command_travel_rad']) > current['command_travel_rad']))
        np.testing.assert_allclose(np.asarray(later['physical_travel_rad']) - later['command_travel_rad'], .002)
        self.assertAlmostEqual(later['physical_stop_s'] - later['command_stop_s'], .03)
        self.assertAlmostEqual(later['required_until_s'], 2.04 + later['physical_stop_s'])

    def test_expired_future_and_stale_receipts_have_no_budget(self):
        for args in [dict(lease_end_s=2.039), dict(measured_s=2.041), dict(measured_s=1.98)]:
            with self.subTest(args=args):
                result = budget(**args)
                self.assertFalse(result['qualified'])
                self.assertEqual(result['budgets'], {})

    def test_coverage_must_last_to_delayed_state_corrected_stop(self):
        snapshot = dict(complete=True, stable_version=True, enclosure_sha256='a' * 64,
                        bound_modes=['withdraw_now', 'continue_then_stop'],
                        inside_map=True, all_centers_free=True, valid_until_s=2.65)
        result = budget(snapshot=snapshot)
        self.assertFalse(result['qualified'])
        self.assertIn('evidence_expires_before_stop', result['reasons'])
        self.assertFalse(result['runtime_admission'])

    def test_command_only_map_diagnostic_cannot_qualify_physical_stop(self):
        snapshot=dict(complete=True,stable_version=True,enclosure_sha256='a'*64,
            bound_modes=['withdraw_now','continue_then_stop'],inside_map=True,
            all_centers_free=True,valid_until_s=20.,enclosure_kind='command_only')
        result=budget(snapshot=snapshot)
        self.assertFalse(result['qualified'])
        self.assertIn('command_only_snapshot_not_physical_evidence',result['reasons'])

    def test_independent_production_policy_matrix(self):
        self.run_policy_matrix(False)

    def test_independent_controller_enforced_target_range_matrix(self):
        self.run_policy_matrix(True)

    def run_policy_matrix(self,bounded):
        temporary = None
        if os.environ.get('CARE_RECEIPT_REPLAY_DIR'):
            out = Path(os.environ['CARE_RECEIPT_REPLAY_DIR']+('_bounded' if bounded else ''))
            out.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory(prefix='care_receipt_matrix_', dir='/tmp')
            self.addCleanup(temporary.cleanup)
            out = Path(temporary.name)
        binary = out / 'policy_replay'
        subprocess.run(['g++', '-std=c++14', '-O2', '-I' + str(ROOT / 'src/care_velocity_controller/include'),
                        str(ROOT / 'scripts/simulate_phase_e_receipt_age.cpp'), '-o', str(binary)], check=True)
        raw = subprocess.check_output([str(binary)]+(['--bounded-targets'] if bounded else []), text=True)
        (out / 'controller_replays.jsonl').write_text(raw)
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(rows), 2160)
        max_ratio = 0.
        for row in rows:
            result = budget(now_s=row['now_ns'] * 1e-9, measured_s=row['measured_ns'] * 1e-9,
                            applied=row['applied'], lease_end_s=row['lease_end_ns'] * 1e-9,
                            update_gap_s=row['gap_ns'] * 1e-9, max_braking_step_s=row['max_step_s'])
            if bounded:
                result=budget(now_s=row['now_ns']*1e-9,measured_s=row['measured_ns']*1e-9,
                    applied=row['applied'],lease_end_s=row['lease_end_ns']*1e-9,
                    update_gap_s=row['gap_ns']*1e-9,max_braking_step_s=row['max_step_s'],
                    authorized_target_abs=row['target_abs_velocity'])
            mode = 'continue_then_stop' if row['extra_ns'] else 'withdraw_now'
            self.assertIn(mode, result['budgets'], row)
            bound = result['budgets'][mode]
            travel = np.asarray(row['absolute_command_travel'])
            self.assertTrue(np.all(travel <= np.asarray(bound['command_travel_rad']) + 1e-9), row)
            self.assertLessEqual(row['stopped_ns'] * 1e-9, bound['required_until_s'] + 1e-9, row)
            denominator=np.asarray(bound['command_travel_rad'])
            ratio=np.divide(travel,denominator,out=np.zeros(7),where=denominator>0)
            max_ratio = max(max_ratio, float(np.max(ratio)))
        (out / 'result.json').write_text(json.dumps(dict(status='PASS', scenarios=len(rows),
            joint_scenarios=7 * len(rows), maximum_travel_to_budget_ratio=max_ratio,
            target_range_enforced=bounded,
            scope='conditional command-domain policy replays; no physical or map qualification',
            runtime_admission=False), indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()

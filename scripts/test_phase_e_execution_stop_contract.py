#!/usr/bin/env python3
"""Recorded motion counterexample and production receiver/regime regressions."""
import copy
import csv
import io
import json
import math
from pathlib import Path
import sys
import types
import unittest
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/egocentric_arm_planner/scripts'))
from execution_motion_limits import effective_limits, violation
from phase_e_stopping_contract import bounded_update_travel, qualify


def recorded_motion():
    rows = json.loads((ROOT/'outputs/phase_e_soft_mount_targeted_20261019/results_partial.json').read_text())
    case = next(x for x in rows if x['tag'] == 'r2_010')
    with zipfile.ZipFile(ROOT/case['archive']) as z:
        row = next(r for r in csv.reader(io.StringIO(z.read('run/committed_trajectory.csv').decode()))
                   if len(r) > 11 and r[2] == '24463000000')
    points = []
    for i in range(11, len(row), 22):
        ns = int(row[i+21])
        points.append(types.SimpleNamespace(positions=list(map(float, row[i:i+7])),
            velocities=list(map(float, row[i+7:i+14])),
            time_from_start=types.SimpleNamespace(to_sec=lambda ns=ns: ns*1e-9)))
    return types.SimpleNamespace(joint_names=row[4:11], points=points)


class MotionContractTests(unittest.TestCase):
    def test_recorded_010_curve_is_rejected_before_certification(self):
        motion = recorded_motion()
        self.assertEqual(violation(motion, [2.]*7, [3.,3.,4.,4.,6.,6.,6.]), 'reference_velocity_limit')

    def test_mainline_limits_agree_and_do_not_raise_actuator_speed(self):
        c = yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        names = c['joint_names']
        self.assertEqual(c['execution']['max_command_velocity'], 2.)
        for key in ['mpc', 'task_generator']:
            self.assertEqual(effective_limits(names, c[key]['joint_velocity_limits'], 2.), [2.]*7)
        with self.assertRaises(ValueError):
            effective_limits(names, dict(zip(names, [2.,2.,2.,2.,2.5,2.5,2.5])), 2.)

    def test_finite_curve_time_and_terminal_velocity_checks(self):
        def point(t, q, v):
            return types.SimpleNamespace(positions=[q], velocities=[v],
                time_from_start=types.SimpleNamespace(to_sec=lambda:t))
        t = types.SimpleNamespace(joint_names=['joint'], points=[point(0.,0.,0.),point(1.,.5,1.),point(2.,1.,0.)])
        self.assertIsNone(violation(t,[2.],[2.]))
        bad = copy.deepcopy(t);bad.points[-1].velocities=[.2]
        self.assertIsNotNone(violation(bad,[2.],[2.]))
        bad = copy.deepcopy(t);bad.points[1].positions=[math.nan]
        self.assertEqual(violation(bad,[2.],[2.]),'invalid_motion_state')


class StopEnvelopeTests(unittest.TestCase):
    def test_arbitrary_updates_then_braking_fit_the_absolute_envelope(self):
        rng = np.random.default_rng(73)
        a = np.array([3.,3.,4.,4.,6.,6.,6.]); vmax = np.full(7,2.)
        for scenario in range(80):
            gap = [.001,.01,.02,.05][scenario%4]
            hold = (scenario%11)*gap
            u = rng.uniform(-1.,1.,7)*vmax
            bound, horizon = bounded_update_travel(u,a,vmax,hold,gap)
            q = np.zeros(7); elapsed = 0.
            # Independent step-by-step command integration with arbitrary
            # reversals until deadline detection, then latched braking.
            for k in range(2000):
                q += np.abs(u)*gap
                elapsed += gap
                target = rng.uniform(-1.,1.,7)*vmax if elapsed <= hold+1e-12 else np.zeros(7)
                step = min(gap,.02)
                u += np.clip(target-u,-a*step,a*step)
                self.assertTrue(np.all(q<=bound+1e-10),(scenario,q,bound))
                if elapsed > hold and np.max(np.abs(u))<1e-12:
                    self.assertLessEqual(elapsed,horizon+1e-10)
                    break
            else:
                self.fail('braking never terminated')

    def test_next_cycle_and_inflight_commands_cannot_be_omitted(self):
        args = dict(now_s=10.,measured_s=9.999,applied=[0.]*7,acceleration=[3.]*7,
                    velocity_limits=[2.]*7,next_decision_s=.01,transport_bound_s=.01,update_gap_s=.01,
                    lease_end_s=10.1,lease_enforced=False,physical_error_rad=None,
                    physical_settle_s=None,snapshot=None)
        result = qualify(**args)
        self.assertFalse(result['qualified']);self.assertFalse(result['runtime_admission'])
        for reason in ['missing_enforced_command_lease','missing_physical_response_bound','missing_complete_stopping_snapshot']:
            self.assertIn(reason,result['reasons'])
        now = np.array(result['budgets']['withdraw_now']['command_travel_rad'])
        nxt = np.array(result['budgets']['continue_then_stop']['command_travel_rad'])
        self.assertTrue(np.all(nxt>now));self.assertTrue(np.all(now>0))

    def test_missing_response_or_time_bounds_never_get_zero_defaults(self):
        r=qualify(now_s=25.125,measured_s=25.12,applied=[0.]*7,acceleration=[3.]*7,
                  velocity_limits=[2.]*7,next_decision_s=None,transport_bound_s=None,
                  update_gap_s=None,lease_end_s=None,lease_enforced=False,
                  physical_error_rad=None,physical_settle_s=None,snapshot=None)
        self.assertFalse(r['qualified']);self.assertEqual(r['budgets'],{})
        self.assertIn('missing_or_invalid_time_bounds',r['reasons'])


if __name__ == '__main__':
    unittest.main()

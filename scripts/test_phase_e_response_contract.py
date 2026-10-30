#!/usr/bin/env python3
"""Response algebra, observer alignment and false-qualification regressions."""
import copy
import unittest
import numpy as np
from phase_e_response_contract import audit_steps, conditional_error_budget, persistent_rest_index


def trace(steps=5):
    rows = []
    q = np.zeros(7)
    for i in range(steps):
        motor = np.full(7, .1)
        velocity = motor + .002
        after = q + .001 * velocity + 1e-9
        rows.append(dict(index=i, t_ns=1000000000+i*1000000, end_time_ns=1000000000+i*1000000,
            step_s=.001, q_before=q.tolist(), q_after=after.tolist(),
            v_before=velocity.tolist(), velocity=velocity.tolist(), motor=motor.tolist(),
            motor_after=motor.tolist(), fmax=[30.]*7))
        q = after
    return rows


class ResponseContractTests(unittest.TestCase):
    def test_velocity_and_pose_defects_both_contribute(self):
        rows = trace()
        r = audit_steps(rows)
        np.testing.assert_allclose(r['velocity_defect'], .002)
        np.testing.assert_allclose(r['increment_defect'], 1e-9, atol=1e-18)
        np.testing.assert_allclose(np.sum(r['command_defect'], axis=0), 5*(.001*.002+1e-9))

    def test_conditional_budget_covers_signed_defect_sequences(self):
        rng = np.random.default_rng(23)
        eps, eta = np.arange(1, 8)*.001, np.arange(1, 8)*1e-8
        for count in (0, 1, 100, 1000):
            defects = .001*rng.uniform(-eps, eps, (count, 7)) + rng.uniform(-eta, eta, (count, 7))
            b = conditional_error_budget(steps=count, step_s=.001,
                velocity_residual_bound=eps, pose_increment_residual_bound=eta)
            self.assertTrue(np.all(np.abs(np.sum(defects, axis=0)) <= np.asarray(b['error_rad'])+1e-16))
            self.assertFalse(b['runtime_admission'])
            self.assertFalse(b['physical_bound_certified'])
            self.assertFalse(b['settling_bound_established'])

    def test_zero_error_does_not_qualify_future_response(self):
        b = conditional_error_budget(steps=100, step_s=.001,
            velocity_residual_bound=[0.]*7, pose_increment_residual_bound=[0.]*7)
        self.assertEqual(b['error_rad'], [0.]*7)
        self.assertFalse(b['physical_bound_certified'])

    def test_initial_error_is_retained(self):
        b = conditional_error_budget(steps=0, step_s=.001, initial_error=[.03]*7,
            velocity_residual_bound=[0.]*7, pose_increment_residual_bound=[0.]*7)
        self.assertEqual(b['error_rad'], [.03]*7)

    def test_negative_bounds_rejected(self):
        with self.assertRaises(ValueError):
            conditional_error_budget(steps=1, step_s=.001,
                velocity_residual_bound=[-.01]*7, pose_increment_residual_bound=[0.]*7)

    def test_nonfinite_bounds_rejected(self):
        with self.assertRaises(ValueError):
            conditional_error_budget(steps=1, step_s=.001,
                velocity_residual_bound=[float('nan')]*7, pose_increment_residual_bound=[0.]*7)

    def test_invalid_step_count_rejected(self):
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                conditional_error_budget(steps=count, step_s=.001,
                    velocity_residual_bound=[0.]*7, pose_increment_residual_bound=[0.]*7)

    def test_future_motor_mistaken_for_current_motor_rejected(self):
        rows = trace()
        rows[2]['motor_after'][0] = .2
        with self.assertRaisesRegex(ValueError, 'motor_changed'):
            audit_steps(rows)

    def test_missing_physics_step_rejected(self):
        rows = trace()
        del rows[2]
        with self.assertRaisesRegex(ValueError, 'missing_observer_record'):
            audit_steps(rows)

    def test_timestamp_gap_with_contiguous_log_indices_rejected(self):
        rows = trace()
        rows[2]['t_ns'] += 1000000
        rows[2]['end_time_ns'] = rows[2]['t_ns']
        with self.assertRaisesRegex(ValueError, 'domain_changed'):
            audit_steps(rows)

    def test_changed_step_does_not_pass_nominal_contract(self):
        rows = trace()
        rows[2]['step_s'] = .002
        with self.assertRaisesRegex(ValueError, 'domain_changed'):
            audit_steps(rows)
        r = audit_steps(rows, allow_discontinuities=True)
        self.assertGreater(len(r['nonnominal_indices']), 0)
        self.assertFalse(r['physical_bound_certified'])

    def test_reset_does_not_merge_epochs(self):
        rows = trace()
        for r in rows[3:]:
            r['t_ns'] -= 1000000000
            r['end_time_ns'] = r['t_ns']
        with self.assertRaises(ValueError):
            audit_steps(rows)
        self.assertIn(2, audit_steps(rows, allow_discontinuities=True)['discontinuity_indices'])

    def test_pose_teleport_is_not_response_error(self):
        rows = trace()
        rows[3]['q_before'][0] += .01
        with self.assertRaisesRegex(ValueError, 'pose_discontinuity'):
            audit_steps(rows)

    def test_wrong_end_phase_is_rejected(self):
        rows = trace()
        rows[2]['end_time_ns'] += 1000000
        with self.assertRaisesRegex(ValueError, 'begin_end_stamp'):
            audit_steps(rows)

    def test_nonfinite_joint_state_rejected(self):
        rows = trace()
        rows[2]['velocity'][1] = None
        with self.assertRaisesRegex(ValueError, 'invalid_physics_velocity'):
            audit_steps(rows)

    def test_force_limit_sign_rejected(self):
        rows = trace()
        rows[2]['fmax'][1] = -1.
        with self.assertRaisesRegex(ValueError, 'negative_motor_force'):
            audit_steps(rows)

    def test_one_low_velocity_sample_is_not_persistent_rest(self):
        v = np.ones((200, 7))*.01
        v[50] = 0.
        self.assertIsNone(persistent_rest_index(v, 0))

    def test_rest_search_includes_last_complete_window(self):
        v = np.zeros((110, 7))
        v[:9] = .01
        self.assertEqual(persistent_rest_index(v, 0), 9)

    def test_rest_cannot_be_borrowed_from_before_stop(self):
        v = np.zeros((150, 7))
        v[80:] = .01
        self.assertIsNone(persistent_rest_index(v, 80, hold_steps=50))

    def test_short_trace_never_establishes_rest(self):
        self.assertIsNone(persistent_rest_index(np.zeros((100, 7)), 0))


if __name__ == '__main__':
    unittest.main()

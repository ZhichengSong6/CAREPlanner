#!/usr/bin/env python3
"""Pure report logic tests; never launch ROS or CUDA."""
import copy
import unittest
from run_phase_e_candidate_tail_pairs import aggregate, quantile
from compare_phase_e_unified_primitives import compare
from test_phase_e_unified_primitives import candidate_summary_matches
from phase_e_acceptance_scope import trajectory_workload


def fixture(backend='samples', value=2.):
    return dict(status='PASS', backend=backend, instrumented=False, knots=2,
                execution_event_driven=True, sha256={'fixed': 'hash'},
                latency_rows_ms=[{'candidate': value} for _ in range(20)],
                node_timing_rows_ms=[{'candidate': {'confidence_rpc_ms': value}} for _ in range(20)],
                latency_ms={'candidate': {'p50': value, 'p95': value}})


class TailPairTests(unittest.TestCase):
    def test_mainline_is_20_intervals_21_states(self):
        workload = trajectory_workload(21)
        self.assertEqual(workload['intervals'], 20)
        self.assertEqual(workload['horizon_duration_s'], 1.)
        self.assertTrue(workload['matches_mainline_horizon'])

    def test_stress_failure_is_not_mainline_failure(self):
        workload = trajectory_workload(50)
        self.assertEqual(workload['role'], 'stress_only')
        self.assertFalse(workload['failure_counts_as_mainline_failure'])

    def test_old_20_state_fixture_is_not_exact_mainline(self):
        self.assertFalse(trajectory_workload(20)['matches_mainline_horizon'])
        self.assertAlmostEqual(trajectory_workload(20)['horizon_duration_s'], .95)
        self.assertEqual(trajectory_workload(2, closed_loop=True)['trajectory_states'], 21)

    def test_nearest_rank(self):
        self.assertEqual(quantile(list(range(1, 21)), .95), 19)

    def test_pools_rows_not_percentiles(self):
        result = aggregate([fixture(value=1.), fixture(value=10.)])
        self.assertEqual(result['latency_ms']['candidate']['p95'], 10.)
        self.assertEqual(len(result['latency_rows_ms']), 40)

    def test_preserves_failure(self):
        fail = fixture(); fail['status'] = 'FAIL'
        self.assertEqual(aggregate([fixture(), fail])['status'], 'FAIL')

    def test_rejects_changed_inputs(self):
        changed = fixture(); changed['sha256']['fixed'] = 'changed'
        with self.assertRaises(AssertionError): aggregate([fixture(), changed])

    def test_no_tolerance(self):
        self.assertEqual(compare(fixture(), fixture('primitive', 2.000001))['status'], 'FAIL')
        self.assertEqual(compare(fixture(), fixture('primitive', 1.9))['status'], 'PASS')

    def test_rejects_mixed_workloads(self):
        for field, value in [('knots', 20), ('execution_event_driven', False), ('instrumented', True), ('persistent_confidence', True),
                             ('shadow_event_driven', True), ('input_period_s', .06), ('measurement_schema', 'different'), ('fixture_tcp_nodelay', True),
                             ('verification_tcp_nodelay', True), ('map_query_dedicated_queue', True), ('unknown_soak', True), ('observer_history_limit', 16)]:
            new = fixture('primitive'); new[field] = value
            with self.assertRaises(AssertionError): compare(fixture(), new)

    def test_candidate_cannot_match_bootstrap_or_wrong_stamp(self):
        self.assertTrue(candidate_summary_matches('trajectory_stamp_ns=123 trajectory_source=predicted success=1', 123))
        for text in ['trajectory_stamp_ns=123 trajectory_source=bootstrap success=1',
                     'trajectory_stamp_ns=1234 trajectory_source=predicted success=1',
                     'trajectory_stamp_ns=123 trajectory_source=predicted success=0']:
            self.assertFalse(candidate_summary_matches(text, 123))


if __name__ == '__main__': unittest.main()

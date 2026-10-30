#!/usr/bin/env python3
"""Exercise real map-query/publication methods and their feedback identity."""
from types import SimpleNamespace as NS
import unittest
import test_phase_e_dependency_publish_chain as chain

class FeedbackTests(unittest.TestCase):
    def node(self):
        n,fixture,clock=chain.PublishChainTests().node()
        n._compute_progressive_shared_target=lambda *args:None
        n._publish_schedule()
        diagnostics=[]
        n._publish_acquisition_summary=lambda rows:diagnostics.extend(rows)
        return n,clock,diagnostics

    def test_real_query_carries_request_time_and_exact_published_target(self):
        n,clock,rows=self.node();token=n._trace_published_target['observation_token']
        n._update_actual_visibility_completion()
        row=next(r for r in rows if r[0]==1);diag=row[9]
        self.assertEqual(diag['status'],'ok')
        self.assertEqual(diag['query_ros_s'],clock.now)
        self.assertEqual(diag['progress_observation_token'],token)

    def test_target_change_during_service_discards_feedback_identity(self):
        n,clock,rows=self.node();client=n.confidence_client
        def changed(req):
            n._trace_published_target=dict(n._trace_published_target,observation_token='care_obs_v1_changed')
            return client(req)
        n.confidence_client=changed;n._update_actual_visibility_completion()
        self.assertEqual(next(r for r in rows if r[0]==1)[9]['progress_observation_token'],'none')

    def test_slow_service_keeps_original_query_time(self):
        n,clock,rows=self.node();client=n.confidence_client
        def slow(req):
            clock.now+=2.;return client(req)
        n.confidence_client=slow;n._update_actual_visibility_completion()
        self.assertEqual(next(r for r in rows if r[0]==1)[9]['query_ros_s'],1.)
        self.assertGreater(clock.now-rows[0][9]['query_ros_s'],.5)

if __name__=='__main__':unittest.main()

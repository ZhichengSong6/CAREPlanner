"""Actual single-flight callbacks: timed lookahead and original ownership."""
import unittest
from std_msgs.msg import String
import test_phase_e_candidate_admission as admission_fixture

class ProbeHandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture=admission_fixture.AdmissionTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.g=self.fixture.g
        self.g.admission_enabled=False;self.g.prefix_binding_enabled=False
        self.g.sender=None
        self.g.probe_active=True;self.g.phase=self.g.VERIFYING

    def commit(self,token=100,streak=0):
        self.g.probe_completed_prefix_streak=streak
        self.g._verification_outcome_cb(String(data=
            'verification_view=probe_prefix_brake_hold result=safe committed=1 '
            f'execution_stamp_ns={token} probe_effective_prefix_s=0.15'))

    def phase(self,t,token=100,extra=''):
        self.g._tracker_summary_cb(String(data=
            f'execution_stamp_ns={token} phase_s={t} active=1 complete=0 '
            'source=active_trajectory '+extra))

    def test_lookahead_waits_for_measured_half_prefix_and_only_requests_once(self):
        self.commit();self.assertEqual(len(self.g.replan_request_pub.messages),0)
        self.phase(.04);self.assertEqual(len(self.g.replan_request_pub.messages),0)
        self.phase(.08);self.phase(.09)
        self.assertEqual(len(self.g.replan_request_pub.messages),1)
        self.assertEqual(self.g.phase,self.g.EXECUTING)

    def test_buffer_cannot_forward_before_the_certified_prefix_boundary(self):
        self.commit();self.phase(.08)
        self.g._candidate_cb(self.fixture.raw)
        self.assertIsNotNone(self.g.buffered_candidate)
        self.assertFalse(self.g.output_pub.messages)
        self.phase(.149);self.assertFalse(self.g.output_pub.messages)
        self.phase(.15);self.assertEqual(len(self.g.output_pub.messages),1)
        self.assertEqual(self.g.phase,self.g.VERIFYING)

    def test_wrong_token_cannot_request_or_release_and_final_prefix_has_no_lookahead(self):
        self.commit();self.phase(.2,token=101)
        self.assertEqual(self.g.phase,self.g.EXECUTING)
        self.assertFalse(self.g.replan_request_pub.messages)
        self.g.phase=self.g.VERIFYING;self.commit(token=200,streak=2)
        self.phase(.1,token=200);self.phase(.15,token=200)
        self.assertFalse(self.g.replan_request_pub.messages)
        self.assertIsNone(self.g.buffered_candidate)

if __name__=='__main__':unittest.main()

"""Owned reserve use retains final learned gates and content-bound admission."""
import copy
import unittest
import test_phase_e_geometric_restore as existing
from geometric_restore import validate, permits_unknown_conflicts


class MarginRecoveryTests(unittest.TestCase):
    def intent(self):
        r=existing.intent()
        r.update(schema=2,query_ros_ns=r['query_stamp_ns'],margin_recovery=True)
        return r

    def test_explicit_margin_intent_retains_every_learned_row(self):
        r=self.intent()
        b=existing.IntentTests().batch()
        self.assertEqual(validate(r),r)
        self.assertTrue(permits_unknown_conflicts(r,b,[.1],.01))
        self.assertFalse(permits_unknown_conflicts(r,b,[-.1],.01))
        for bad in [False,1,'true',None]:
            changed=dict(r,margin_recovery=bad)
            with self.assertRaises(ValueError):validate(changed)

    def test_final_gcdf_conflict_stops_margin_candidate_before_vbc(self):
        b=existing.RecoveryAdmissionTests(methodName='runTest');b.setUp()
        self.addCleanup(b.doCleanups)
        b.r.update(schema=2,query_ros_ns=b.r['query_stamp_ns'],margin_recovery=True)
        b.contract['geometric_restore']=copy.deepcopy(b.r)
        b.dispatch()
        self.assertFalse(b.n.verification_pub.messages)
        self.assertFalse(b.n.committed_pub.messages)


if __name__=='__main__':unittest.main()

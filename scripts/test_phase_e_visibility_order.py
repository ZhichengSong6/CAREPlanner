"""Observation approach ordering preserves the finite-pool fallback."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from observation_task_rejoin import improves_candidate


class OrderTests(unittest.TestCase):
    def cost(self,total,detour):
        return dict(approach_s=total/2,rejoin_s=total/2,total_s=total,
                    task_pose_error=1.,approach_requires_detour=detour)

    def test_clearer_approach_can_win_despite_longer_joint_proxy(self):
        self.assertTrue(improves_candidate(self.cost(6.6,False),self.cost(4.5,True),.1))

    def test_cheaper_body_entry_before_visibility_cannot_replace_clear_approach(self):
        self.assertFalse(improves_candidate(self.cost(1.,True),self.cost(6.,False),.01))

    def test_same_class_keeps_original_gain_and_computation_rule(self):
        for detour in (False,True):
            self.assertFalse(improves_candidate(self.cost(4.9,detour),self.cost(5.,detour),0.))
            self.assertTrue(improves_candidate(self.cost(4.,detour),self.cost(5.,detour),.1))
            self.assertFalse(improves_candidate(self.cost(4.,detour),self.cost(5.,detour),1.))

    def test_missing_map_and_invalid_cost_do_not_gain_ordering_authority(self):
        base=self.cost(4.5,True);base.update(corridor_used=True,corridor_preview=dict(map_identity=['map',1]))
        good=self.cost(6.6,False)
        self.assertFalse(improves_candidate(good,base,0.))
        for value in (float('nan'),float('inf'),-1.):
            good=self.cost(value,False)
            self.assertFalse(improves_candidate(good,self.cost(4.5,True),0.))


if __name__=='__main__':unittest.main(verbosity=2)

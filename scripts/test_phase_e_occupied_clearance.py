import math,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/egocentric_arm_planner/scripts'))
from occupied_clearance import occupied_violations,voxel_volume_clearance
class OccupiedContractTest(unittest.TestCase):
 def batch(self,sources,centres):return NS(num_pairs=len(sources),source_type=sources,approx_body_clearance_m=centres)
 def test_real_case011_positive_learned_distance_does_not_override_geometry(self):
  b=self.batch([1],[.038891538]);b.distance=[.2]
  rows,d=occupied_violations(b,.05,0.)
  self.assertEqual(rows,[0]);self.assertAlmostEqual(d,-.004409732189221932)
 def test_unknown_does_not_become_physical_collision(self):
  self.assertEqual(occupied_violations(self.batch([0],[-1]),.05,0.),([],math.inf))
 def test_equality_and_unchanged_hard_threshold(self):
  r=.5*math.sqrt(3)*.05
  self.assertEqual(occupied_violations(self.batch([1],[r]),.05,0.)[0],[])
  self.assertEqual(occupied_violations(self.batch([1],[r-1e-7]),.05,0.)[0],[0])
 def test_missing_nonfinite_or_invalid_geometry_fails_closed(self):
  for b in [self.batch([1],[]),self.batch([1],[math.nan]),self.batch([2],[.1]),NS(num_pairs=-1,source_type=[],approx_body_clearance_m=[])]:
   with self.subTest(b=b),self.assertRaises(ValueError):occupied_violations(b,.05,0.)
 def test_mixed_sources_worst_physical_index(self):
  rows,d=occupied_violations(self.batch([0,1,1],[-1,.03,.05]),.05,0.)
  self.assertEqual(rows,[1]);self.assertEqual(d,voxel_volume_clearance(.03,.05))
if __name__=='__main__':unittest.main()

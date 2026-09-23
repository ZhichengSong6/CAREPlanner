import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from augment import refine_anchor

class Tests(unittest.TestCase):
 def test_refine_plane(self):
  def geo(q):
   h=np.array([q[0],1.0])
   j=np.zeros((2,7));j[0,0]=1.0
   return h,j
  q=np.zeros(7);q[0]=.08
  r=refine_anchor(q,np.array([1,0,0,0,0,0,0.]),np.full(7,-1.),np.full(7,1.),geo)
  self.assertTrue(r["success"]);self.assertLess(abs(r["g"]),1e-6);self.assertLess(abs(r["q_star"][0]),1e-6)
 def test_inactive_joint_unchanged(self):
  def geo(q):
   h=np.array([q[0]+q[1],1.0]);j=np.zeros((2,7));j[0,:2]=1.
   return h,j
  q=np.zeros(7);q[:2]=(.1,.3)
  r=refine_anchor(q,np.array([1,0,0,0,0,0,0.]),np.full(7,-1.),np.full(7,1.),geo)
  self.assertTrue(r["success"]);self.assertAlmostEqual(r["q_star"][1],.3,12)
if __name__=="__main__":unittest.main()

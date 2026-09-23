import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from fast_core import ExactMemo

class Tests(unittest.TestCase):
 def test_exact_memo_reuses_nonconsecutive_points(self):
  calls={"n":0}
  def geo(q):
   calls["n"]+=1;h=np.array([q[0],1.]);j=np.zeros((2,7));j[0,0]=1.;return h,j
  m=ExactMemo(geo,np.zeros(7),np.array([0]))
  m.evaluate(np.array([.1]));m.evaluate(np.array([.2]));m.evaluate(np.array([.1]))
  self.assertEqual(calls["n"],2);self.assertEqual(m.hits,1)
if __name__=="__main__":unittest.main()

import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from prod_common import select_v3_indices

class Tests(unittest.TestCase):
 def test_v3_selection_covers_both_signs_when_possible(self):
  q=np.zeros((8,7));q[:,0]=np.arange(8)
  g=np.ones((8,8));g[:4,0]=-1;g[::2,1]=-1
  idx,av,co=select_v3_indices(q,g,np.array([1,1,0,0,0,0,0,0],bool),4,np.full(7,-10.),np.full(7,10.))
  self.assertEqual(len(set(idx.tolist())),4)
  self.assertTrue(np.array_equal(av[:2],co[:2]))
if __name__=="__main__":unittest.main()

import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from analytic_geometry import _revolute
class Tests(unittest.TestCase):
 def test_rodrigues_z(self):
  r=_revolute(np.array([0.,0.,1.]),np.pi/2)
  np.testing.assert_allclose(r@np.array([1.,0.,0.]),np.array([0.,1.,0.]),atol=1e-12)
if __name__=="__main__":unittest.main()

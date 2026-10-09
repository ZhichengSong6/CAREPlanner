import sys,unittest,tempfile
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent))
import cache,objective

class Tests(unittest.TestCase):
 def test_group_v3_masks_invalid(self):
  g=dict(x_index=np.array([7]),split=np.array([0],np.uint8),q_pool=np.arange(4*7,dtype=np.float32).reshape(1,4,7),
         g_pool=np.ones((1,4,8),np.float32),v3_selected_index=np.array([[0,1]],np.int16))
  v=dict(x_index=np.array([7,7]),sensor=np.array([0,1]),q_slot=np.array([0,0]),value_valid=np.array([1,0],bool),new_value=np.array([.4,.8],np.float32))
  z=cache.group_v3(g,v)
  self.assertEqual(z["sensor_value_mask"].shape,(2,8));self.assertTrue(z["sensor_value_mask"][0,0]);self.assertFalse(z["sensor_value_mask"][0,1]);self.assertFalse(z["union_value_mask"][0])
 def test_sign_loss_is_lower_for_correct_prediction(self):
  y=torch.tensor([-.5,.5]);s=torch.tensor([-1.,1.])
  good=objective._logistic(y,s,.1).mean();bad=objective._logistic(-y,s,.1).mean()
  self.assertLess(float(good),float(bad))
if __name__=="__main__":unittest.main()

import sys,unittest
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
 def test_sign_loss_prefers_correct(self):
  y=torch.tensor([[-.5],[.5]]);s=torch.tensor([[-1.],[1.]])
  good=objective._logistic(y,s,.1).mean();bad=objective._logistic(-y,s,.1).mean()
  self.assertLess(float(good),float(bad))
 def test_class_cells_expose_majority_baseline(self):
  y=torch.full((101,1),-1.0);s=torch.full((101,1),-1.0);s[-1,0]=1.0
  _,cnt,correct=objective._class_cells(y,s,.1)
  raw=float(correct.sum()/cnt.sum())
  bal=float((correct[cnt>0]/cnt[cnt>0]).mean())
  self.assertGreater(raw,.98);self.assertAlmostEqual(bal,.5,places=6)
  self.assertEqual(cnt.tolist(),[[100.0,1.0]])
if __name__=="__main__":unittest.main()

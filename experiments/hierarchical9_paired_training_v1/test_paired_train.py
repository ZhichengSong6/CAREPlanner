import unittest
from pathlib import Path
import numpy as np,torch,sys
sys.path.insert(0,str(Path(__file__).parent))
from common import stream_indices,arm_spec,paired_loss
class Tiny(torch.nn.Module):
 def __init__(self):super().__init__();self.w=torch.nn.Linear(10,9,bias=False)
 def forward(self,x):return self.w(x)
class Tests(unittest.TestCase):
 def batch(self):
  n=5;return dict(inputs=torch.randn(n,10),sensor_value=torch.randn(n,8),sensor_grad=torch.randn(n,8,7),sensor_value_mask=torch.ones(n,8,dtype=torch.bool),
   sensor_grad_mask=torch.ones(n,8,dtype=torch.bool),union_value=torch.randn(n),union_grad=torch.randn(n,7),union_value_mask=torch.ones(n,dtype=torch.bool),
   union_grad_mask=torch.ones(n,dtype=torch.bool),reference_g=torch.randn(n,8),support=torch.ones(n,8,dtype=torch.bool))
 def test_arm_symmetry(self):self.assertEqual(arm_spec("old_value"),("old",False));self.assertEqual(arm_spec("new_value_grad"),("new",True))
 def test_stream_deterministic(self):self.assertTrue(np.array_equal(stream_indices(256,7,3,64),stream_indices(256,7,3,64)));self.assertFalse(np.array_equal(stream_indices(256,7,3,64),stream_indices(256,7,4,64)))
 def test_value_only_ignores_grad_targets(self):
  torch.manual_seed(0);m=Tiny();b=self.batch();l1,_=paired_loss(m,b,"new",False,True);b["sensor_grad"].normal_(100,20);b["union_grad"].normal_(100,20);l2,_=paired_loss(m,b,"new",False,True);self.assertAlmostEqual(float(l1),float(l2),6)
 def test_empty_grad_mask_matches_value_only(self):
  torch.manual_seed(0);m=Tiny();b=self.batch();b["sensor_grad_mask"].zero_();b["union_grad_mask"].zero_();a,_=paired_loss(m,b,"new",False,True);c,_=paired_loss(m,b,"new",True,True);self.assertAlmostEqual(float(a),float(c),6)
if __name__=="__main__":unittest.main()

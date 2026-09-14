#!/usr/bin/env python3
from __future__ import annotations

import unittest
import torch

import abc_protocol as proto
from abc_model import build_arm, function_equivalence
import sensor_objective as routed

old=proto.old


class ABCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_cls=old.module("model",old.SCRATCH).HierarchicalVisibilityCDF
        torch.manual_seed(7)
        cls.base=model_cls(shared_layers=(16,12,8),branch_layers=(6,4)).float().eval()
        cls.x=torch.randn(11,10)

    def test_all_arms_are_function_equivalent_at_initialization(self):
        for arm in proto.ARMS:
            model=build_arm(self.base,arm)
            result=function_equivalence(self.base,model,self.x)
            self.assertLess(result["max_abs_value_error"],1e-7,arm)
            self.assertLess(result["max_abs_q_gradient_error"],1e-7,arm)

    def test_gradient_routing_is_sensor_specific(self):
        for arm in proto.ARMS:
            model=build_arm(self.base,arm)
            model.zero_grad(set_to_none=True)
            loss=model.forward_sensor(self.x,6).sum()
            loss.backward()
            got={n for n,p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0}
            self.assertTrue(any(n.startswith("sensor_heads.6") for n in got),arm)
            self.assertFalse(any(n.startswith("union_head") for n in got),arm)
            self.assertFalse(any(n.startswith("shared") for n in got),arm)
            self.assertFalse(any(n.startswith("early") or n.startswith("union_tail") for n in got),arm)
            self.assertFalse(any(n.startswith("sensor_heads.0") for n in got),arm)
            if arm=="B":
                self.assertTrue(any(n.startswith("adapters.6") for n in got))
            if arm=="C":
                self.assertTrue(any(n.startswith("private_tails.6") for n in got))

    def test_b_adapter_starts_identity(self):
        model=build_arm(self.base,"B")
        with torch.no_grad():
            h=model.shared(model.encode(self.x))
            for adapter in model.adapters:
                self.assertTrue(torch.equal(adapter(h),h))

    def test_c_private_tail_is_a_real_trainable_copy(self):
        model=build_arm(self.base,"C")
        union=list(model.union_tail.parameters())
        private=list(model.private_tails[0].parameters())
        self.assertEqual(len(union),len(private))
        for a,b in zip(union,private):
            self.assertTrue(torch.equal(a,b))
            self.assertFalse(a.requires_grad)
            self.assertTrue(b.requires_grad)

    def test_routed_loss_microbatch_additivity(self):
        """Optimizer loss must match; diagnostic FP32 reductions need FP32 tolerance.

        The production objective intentionally accumulates SDF/gradient/Eikonal sums
        in FP32 (matching hierarchical9). Splitting a batch changes floating-point
        reduction order, so continuous diagnostic sums are not bitwise additive.
        Integer-valued count/sign statistics, however, must remain exact.
        """
        obj=old.module("objective",old.SCRATCH)
        weights=obj.LossWeights(tension=0.0)
        model=build_arm(self.base,"A")
        torch.manual_seed(11)
        inputs=torch.randn(20,10)
        target=torch.randn(20,8)
        grad=torch.randn(20,8,7)
        grad=grad/grad.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        mask=torch.rand(20,8)>0.15
        counts=routed.counts_from_mask(mask)
        full,st_full=routed.loss_for_microbatch(model,inputs,target,grad,mask,counts,weights,training=False)
        l0,s0=routed.loss_for_microbatch(model,inputs[:9],target[:9],grad[:9],mask[:9],counts,weights,training=False)
        l1,s1=routed.loss_for_microbatch(model,inputs[9:],target[9:],grad[9:],mask[9:],counts,weights,training=False)
        self.assertTrue(torch.allclose(full,l0+l1,atol=2e-6,rtol=2e-6))
        summed=s0+s1
        # count and sign_correct are integer sums represented in float64 stats.
        for col in (routed.COUNT,routed.SIGN):
            self.assertTrue(torch.equal(st_full[:,col],summed[:,col]),f"column={col}")
        # Continuous entries originate from FP32 reductions, so use an FP32-scale
        # tolerance rather than an impossible 1e-8 exact-reduction requirement.
        continuous=[routed.SDF,routed.GRAD,routed.EIK,routed.TENSION,routed.ABS,routed.NORM]
        self.assertTrue(torch.allclose(st_full[:,continuous],summed[:,continuous],atol=3e-6,rtol=3e-6))

    def test_trainable_capacity_order(self):
        a=build_arm(self.base,"A").trainable_parameter_count()
        b=build_arm(self.base,"B").trainable_parameter_count()
        c=build_arm(self.base,"C").trainable_parameter_count()
        self.assertLess(a,b)
        self.assertLess(a,c)


if __name__=="__main__":
    unittest.main(verbosity=2)

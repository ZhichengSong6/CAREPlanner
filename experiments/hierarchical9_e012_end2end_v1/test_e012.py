#!/usr/bin/env python3
from __future__ import annotations

import unittest
import torch

import e012_protocol as proto
from e2e_model import build_from_p0,function_equivalence
from replay import ReplayBuffer

old=proto.old


class E012Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_cls=old.module("model",old.SCRATCH).HierarchicalVisibilityCDF
        torch.manual_seed(17)
        cls.p0=model_cls(shared_layers=(16,12,8),branch_layers=(6,4)).float().eval()
        cls.inputs=torch.randn(13,10)

    def test_private_tail_model_is_p0_equivalent_at_initialization(self):
        model=build_from_p0(self.p0)
        r=function_equivalence(self.p0,model,self.inputs)
        self.assertLess(r["max_abs_value_error"],1e-7)
        self.assertLess(r["max_abs_q_gradient_error"],1e-7)

    def test_all_saved_network_parameters_are_trainable(self):
        frozen=self.p0.requires_grad_(False)
        model=build_from_p0(frozen)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertGreater(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in self.p0.parameters()))

    def test_sensor_parameter_frozen_early_preserves_value_and_q_gradient(self):
        model=build_from_p0(self.p0)
        x=self.inputs[:7,:3].detach(); q0=self.inputs[:7,3:].detach().clone().requires_grad_(True)
        q1=q0.detach().clone().requires_grad_(True)
        y0=model.forward_sensor(torch.cat((x,q0),1),4,freeze_sensor_early=False)
        y1=model.forward_sensor(torch.cat((x,q1),1),4,freeze_sensor_early=True)
        g0=torch.autograd.grad(y0.sum(),q0,retain_graph=True)[0]
        g1=torch.autograd.grad(y1.sum(),q1,retain_graph=True)[0]
        self.assertTrue(torch.equal(y0,y1))
        self.assertTrue(torch.equal(g0,g1))

    def test_sensor_parameter_frozen_early_blocks_only_early_param_grad(self):
        model=build_from_p0(self.p0); model.zero_grad(set_to_none=True)
        x=self.inputs[:7,:3].detach(); q=self.inputs[:7,3:].detach().clone().requires_grad_(True)
        with model.sensor_early_parameter_frozen():
            y=model.forward_sensor(torch.cat((x,q),1),6)
            g=torch.autograd.grad(y.sum(),q,create_graph=True,retain_graph=True)[0]
            loss=y.square().mean()+g.square().mean()
        loss.backward()
        early=[p.grad for p in model.early.parameters()]
        sensor=[p.grad for p in model.sensor_tails[6].parameters()]+[p.grad for p in model.sensor_heads[6].parameters()]
        self.assertTrue(all(v is None or float(v.abs().sum())==0 for v in early))
        self.assertTrue(any(v is not None and float(v.abs().sum())>0 for v in sensor))
        self.assertTrue(torch.isfinite(g).all())

    def test_optimizer_groups_cover_each_parameter_exactly_once(self):
        model=build_from_p0(self.p0)
        groups=model.optimizer_groups(lr_early=1e-5,lr_union=5e-5,lr_sensor=1e-4)
        ids=[id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        self.assertEqual([g["lr"] for g in groups],[1e-5,5e-5,1e-4])

    def test_round_robin_shared_sensor_is_balanced(self):
        seq=[proto.selected_shared_sensor(i) for i in range(1,17)]
        self.assertEqual(seq,list(range(8))+list(range(8)))

    def test_replay_sampling_is_deterministic_and_per_sensor(self):
        buf=ReplayBuffer(16)
        for s in (0,3,7):
            x=torch.randn(9,3);q=torch.randn(9,7);labels=torch.where(torch.arange(9)%2==0,1.,-1.);g=labels*.01
            buf.add(s,x,q,labels,g)
        a=buf.sample(5,seed=2,step=9,device=torch.device("cpu"))
        b=buf.sample(5,seed=2,step=9,device=torch.device("cpu"))
        self.assertEqual(set(a),{0,3,7})
        for s in a:
            self.assertTrue(torch.equal(a[s]["inputs"],b[s]["inputs"]))
            self.assertTrue(torch.equal(a[s]["labels"],b[s]["labels"]))
            self.assertEqual(a[s]["inputs"].shape,(5,10))


if __name__=="__main__": unittest.main(verbosity=2)

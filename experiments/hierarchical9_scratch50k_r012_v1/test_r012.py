#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
m=load("r012_test_model",HERE/"model.py")

class R012Tests(unittest.TestCase):
    def test_r1_r2_same_architecture_and_initialization(self):
        torch.manual_seed(0); r1=m.build_model("R1")
        torch.manual_seed(0); r2=m.build_model("R2")
        self.assertEqual(r1.architecture(),r2.architecture())
        for (n1,p1),(n2,p2) in zip(r1.state_dict().items(),r2.state_dict().items()):
            self.assertEqual(n1,n2); self.assertTrue(torch.equal(p1,p2),n1)

    def test_r0_and_private_tail_shapes(self):
        x=torch.randn(11,10)
        for arm in ("R0","R1","R2"):
            torch.manual_seed(0); model=m.build_model(arm)
            y=model(x)
            self.assertEqual(y.shape,(11,9))
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_parameter_frozen_early_preserves_value_and_q_gradient(self):
        torch.manual_seed(3); model=m.build_model("R2")
        x=torch.randn(7,3); q0=torch.randn(7,7,requires_grad=True); q1=q0.detach().clone().requires_grad_(True)
        y0=model.forward_sensor(torch.cat((x,q0),1),4,freeze_sensor_early=False)
        y1=model.forward_sensor(torch.cat((x,q1),1),4,freeze_sensor_early=True)
        g0=torch.autograd.grad(y0.sum(),q0,retain_graph=True)[0]
        g1=torch.autograd.grad(y1.sum(),q1,retain_graph=True)[0]
        self.assertTrue(torch.equal(y0,y1))
        self.assertTrue(torch.equal(g0,g1))

    def test_frozen_sensor_path_blocks_only_early_parameter_grad(self):
        torch.manual_seed(4); model=m.build_model("R2"); model.zero_grad(set_to_none=True)
        inp=torch.randn(9,10,requires_grad=True)
        with model.sensor_early_parameter_frozen():
            y=model.forward_sensor(inp,6)
            qgrad=torch.autograd.grad(y.sum(),inp,create_graph=True,retain_graph=True)[0][:,3:]
            loss=y.square().mean()+qgrad.square().mean()
        loss.backward()
        early=list(model.early.parameters())
        private=list(model.sensor_tails[6].parameters())+list(model.sensor_heads[6].parameters())
        self.assertTrue(all(p.grad is None or float(p.grad.abs().sum())==0 for p in early))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in private))
        self.assertTrue(torch.isfinite(qgrad).all())

    def test_normal_private_tail_sensor_path_updates_early(self):
        torch.manual_seed(5); model=m.build_model("R1"); model.zero_grad(set_to_none=True)
        inp=torch.randn(9,10,requires_grad=True); y=model.forward_sensor(inp,2); y.square().mean().backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.early.parameters()))

if __name__=="__main__": unittest.main(verbosity=2)

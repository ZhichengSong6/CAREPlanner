"""CPU safeguards: original-R1 provenance, normal-only objective, no RNG leakage."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from boundary_aug import BoundaryDirection
import train


class SmallLinearHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        v=x[:,3:4]*self.weight
        return torch.cat((v,v.repeat(1,8)),dim=1)


class Tests(unittest.TestCase):
    def test_protocol_keeps_all_original_baseline_controls(self):
        cfg=train.protocol(); old=cfg["baseline"]
        self.assertEqual(cfg["steps"],50000)
        self.assertEqual(cfg["normal_direction_weight"],0.01)
        self.assertEqual(cfg["v4_cache_sha256"],
             "060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc")
        self.assertEqual((old["global_batch_x"],old["batch_q"],old["microbatch_x"]),(4000,100,250))
        self.assertEqual((old["seed"],old["stream_seed"],old["lr"],old["amp"]),(0,190915,1e-3,"fp16"))
        self.assertEqual((old["weight_sdf"],old["weight_grad"],old["weight_eikonal"],
                           old["weight_tension"]),(5.,.1,.01,.01))
        self.assertEqual((old["weight_union_objective"],old["weight_sensor_objective"],old["weight_consistency"]),(1.,1.,.1))
        self.assertEqual(cfg["anchors_per_sensor_per_rank"]*4*8,1024)

    def test_original_frozen_source_is_exact(self):
        train.verify_original_r1_code()

    def test_evaluator_imports_this_experiment_not_r012_train(self):
        source=(train.HERE/"evaluate.py").read_text()
        self.assertIn("sys.path.insert(0,str(HERE))\nfrom train import FORMAT", source)

    def test_original_r1_model_is_unmodified(self):
        mod=train.load_r012_module()
        import hashlib
        self.assertEqual(train.sha(train.R012/"model.py"),
                         "0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b")
        model=mod.models.build_model("R1")
        self.assertEqual(model.parameter_count(),2184329)
        self.assertEqual(len(model.sensor_heads),8)

    def test_synthetic_direction_loss_uses_cosine_only(self):
        cfg={"normal_direction_weight":.01}
        model=SmallLinearHeads()
        x=torch.zeros((16,3))
        q=torch.zeros((16,7),dtype=torch.float32)
        norm=torch.zeros((16,7));norm[:,0]=1
        sensor=torch.arange(8).repeat_interleave(2)
        loss,diag=BoundaryDirection.direction_loss(model,(x,q,norm,sensor),cfg,training=True,monitor=False)
        self.assertAlmostEqual(float(loss),0.,places=6)
        self.assertEqual(diag.shape,(8,5))
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad))
        reverse=-norm
        loss2,_=BoundaryDirection.direction_loss(model,(x,q,reverse,sensor),cfg,training=True,monitor=False)
        self.assertAlmostEqual(float(loss2),.02,places=6)
        self.assertAlmostEqual(float(diag[:,0].mean()),1.,places=5)

    def test_sampler_stateless_and_never_uses_original_validation_x(self):
        class FakeCache:
            def __init__(self,_path,verify=False):
                self.identity="cache"
                self.x=np.asarray([[float(i),0,0] for i in range(10)],np.float32)
                self.a={}
                for split, ids in (("train",[0,1,2,3,4,5,6,7,8,9]),
                                   ("val",[0,1,2,3,4,5,6,7,8,9])):
                    rows=[]
                    for s in range(8):
                        for ix in ids:
                            rows.append((ix,s))
                    xidx=np.asarray([x for x,s in rows],np.int32)
                    sensors=np.asarray([s for x,s in rows],np.uint8)
                    q=np.zeros((len(rows),7),np.float32)
                    n=np.zeros_like(q);n[:,0]=1.
                    self.a[split]={"boundary":{"x_index":xidx,"sensor":sensors,
                                                  "q":q,"grad":n}}
        class DummyMod:
            Cache=FakeCache
        class DS:
            x_cpu=torch.tensor([[float(i),0,0] for i in range(10)],dtype=torch.float32)
            train_indices_np=np.asarray([0,1,2,3,4,5,6,7])
            val_indices_np=np.asarray([8,9])
        cfg={"anchors_per_sensor_per_rank":3,"aux_stream_seed":261013}
        with mock.patch("boundary_aug.load_original_cache_reader",return_value=DummyMod):
            a=BoundaryDirection(Path("."),Path("."),DS(),cfg)
        self.assertEqual(a.counts["train"],[8]*8)
        self.assertEqual(a.counts["val"],[2]*8)
        before=torch.get_rng_state().clone()
        p=a.batch("train",11,0,torch.device("cpu"))
        q=a.batch("train",11,0,torch.device("cpu"))
        self.assertTrue(torch.equal(p[0],q[0]))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue((p[0][:,0]<8).all())
        self.assertTrue((a.batch("val",0,0,torch.device("cpu"))[0][:,0]>=8).all())
        self.assertEqual(len(p[3]),24)

    def test_original_r1_prepared_loss_function_and_scheduler_no_direct_edits(self):
        from pathlib import Path
        source=(train.SCRATCH/"train.py").read_text()
        r012=(train.R012/"train.py").read_text()
        self.assertIn("scaler.step(optimizer)",source)
        self.assertIn("scheduler.step(train_stats[\"loss\"])",r012)

    def test_source_isolation_fingerprints(self):
        f=train.dependency_fingerprints()
        self.assertIn("experiments/hierarchical9_r1_normal_aug_v1/boundary_aug.py",f)
        self.assertIn("experiments/hierarchical9_scratch50k_r012_v1/train.py",f)
        self.assertIn("experiments/hierarchical9_scratch_v1/objective.py",f)
        self.assertEqual(len(f),len(set(f)))
        self.assertEqual(len(set(f.values())),len(f))

    def test_split_mismatch_fails_loudly(self):
        class FakeCache:
            def __init__(self,_path,verify=False):
                self.identity="cache"
                self.x=np.ones((3,3),np.float32)*100
                self.a={}
        class DummyMod:Cache=FakeCache
        class DS:
            x_cpu=torch.zeros((3,3))
            train_indices_np=np.array([0,1])
            val_indices_np=np.array([2])
        with mock.patch("boundary_aug.load_original_cache_reader",return_value=DummyMod):
            with self.assertRaisesRegex(ValueError,"mapping mismatch"):
                BoundaryDirection(Path("."),Path("."),DS(),{"anchors_per_sensor_per_rank":1})


if __name__=="__main__":
    unittest.main()

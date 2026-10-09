"""CPU regression tests. Set BF_DISTRIBUTED_TEST=1 for two-rank Gloo parity."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data import Cache, FIELDS, OFFSETS, offset_codes, sha, write_json
from objective import cell_reduce, loss_and_metrics, average_parameter_gradients
from train import config, learning_rate, checked_load, atomic_checkpoint
from model import build_model


def batch_fixture(n=112):
    gen = torch.Generator().manual_seed(124)
    inp = torch.randn(n, 10, generator=gen)
    ss = torch.where(torch.rand(n, 8, generator=gen) > .4, 1, -1)
    sensor = torch.arange(n) % 8
    mask = torch.rand(n, 8, generator=gen) > .3
    vals = torch.randn(n, 8, generator=gen)
    vals[~mask] = float("nan")
    return {
        "global": {"inputs": inp, "sensor_sign": ss, "union_sign": torch.where((ss > 0).any(1), 1, -1)},
        "v3": {"inputs": inp.clone(), "sensor_value_mask": mask, "sensor_value": vals,
               "union_value": torch.full((n,), float("nan"))},
        "v4": {"inputs": inp.clone(), "sensor": sensor,
               "value": torch.as_tensor(OFFSETS[np.arange(n)%7]), "cell": sensor*7 + torch.arange(n)%7},
        "boundary": {"inputs": inp.clone(), "q": inp[:, 3:].clone(), "sensor": sensor,
                     "grad": torch.nn.functional.one_hot(sensor%7, 7).float()},
    }


def tiny_model():
    torch.manual_seed(199)
    return torch.nn.Sequential(torch.nn.Linear(10, 12), torch.nn.Tanh(), torch.nn.Linear(12, 9))


def make_cache(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    np.save(root/"x.npy", np.array([[0., 0., 0.], [1., 1., 1.]], np.float32))
    n = 112; counts = {}
    for split, xi in (("train", 0), ("val", 1)):
        (root/split).mkdir()
        b = batch_fixture(n)
        counts[split] = {k:n for k in FIELDS}
        for kind, fields in FIELDS.items():
            values = {"x_index": np.full(n, xi, np.int32), "q": b[kind]["inputs"][:, 3:].numpy()}
            for name in fields:
                if name not in values:
                    values[name] = b[kind][name].numpy()
            if kind == "boundary": values["sensor"] = values["sensor"].astype(np.uint8)
            for key in fields:
                np.save(root/split/f"{kind}_{key}.npy", values[key], allow_pickle=False)
    files = {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes":p.stat().st_size} for p in root.rglob("*.npy")}
    write_json(root/"manifest.json", {"format":"care_h9_v4_training_cache_v1", "status":"COMPLETE",
                "training_ready":True, "counts":counts, "files":files})


def distributed_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    try:
        b = batch_fixture()
        # Unequal validity per rank is intentional, including absent local heads.
        b["v3"]["sensor_value_mask"][1::2, 7] = False
        shard = {k:{f:v[rank::2].clone() for f,v in block.items()} for k,block in b.items()}
        m = tiny_model()
        loss, _ = loss_and_metrics(m, shard, config(), training=True)
        loss.backward(); average_parameter_gradients(m)
        flat = torch.cat([p.grad.flatten() for p in m.parameters()])
        if rank == 0: torch.save(flat, output)
    finally:
        dist.destroy_process_group()


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_model_is_byte_identical_r1_definition(self):
        raw = (Path(__file__).parent/"model.py").read_bytes()
        blob = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        self.assertEqual(blob, "0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b")
        m = build_model("R1")
        self.assertEqual(m.parameter_count(), 2184329)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_offsets_and_invalid_offsets(self):
        np.testing.assert_array_equal(offset_codes(OFFSETS), np.arange(7))
        for bad in (np.array([.3]), np.array([np.nan])):
            with self.assertRaises(ValueError): offset_codes(bad)

    def test_per_cell_mean_not_population_mean(self):
        x = torch.tensor([1., 1., 10.], requires_grad=True)
        loss, means, count = cell_reduce(x, torch.tensor([0, 0, 1]), 2)
        self.assertAlmostEqual(float(loss[0].detach()), 5.5)
        loss.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([.25, .25, .5]))

    def test_real_model_backward_and_finite_all_heads(self):
        m = build_model("R1")
        loss, met = loss_and_metrics(m, batch_fixture(56), config(), training=True)
        loss.backward(); average_parameter_gradients(m)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertIn("boundary_zero_mae_rad", met)
        # Evaluation must work without building a second-order training graph.
        _, v = loss_and_metrics(m, batch_fixture(56), config(), training=False)
        self.assertTrue(torch.isfinite(v["selection_score"]))

    def test_invalid_v3_and_union_values_are_not_supervised(self):
        m = tiny_model(); b = batch_fixture()
        l1, _ = loss_and_metrics(m, b, config(), training=True)
        b["v3"]["union_value"].fill_(1e15)
        b["v3"]["sensor_value"][~b["v3"]["sensor_value_mask"]] = -1e20
        l2, _ = loss_and_metrics(m, b, config(), training=True)
        torch.testing.assert_close(l1, l2)

    def test_zero_baseline_has_nonzero_normal_error(self):
        b = batch_fixture(); m = torch.nn.Linear(10, 9)
        torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
        _, v = loss_and_metrics(m, b, config(), training=True)
        self.assertAlmostEqual(float(v["loss_normal"].detach()), 1., places=6)
        self.assertAlmostEqual(float(v["local_mse_over_zero_baseline"]), 1., places=5)
        self.assertAlmostEqual(float(v["boundary_zero_mae_rad"]), 0.)

    def test_reader_indices_do_not_modify_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"cache"; make_cache(root)
            before = {str(p):sha(p) for p in root.rglob("*") if p.is_file()}
            c = Cache(root, verify=True, expected=None); c.build_indices(Path(d)/"indices")
            for kind in ("v4", "boundary"):
                ids = c.ids("train", kind, 224, 0, 12, 0, 1)
                batch = c.batch("train", kind, ids, torch.device("cpu"))
                groups = batch["cell"] if kind == "v4" else batch["sensor"]
                count = torch.bincount(groups.long())
                self.assertEqual(int(count.max()-count.min()), 0)
                np.testing.assert_array_equal(ids, c.ids("train", kind, 224, 0, 12, 0, 1))
            np.testing.assert_array_equal(c.ids("val", "v3", 0, 0, 0, 1, 4), np.arange(1,112,4))
            self.assertEqual(before, {str(p):sha(p) for p in root.rglob("*") if p.is_file()})
            # Reuse is checksum-verified rather than rebuilding indices.
            c.build_indices(Path(d)/"indices")

    def test_corrupt_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"cache"; make_cache(root)
            with (root/"train/global_q.npy").open("ab") as f: f.write(b"corrupt")
            with self.assertRaises(ValueError): Cache(root, verify=True, expected=None)

    def test_checkpoint_and_adam_resume(self):
        b = batch_fixture(56); c = config()
        m = tiny_model(); opt = torch.optim.Adam(m.parameters(), lr=.001)
        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_and_metrics(model, b, c, training=True)
            loss.backward(); average_parameter_gradients(model); optimizer.step()
        update(m,opt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"latest.pt"
            atomic_checkpoint(path,{"model_state":m.state_dict(),"optimizer_state":opt.state_dict()})
            cp=checked_load(path); m2=tiny_model(); o2=torch.optim.Adam(m2.parameters(),lr=.001)
            m2.load_state_dict(cp["model_state"]);o2.load_state_dict(cp["optimizer_state"])
            update(m,opt);update(m2,o2)
            for a,bp in zip(m.parameters(),m2.parameters()): torch.testing.assert_close(a,bp,rtol=0,atol=0)
            with path.open("ab") as f:f.write(b"broken")
            with self.assertRaises(ValueError): checked_load(path)

    def test_lr_schedule_boundaries(self):
        c=config()
        self.assertAlmostEqual(learning_rate(c,1000),c["lr_peak"])
        self.assertAlmostEqual(learning_rate(c,50000),c["lr_final"])
        self.assertLess(learning_rate(c,1),learning_rate(c,1000))

    @unittest.skipUnless(os.getenv("BF_DISTRIBUTED_TEST")=="1", "optional two-rank CPU parity")
    def test_two_rank_gradient_matches_single_process(self):
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/"grads.pt")
            mp.spawn(distributed_worker,args=(str(Path(d)/"gloo"),path),nprocs=2,join=True)
            distributed=torch.load(path,weights_only=True)
            b=batch_fixture();b["v3"]["sensor_value_mask"][1::2,7]=False
            m=tiny_model(); loss,_=loss_and_metrics(m,b,config(),training=True);loss.backward()
            reference=torch.cat([p.grad.flatten() for p in m.parameters()])
            torch.testing.assert_close(distributed,reference,rtol=2e-5,atol=3e-5)

if __name__=="__main__": unittest.main()

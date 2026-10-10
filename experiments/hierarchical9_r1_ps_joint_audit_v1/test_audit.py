"""CPU tests: actual upstream readers/objective when JEA_TEST_REPO is set.

No training job, no optimizer step, no production artifacts required.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Dependencies, aliases, assert_disjoint_outputs, assert_unchanged, blob_sha,
                    json_safe, sha, state_digest, urdf_masks, write_json, checkpoint_manifest)
from plan import Reservoir, make_plan, verify_selected_links
from boundary import geometry_check, point_metrics, activation_audit
from gradients import diagnose_model, flatten, group_stats, parameter_groups
from report import exact_p, load_rows

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/"protocol.json").read_text())
REPO = os.environ.get("JEA_TEST_REPO")
torch.set_num_threads(2)


class MiniH9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.early = torch.nn.Sequential(torch.nn.Linear(10, 12), torch.nn.Softplus())
        self.union_tail = torch.nn.Linear(12, 6)
        self.union_head = torch.nn.Linear(6, 1)
        self.sensor_tails = torch.nn.ModuleList([torch.nn.Linear(12, 6) for _ in range(8)])
        self.sensor_heads = torch.nn.ModuleList([torch.nn.Linear(6, 1) for _ in range(8)])

    def forward(self, x):
        h = self.early(x)
        return torch.cat([self.union_head(self.union_tail(h))] +
            [head(tail(h)) for head, tail in zip(self.sensor_heads, self.sensor_tails)], 1)


class PlaneField(torch.nn.Module):
    def __init__(self, slope=1.):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(float(slope)))

    def forward(self, x):
        return self.slope*x[:, 3:4].expand(-1, 9)


class PlaneOracle:
    def planes(self, x, q, s):
        first = q[:, 0]
        return torch.stack([first]+[first*0+10+i for i in range(5)], -1)

    def value(self, x, q, s):
        return float(q[0].detach())

    def verify_against_upstream(self, x, qs):
        return {"status": "PASS", "max_abs_error_m": 0.}


def fixture_batch(n=56):
    generator = torch.Generator().manual_seed(1121)
    inputs = torch.randn(n, 10, generator=generator)*.1
    sid = torch.arange(n)%8
    rid = torch.arange(n)%3
    normal = torch.nn.functional.one_hot(sid%7, 7).float()
    plus, minus = inputs.clone(), inputs.clone()
    radius = torch.tensor([.005, .01, .02])[rid]
    plus[:, 3:] += radius[:, None]*normal
    minus[:, 3:] -= radius[:, None]*normal
    offsets = torch.tensor([-.02, -.01, -.005, 0, .005, .01, .02])[torch.arange(n)%7]
    return {
        "global": {"inputs": inputs, "sensor_sign": torch.where(torch.randn(n, 8, generator=generator) > 0, 1, -1),
                   "union_sign": torch.where(torch.arange(n)%2 == 0, 1, -1)},
        "v3": {"inputs": inputs, "sensor_value_mask": torch.ones(n, 8, dtype=torch.bool),
               "sensor_value": torch.randn(n, 8, generator=generator)},
        "v4": {"inputs": inputs, "sensor": sid, "value": offsets, "cell": sid*7+torch.arange(n)%7},
        "boundary": {"inputs": inputs, "q": inputs[:, 3:], "sensor": sid, "grad": normal},
        "pair": {"plus_inputs": plus, "minus_inputs": minus, "sensor": sid, "radius_id": rid},
    }


def production_fixture(root: Path, deps):
    prod, cr = root/"production", root/"cache"
    (prod/"v4_tubes").mkdir(parents=True)
    records = []
    offsets = np.asarray([0, -.005, .005, -.01, .01, -.02, .02], np.float32)
    for sp in (0, 1):
        for s in range(8):
            normal = np.eye(7, dtype=np.float32)[s%7]
            q0 = np.full(7, .05*s, np.float32)
            for off in offsets:
                records.append((sp, s, off, q0+off*normal, normal))
    raw = {
        "x_index": np.asarray([r[0] for r in records], np.int64),
        "split": np.asarray([r[0] for r in records], np.uint8),
        "sensor": np.asarray([r[1] for r in records], np.uint8),
        "source_slot": np.zeros(len(records), np.int32),
        "value": np.asarray([r[2] for r in records], np.float32),
        "q": np.asarray([r[3] for r in records], np.float32),
        "grad": np.asarray([r[4] for r in records], np.float32),
        "g_m": np.asarray([r[2] for r in records], np.float32),
    }
    for sp in (0, 1):
        path = prod/f"v4_tubes/shard_{sp:04d}.npz"
        np.savez_compressed(path, **{k:v[raw["split"] == sp] for k,v in raw.items()})
        write_json(path.with_suffix(".json"), {"sha256": sha(path)})
    write_json(prod/"manifest.json", {"type": "TEST_ONLY"})
    write_json(prod/"v4_summary.json", {"status": "COMPLETE"})
    cr.mkdir()
    np.save(cr/"x.npy", np.asarray([[.1, .2, .3], [.2, .3, .4]], np.float32))
    counts = {}
    for sp, name in ((0, "train"), (1, "val")):
        d = cr/name
        d.mkdir()
        vm = raw["split"] == sp
        bm = vm & (raw["value"] == 0)
        groups = {
            "global": {"x_index": np.asarray([sp], np.int32), "q": np.zeros((1, 7), np.float32),
                       "sensor_sign": np.ones((1, 8), np.int8), "union_sign": np.ones(1, np.int8)},
            "v3": {"x_index": np.asarray([sp], np.int32), "q": np.zeros((1, 7), np.float32),
                   "sensor_value": np.zeros((1, 8), np.float32), "sensor_value_mask": np.ones((1, 8), bool)},
            "v4": {k: raw[k][vm] for k in ("x_index", "q", "sensor", "value")},
            "boundary": {k: raw[k][bm] for k in ("x_index", "q", "sensor", "grad")},
        }
        counts[name] = {k: len(g["q"]) for k, g in groups.items()}
        for kind, arrays in groups.items():
            for key, values in arrays.items():
                np.save(d/f"{kind}_{key}.npy", values)
    manifest = {"format": "care_h9_v4_training_cache_v1", "status": "COMPLETE", "training_ready": True,
        "source": str(prod), "source_manifest_sha256": sha(prod/"manifest.json"), "counts": counts,
        "files": {str(f.relative_to(cr)): {"bytes": f.stat().st_size, "sha256": sha(f)} for f in cr.rglob("*.npy")}}
    write_json(cr/"manifest.json", manifest)
    cache = deps.data.Cache(cr, verify=True, expected=None)
    deps.pairdata.build_pairs(cache, root/"pairs")
    pairs = deps.pairdata.PairCache(root/"pairs", cache, verify=True)
    return cache, pairs, raw


class CoreTests(unittest.TestCase):
    def test_json_nonfinite_is_null_not_fabricated_zero(self):
        x = json_safe({"a": np.float32(np.nan), "b": torch.tensor([float("inf"), 2.]), "c": np.bool_(True)})
        self.assertEqual(x, {"a": None, "b": [None, 2.], "c": True})
        json.dumps(x, allow_nan=False)

    def test_model_content_digest_ignores_serialization_container(self):
        a = {"weights": torch.arange(6.).reshape(2, 3)}
        self.assertEqual(state_digest(a), state_digest(copy.deepcopy(a)))
        b = copy.deepcopy(a)
        b["weights"][0, 0] += .001
        self.assertNotEqual(state_digest(a), state_digest(b))

    def test_checkpoint_dedup_uses_actual_weights_and_rejects_bad_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rr, pp = root/"r012/formal/R1", root/"ps/formal"
            rr.mkdir(parents=True); pp.mkdir(parents=True)
            torch.manual_seed(11)
            r1, ps = MiniH9(), MiniH9()
            rcp = {"format":"care_h9_scratch50k_r012_v1", "arm":"R1", "completed":True,
                   "initialization":"random_from_scratch", "step":50000, "model_state":r1.state_dict()}
            common = {"cache_identity":"CACHE", "index_identity":"INDEX", "pair_identity":"PAIR",
                      "package_identity":"PKG", "protocol":{"steps":50000}}
            pcp = dict(common, format="care_h9_paired_slope_scratch_v1", completed=True,
                       initialization="random_from_scratch", step=50000, model_state=ps.state_dict())
            torch.save(rcp, rr/"final.pt")
            torch.save(pcp, pp/"final.pt"); torch.save(pcp, pp/"best_val.pt")
            rh, ph, bh = sha(rr/"final.pt"), sha(pp/"final.pt"), sha(pp/"best_val.pt")
            write_json(rr/"run.json", {"status":"COMPLETE", "arm":"R1", "successful_updates":50000, "final_sha256":rh})
            write_json(pp/"run.json", dict(common, status="COMPLETE", initialization="random_from_scratch",
                       successful_updates=50000, best_step=50000, final_sha256=ph, best_val_sha256=bh))
            for filename in ("best_val.pt", "final.pt"):
                write_json(pp/(filename+".sha256.json"), {"sha256":sha(pp/filename)})
            cfg = dict(CFG, r1_sha256=rh, ps_final_sha256=ph, ps_best_sha256=bh, cache_identity="CACHE")
            dep = SimpleNamespace(model=SimpleNamespace(build_model=lambda _:MiniH9()))
            paths = {"r012_root":str(root/"r012"), "ps_root":str(root/"ps")}
            manifest, alias, _ = checkpoint_manifest(paths, cfg, dep)
            self.assertEqual(alias["PS_BEST"], "PS_FINAL")
            self.assertEqual(manifest["PS_BEST"]["state_sha256"], manifest["PS_FINAL"]["state_sha256"])
            write_json(pp/"best_val.pt.sha256.json", {"sha256":"wrong"})
            with self.assertRaisesRegex(ValueError, "guard"):
                checkpoint_manifest(paths, cfg, dep)

    def test_priority_sampler_streaming_and_seed(self):
        data = {"id": np.arange(100)}
        a, b = Reservoir(7, 31), Reservoir(7, 31)
        a.add(data)
        b.add({"id": data["id"][:30]}); b.add({"id": data["id"][30:]})
        np.testing.assert_array_equal(a.rows["id"], b.rows["id"])
        self.assertEqual(len(set(a.rows["id"])), 7)
        self.assertEqual(a.count, 100)

    def test_output_may_not_overlap_source(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for out in (p/"cache", p/"cache/subdir", p):
                with self.assertRaises(ValueError):
                    assert_disjoint_outputs(out, [p/"cache"])
            assert_disjoint_outputs(p/"audit", [p/"cache"])

    def test_alias_context_restores_imports(self):
        previous = sys.modules.get("data")
        sentinel = object()
        with aliases({"data": sentinel}):
            self.assertIs(sys.modules["data"], sentinel)
        self.assertIs(sys.modules.get("data"), previous)

    def test_urdf_ancestry_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x.urdf"
            p.write_text('<robot name="a"><joint name="j1" type="revolute"><parent link="base_link"/><child link="a"/></joint><joint name="j2" type="fixed"><parent link="a"/><child link="b"/></joint></robot>')
            np.testing.assert_array_equal(urdf_masks(p, ("j1", "not_present"), ("a", "b")), [[1, 0], [1, 0]])

    def test_independent_plane_fd_and_signs(self):
        q = torch.zeros(7); n = torch.eye(7)[0]
        qs = q+.01*n, q-.01*n
        oracle = PlaneOracle()
        report, normal = geometry_check(oracle, torch.zeros(3), q, *qs, n, 0, n,
                          -torch.ones(7), torch.ones(7), [0, .01, -.01], CFG)
        self.assertEqual(report["flags"], [])
        self.assertTrue(torch.allclose(normal, n))
        self.assertTrue(all(z["within_tolerance"] for z in report["finite_difference"]))

    def test_bad_normal_is_flagged_not_silently_fixed(self):
        q = torch.zeros(7); n = torch.eye(7)[0]
        report, _ = geometry_check(PlaneOracle(), torch.zeros(3), q, q+.01*n, q-.01*n, -n, 0, n,
                               -torch.ones(7), torch.ones(7), [0, .01, -.01], CFG)
        self.assertIn("STORED_NORMAL_MISMATCH", report["flags"])

    def test_known_field_metrics_and_no_weight_change(self):
        model = PlaneField(); before = state_digest(model.state_dict())
        q = torch.zeros(7); n = torch.eye(7)[0]
        r = point_metrics(model, PlaneOracle(), torch.zeros(3), torch.stack((q, q+.01*n, q-.01*n)),
                          n, 0, .01, n, -torch.ones(7), torch.ones(7), CFG)
        self.assertAlmostEqual(r["pair_slope"], 1., places=5)
        self.assertTrue(r["pair_both_sign_correct"])
        self.assertGreater(r["direction_probe"]["delta_g_m"], 0)
        self.assertEqual(before, state_digest(model.state_dict()))

    def test_zero_gradient_explicit_status(self):
        model = PlaneField(0.)
        q = torch.zeros(7); n = torch.eye(7)[0]
        r = point_metrics(model, PlaneOracle(), torch.zeros(3), torch.stack((q, q+.01*n, q-.01*n)),
                          n, 0, .01, n, -torch.ones(7), torch.ones(7), CFG)
        self.assertEqual(r["direction_probe"]["status"], "SMALL_GRADIENT")
        self.assertIsNone(r["direction_probe"]["delta_g_m"])
        self.assertFalse(r["pair_both_sign_correct"])

    def test_exact_p_no_large_integer_overflow(self):
        self.assertAlmostEqual(exact_p(1, 0), 1.)
        self.assertAlmostEqual(exact_p(0, 2), .5)
        self.assertTrue(0 <= exact_p(72, 376) <= 1)
        self.assertTrue(0 <= exact_p(0, 1963) <= 1)

    def test_merge_duplicate_or_missing_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"solves.rank0.jsonl").write_text('{"case_id":0}\n')
            (p/"solves.rank1.jsonl").write_text('{"case_id":0}\n')
            with self.assertRaises(ValueError):
                load_rows(p, "solves", 2, 2)

    def test_full_1963_merge_and_baseline_gate(self):
        from report import merge
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out = root/"out"; out.mkdir()
            starts = []
            for i in range(1963):
                starts.append({"x":[0.,0.,0.], "q_init":[0.]*7, "sensor":i%8,
                    "x_index":i%118, "group":"S/local_test" if i<683 else "S/uniform_outside"})
            source = root/"starts.jsonl"
            source.write_text("".join(json.dumps(x)+"\n" for x in starts))
            cfg = dict(CFG, starts_sha256=sha(source))
            write_json(root/"manifest.json", {"starts_sha256":sha(source)})
            job = {"protocol":cfg, "paths":{"starts":str(source)}, "models":{},
                   "aliases":{"R1":"R1", "PS_BEST":"PS_FINAL", "PS_FINAL":"PS_FINAL"}}
            write_json(out/"job.json", job)
            np.savez(out/"selected.npz", case_id=np.arange(16))
            write_json(out/"selection.json", {"selected_pairs":16,"selected_sha256":sha(out/"selected.npz")})
            for rank in range(4):
                rows = []
                for i in range(rank, 1963, 4):
                    ok = i < 618 if i < 683 else i-683 < 673
                    m = {"fov_pass":ok, "failure_stage":"PASS" if ok else "FAIL", "solver_ms":1., "gradient_calls":2}
                    rows.append(dict(starts[i], case_id=i, models={"R1":m,"PS_FINAL":m}))
                (out/f"solves.rank{rank}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
                points = []
                for i in range(rank, 16, 4):
                    m = {"nonfinite":False,"f_zero":0.,"pair_slope":1.,"pair_slope_abs_error":0.,
                         "pair_both_sign_correct":True,"boundary_cosine":1.,"boundary_grad_norm":1.,
                         "boundary_inactive_grad_norm":0.,"midpoint_bias":0.,"side_mean_squared_error_rad2":0.,
                         "direction_probe":{"status":"EVALUATED","delta_g_m":.01}}
                    points.append({"case_id":i,"split":i//8,"sensor":i%8,"geometry":{"flags":[]},
                                   "models":{"R1":m,"PS_FINAL":m}})
                (out/f"boundary.rank{rank}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in points))
                for name in ("gradients", "activations"):
                    write_json(out/f"{name}.rank{rank}.json", {"status":"TEST_ONLY"})
                write_json(out/f"mapping.rank{rank}.json", {"workspace_index_arrays_equal":True})
                names = [f"{stem}.rank{rank}.{ext}" for stem,ext in
                         (("solves","jsonl"),("boundary","jsonl"),("gradients","json"),("activations","json"),("mapping","json"))]
                write_json(out/f"rank{rank}.complete.json", {"status":"COMPLETE",
                          "selected_sha256":sha(out/"selected.npz"), "files":{n:sha(out/n) for n in names}})
            with patch("builtins.print"):
                result = merge(out)
            self.assertTrue(result["baseline_reproduced"])
            self.assertEqual(result["cohorts"]["all"]["models"]["R1"]["passed"],1291)
            self.assertEqual(result["cohorts"]["all"]["paired"]["PS_FINAL"]["net"],0)
            self.assertTrue((out/"summary.md").is_file())
            # A recomputed-but-changed R1 baseline cannot silently be accepted.
            p = out/"solves.rank0.jsonl"
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            rows[0]["models"]["R1"]["fov_pass"] = False
            p.write_text("".join(json.dumps(x)+"\n" for x in rows))
            marker_file = out/"rank0.complete.json"; data = json.loads(marker_file.read_text())
            data["files"][p.name] = sha(p); write_json(marker_file, data)
            with patch("builtins.print"), self.assertRaisesRegex(ValueError,"historical"):
                merge(out)
            self.assertEqual(json.loads((out/"report.json").read_text())["status"], "BASELINE_MISMATCH")

    def test_private_parameter_partition_complete(self):
        m = MiniH9()
        groups = parameter_groups(m)
        total = sum(b-a for spans in groups.values() for a, b in spans)
        self.assertEqual(total, sum(p.numel() for p in m.parameters()))
        self.assertEqual(len(groups), 10)

    def test_zero_gradient_cosine_is_missing_not_zero(self):
        report = group_stats({"a": torch.zeros(2), "b": torch.ones(2)}, torch.ones(2),
                             {"a": 1., "b": 1.}, {"shared": [(0, 2)]})
        self.assertIsNone(report["shared"]["component_cosines"]["a"]["b"])

    def test_slurm_spooled_worker_uses_pinned_absolute_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            spool, source = root/"slurm/job1", root/"output/source"
            spool.mkdir(parents=True); source.mkdir(parents=True)
            shutil.copy(HERE/"worker.sh", spool/"slurm_script")
            (source/"cli.py").write_text('import sys\nprint("PINNED_SOURCE_OK", sys.argv[1:])\n')
            job = root/"job.json"; job.write_text('{}')
            env = dict(os.environ, JEA_PY=sys.executable, JEA_SOURCE=str(source), JEA_JOB_JSON=str(job))
            result = subprocess.run(["bash", str(spool/"slurm_script")], env=env, capture_output=True, text=True, check=True)
            self.assertIn("PINNED_SOURCE_OK", result.stdout)

    def test_failed_rank_does_not_leave_other_workers_running(self):
        import cli
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fake = root/"source"; fake.mkdir()
            (fake/"cli.py").write_text('import sys,time\nr=int(sys.argv[-1])\nprint("fake rank",r,flush=True)\nif r==1: raise SystemExit(2)\ntime.sleep(60)\n')
            job = root/"job.json"; write_json(job, {"protocol": {"world": 4}})
            args = type("Args", (), {"job": job})()
            with patch.object(cli, "PACKAGE", fake), patch.object(cli, "check_job"), patch.object(cli, "prepare"), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
                with self.assertRaisesRegex(RuntimeError, "rank"):
                    cli.execute(args)


@unittest.skipUnless(REPO, "set JEA_TEST_REPO to test ACTUAL upstream readers/model/objective")
class UpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deps = Dependencies(Path(REPO))
        cls.training_cfg = json.loads((Path(REPO)/"experiments/hierarchical9_paired_slope_scratch_v1/protocol.json").read_text())

    def test_real_source_schema_and_all_cache_links(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cache, pairs, _ = production_fixture(root, self.deps)
            before = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
            out = root/"audit"; out.mkdir()
            cfg = dict(CFG, pairs_per_split_sensor_radius=2)
            plan = make_plan(out, cfg, self.deps, cache, pairs)
            self.assertEqual(plan["selected_pairs"], 48)
            self.assertTrue(all(s["shortfall"] == 1 for s in plan["strata"]))
            self.assertEqual(before, {p: sha(Path(p)) for p in before})
            with np.load(out/"selected.npz") as z:
                arrays = {k: z[k] for k in z.files}
            arrays["q_plus"][0, 0] += .02
            with self.assertRaisesRegex(ValueError, "mapping"):
                verify_selected_links(arrays, cache, pairs)

    def test_modified_original_shard_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cache, pairs, _ = production_fixture(root, self.deps)
            shard = root/"production/v4_tubes/shard_0001.npz"
            with shard.open("ab") as handle:
                handle.write(b"unexpected mutation")
            out = root/"audit"; out.mkdir()
            with self.assertRaisesRegex(ValueError, "Corrupt production"):
                make_plan(out, CFG, self.deps, cache, pairs)

    def test_current_objective_gradient_decomposition_read_only(self):
        torch.manual_seed(120)
        model = MiniH9().requires_grad_(False)
        before = state_digest(model.state_dict())
        report = diagnose_model(model, fixture_batch(), self.training_cfg, self.deps.objective)
        self.assertLess(report["component_sum_relative_error"], 5e-4)
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(before, state_digest(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_actual_R1_model_input_gradient_and_parameter_backward(self):
        torch.manual_seed(8)
        m = self.deps.model.build_model("R1")
        before = state_digest(m.state_dict())
        b = fixture_batch(24)
        loss, metrics = self.deps.objective.loss_and_metrics(m, b, self.training_cfg, training=True)
        gradients = torch.autograd.grad(loss, tuple(m.parameters()), allow_unused=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))
        self.assertEqual(before, state_digest(m.state_dict()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_wrong_production_alias_fields_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, raw = production_fixture(Path(d), self.deps)
            raw["normal"] = raw.pop("grad")
            with self.assertRaisesRegex(ValueError, "grad"):
                self.deps.pairdata._pairs(raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)

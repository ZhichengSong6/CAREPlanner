"""Deterministic CPU regression tests for the read-only retrospective diagnosis."""
from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import diagnose
import boundary_cpu


def fake_model(pass_status,number):
    fov_margin = 0.01 if pass_status else -0.01
    return {
       "fov_pass":bool(pass_status),"candidate_in_limits":True,
       "candidate_g_m":fov_margin,
       "initial_g_m":-0.1,"zero_g_m":fov_margin,
       "initial_score":-.12 if number%3 else -.01,
       "final_score":.3 if pass_status else -.01,
       "f_zero":0.001,"root_source":"branch_projection_tolerance" if number%2 else "branch_root_not_found",
       "solution_mode":"branch_projection_root_ascent" if number%2 else "branch_best_effort_ascent",
       "q_candidate":[.1]*7,"q_zero":[.05]*7,"q_start":[0]*7,
       "projection_history":[{"iter":0,"score":-.12,"q":[0]*7}],
       "ascent_history":[],"joint_clamp_events":0,
       "failure_stage":"FOV_PASS_NOT_EXECUTION_CERTIFIED" if pass_status else "LEARNED_CANDIDATE_BUT_FOV_FAIL",
       "value_calls":10,"gradient_calls":4,
    }


def make_cases():
    rows=[];i=0
    for name,groups in (
        ("local",[(560,True,True),(58,True,False),(23,False,True),(42,False,False)]),
        ("uniform",[(595,True,True),(78,True,False),(59,False,True),(548,False,False)])):
        for n,old,new in groups:
            for _ in range(n):
                rows.append({"case_id":i,"group":"paper/local_boundary" if name=="local" else "paper/uniform_outside",
                             "x":[.2,.3,.4],"q_init":[0.]*7,"sensor":i%8,
                             "models":{"R1":fake_model(old,i),"R1_AUG":fake_model(new,i)}})
                i+=1
    return rows


def fake_validation(bonus):
    result=[]
    for step in [1]+list(range(1000,50001,1000)):
        heads={}
        for s in ["union"]+[f"s{k}" for k in range(8)]:
            heads[s]={"rmse":.25+bonus,"mae":.2+bonus,
                      "grad_cosine":.6+bonus,"grad_norm":.8+bonus,
                      "sign_accuracy":.7+bonus,"eikonal_loss":.1+bonus}
        diag={"normal_cosine":[.6+bonus]*8,
              "normal_grad_norm":[.9]*8,
              "boundary_abs_field":[.12]*8,
              "first_head_relu_active":[.25]*8,
              "first_head_relu_ever_active_count":[50.]*8}
        result.append({"split":"val","step":step,"loss":.4+bonus,
            "union_objective":.3,"sensor_objective":.2,
            "consistency_loss":.15,"winner_accuracy":.85,
            "heads":heads,"boundary_direction_diagnostic":diag})
    return result


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        heads=[]
        for _ in range(8):
            h=nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Linear(128,1))
            with torch.no_grad():
                h[0].weight.fill_(.2)
                h[0].bias.fill_(.1)
                h[2].weight.fill_(.01)
                h[2].bias.fill_(.0)
            heads.append(h)
        self.sensor_heads=nn.ModuleList(heads)

    def forward_sensor(self,inp,sensor):
        return self.sensor_heads[sensor](inp[:,3:4]).squeeze(1)


class TestCore(unittest.TestCase):
    def test_exact_1963_paired_and_sensor_group_counts(self):
        rows=make_cases()
        self.assertEqual(len(rows),1963)
        self.assertEqual(sum(r["models"]["R1"]["fov_pass"] for r in rows),1291)
        self.assertEqual(sum(r["models"]["R1_AUG"]["fov_pass"] for r in rows),1237)
        cases=[diagnose.extract_case(r) for r in rows]
        report=diagnose.summarize_cases(cases,{"paired":{"aug_only":82,"R1_only":136,"net":-54}})
        self.assertEqual(report["categories"]["lost"]["count"],136)
        self.assertEqual(report["categories"]["gained"]["count"],82)
        self.assertEqual(report["cohorts"]["local"],{
            "n":683,"lost":58,"gained":23,"r1_pass":618,"aug_pass":583})
        self.assertEqual(report["cohorts"]["uniform"],{
            "n":1280,"lost":78,"gained":59,"r1_pass":673,"aug_pass":654})
        self.assertEqual(sum(g["lost"]-g["gained"] for g in report["per_sensor"].values()),54)
        self.assertEqual(len(report["paired_case_ids"]["lost"]),136)
        self.assertEqual(len(report["paired_case_ids"]["gained"]),82)
        self.assertFalse(any(c["outcome"]=="lost" and c["aug_pass"] for c in cases))
        self.assertEqual(len(report["largest_margin_shift_examples"]["lost"]),20)

    def test_nonfinite_is_missing_not_fabricated_zero(self):
        r=diagnose.numeric_stats([float("nan"),None,1.,2.])
        self.assertEqual(r["n"],4)
        self.assertEqual(r["valid"],2)
        self.assertEqual(r["median"],1.5)
        self.assertEqual(diagnose.clean(float("inf")),None)
        self.assertEqual(diagnose.l2([0]*7,[1]*7),math.sqrt(7))
        self.assertIsNone(diagnose.l2([0]*7,None))

    def test_cohort_rejects_unknown_group(self):
        with self.assertRaises(ValueError):
            diagnose.cohort("something_unrecognised")

    def test_duplicate_metrics_step_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"metrics.jsonl"
            rows=fake_validation(0)
            f.write_text("\n".join(map(json.dumps,rows+[rows[0]]))+"\n")
            with self.assertRaisesRegex(ValueError,"Duplicate/invalid"):
                diagnose.metrics_path_data(f)

    def test_exact_original_objective_metric_compare(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/"old.jsonl",Path(td)/"aug.jsonl"
            a.write_text("\n".join(map(json.dumps,fake_validation(0)))+"\n")
            b.write_text("\n".join(map(json.dumps,fake_validation(.01)))+"\n")
            out=diagnose.metrics_comparison(a,b)
            self.assertEqual(out["compared_validation_steps"],51)
            self.assertAlmostEqual(out["last"]["s6.grad_cosine"]["delta"],.01)
            self.assertEqual(len(out["aug_boundary_diagnostic"]["50000"]["first_head_relu_active"]),8)
            self.assertEqual(out["boundary_diagnostic_baseline"],"NOT_RECORDED_IN_ORIGINAL_R1_METRICS")
            summary=diagnose.markdown(diagnose.summarize_cases(
                [diagnose.extract_case(r) for r in make_cases()],
                {"paired":{"aug_only":82,"R1_only":136,"net":-54}}),
                out,"EXACT SAME CPU BOUNDARY POINTS")
            self.assertIn("136",summary)
            self.assertIn("82",summary)
            self.assertIn("EXACT SAME CPU BOUNDARY POINTS",summary)

    def test_full_1963_readonly_input_schema_and_pair_guard(self):
        rows=make_cases()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            r1=root/"r012"
            old=r1/"formal/R1"
            aug=root/"aug"
            new=aug/"formal"
            solver=root/"eval"
            starts_parent=r1/"evaluation_fresh_holdout_v1"
            for p in (old,new,solver,starts_parent):
                p.mkdir(parents=True,exist_ok=True)
            frozen=[{k:entry[k] for k in ("group","x","q_init","sensor")}
                    for entry in rows]
            starts=starts_parent/"starts.jsonl"
            starts.write_text("".join(json.dumps(x)+"\n" for x in frozen))
            (starts_parent/"manifest.json").write_text(json.dumps({
                "starts_sha256":diagnose.EXPECTED_STARTS}))
            for f in (old/"final.pt",new/"final.pt",old/"metrics.jsonl",new/"metrics.jsonl"):
                f.write_bytes(b"trusted test fixture placeholder")
            (old/"run.json").write_text(json.dumps({
                "status":"COMPLETE","arm":"R1","successful_updates":50000,
                "training_stream_updates":50000,"training_stream_sha256":"abc",
                "final_sha256":diagnose.EXPECTED_R1}))
            (new/"run.json").write_text(json.dumps({
                "status":"COMPLETE","variant":diagnose.VARIANT,"successful_updates":50000,
                "baseline_stream_exact_match":True,"training_stream_sha256":"abc",
                "original_r1_final_sha256":diagnose.EXPECTED_R1,
                "final_sha256":"fake_aug_final_sha"}))
            summary={"status":"COMPLETE","starts_sha256":diagnose.EXPECTED_STARTS,
                     "count":1963,"eps_f":.03,"paired":{
                         "aug_only":82,"R1_only":136,"net_gain":-54},
                     "aggregate":{
                         "local":{"R1":{"pass":618},"R1_AUG":{"pass":583}},
                         "uniform":{"R1":{"pass":673},"R1_AUG":{"pass":654}},
                         "all":{"R1":{"pass":1291},"R1_AUG":{"pass":1237}}}}
            (solver/"report.json").write_text(json.dumps(summary))
            (solver/"manifest.json").write_text(json.dumps({
                "starts_sha256":diagnose.EXPECTED_STARTS,
                "r1_sha256":diagnose.EXPECTED_R1,"aug_sha256":"fake_aug_final_sha"}))
            (solver/"complete.json").write_text(json.dumps({
                "status":"COMPLETE","rows":1963}))
            for rank in range(4):
                (solver/f"solves.rank{rank}.jsonl").write_text(
                    "".join(json.dumps(x)+"\n" for x in rows[rank::4]))
            args=SimpleNamespace(r012=r1,aug=aug,solver=solver,starts=starts)
            true_sha=diagnose.sha
            def digest(path):
                path=Path(path)
                if path == starts or path == old/"final.pt":
                    return diagnose.EXPECTED_STARTS if path==starts else diagnose.EXPECTED_R1
                if path == new/"final.pt":
                    return "fake_aug_final_sha"
                return true_sha(path)
            with mock.patch.object(diagnose,"sha",side_effect=digest):
                verified=diagnose.verify_inputs(args)
            self.assertEqual(verified["paired"],
                             {"aug_only":82,"R1_only":136,"net":-54})
            self.assertEqual(len(verified["rows"]),1963)
            self.assertEqual(verified["counts"]["all"]["R1_AUG"],1237)
            # A source-row mismatch must NOT be silently reconciled.
            corrupted=rows.copy()
            corrupted[17]=dict(corrupted[17],sensor=7 if rows[17]["sensor"]!=7 else 6)
            (solver/"solves.rank1.jsonl").write_text(
                "".join(json.dumps(x)+"\n" for x in corrupted[1::4]))
            with mock.patch.object(diagnose,"sha",side_effect=digest):
                with self.assertRaisesRegex(ValueError,"Frozen case 17 mismatch"):
                    diagnose.verify_inputs(args)

    def test_same_point_cpu_gradient_and_first_relu_probe(self):
        m=FakeHead().eval().requires_grad_(False)
        q=np.zeros((32,7),np.float32)
        x=np.zeros((32,3),np.float32)
        n=np.zeros_like(q);n[:,0]=1.
        original={key:v.detach().clone() for key,v in m.state_dict().items()}
        rows,ever=boundary_cpu.evaluate_sensor(m,x,q,n,6)
        self.assertEqual(len(rows),32)
        self.assertEqual(ever,128)
        self.assertTrue(all(v["normal_cosine"] is not None and v["normal_cosine"]>.99
                            for v in rows))
        for k,v in m.state_dict().items():
            self.assertTrue(torch.equal(original[k],v))
        self.assertEqual(len(m.sensor_heads[6][1]._forward_hooks),0)

    def test_no_overwrite_and_integrity_fails_with_missing_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(FileNotFoundError):
                list(diagnose.read_jsonl(root/"missing.jsonl"))
            file=root/"out.json"
            diagnose.write_json(file,{"nonfinite":float("nan")})
            self.assertEqual(diagnose.read_json(file)["nonfinite"],None)
            with self.assertRaises(FileExistsError):
                diagnose.write_json(file,{"overwrite":True})

    def test_failure_stage_and_root_cross_case_metrics(self):
        rows=make_cases()
        row=diagnose.extract_case(rows[560])
        self.assertEqual(row["outcome"],"lost")
        self.assertTrue(row["r1_pass"])
        self.assertFalse(row["aug_pass"])
        self.assertEqual(row["delta_candidate_g_m"],-.02)
        self.assertEqual(row["r1_q_displacement_rad"],row["aug_q_displacement_rad"])


if __name__=="__main__":
    unittest.main(verbosity=2)

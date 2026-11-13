#!/usr/bin/env python3
"""Manual Grid25 production segments 2..7: ORIGINAL global_pool + V3 only.

Original full-V4 solver, sampling and stage code is not modified; full-V4
segments 0,1 and all checkpoints remain immutable. Single segment per Slurm
job, 4 CPU workers, stage shard hashing and resumable execution.

This module adapts the existing frozen full-V4 production runner's per-shard
verification and stage workers, but stops after the original V3 stage.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = REPO / "experiments/hierarchical9_grid25_v4_production_runner_v1"
sys.path.insert(0,str(LEGACY))
import runner as base

prod=base.prod
FORMAT="care_grid25_v3only_segment_production_v1"
SEGMENTS=tuple(range(2,8))
STAGES=("global_pool","v3_labels")
FROZEN_FILES=("runner.py","run.sh","worker.sbatch","submit.sh")
DEFAULT_ROOT=base.DEFAULT_ROOT

# Patch ONLY this Python process' copy of the old, frozen wrapper.  The
# original upstream workers, scientific Config, V3 solver, and disk locations
# for Segment 0/1 are never modified.
ORIGINAL_SUMMARIES = dict(base.STAGE_SUMMARIES)
base.STAGES=STAGES
base.STAGE_SUMMARIES={
    stage:ORIGINAL_SUMMARIES[stage] for stage in STAGES
}
base.FORMAT=FORMAT


def require(ok,msg):
    if not ok: raise RuntimeError(msg)


def sha(path):
    return base.sha(path)


def codes():
    d={name:sha(HERE/name) for name in FROZEN_FILES}
    d["original_v4_production_runner.py"]=sha(LEGACY/"runner.py")
    return d


def segment_out(root,id):
    require(id in SEGMENTS,"Only V3-only segments 2..7 allowed")
    return Path(root).resolve()/"derived/grid25_v3only_production_v1"/f"segment_{id:03d}"


def verify_original_predecessor(root):
    # Original Segment1 was prepared under the old runner and requires all
    # three original stages. Do not call the patched two-stage verifier on it.
    old=base.segment_out(root,1)
    marker=base.read(old/"segment_complete.json")
    require(marker["status"]=="SEGMENT_COMPLETE" and
            marker["segment_id"]==1 and
            marker["format"]=="careplanner_grid25_v4_production_segment_v1",
            "Original Segment1 not complete")
    require(sha(old/"manifest.json")==marker["manifest_sha256"],
            "Original Segment1 manifest changed")
    m=base.read(old/"manifest.json")
    require(m["segment_id"]==1 and
            m["runner_source_hashes"]==base.runner_code_hashes() and
            m["original_algorithms"]==prod.code_identity(),
            "Original Segment1 code identity changed")
    require(sha(old/"x_plan.npz")==marker["x_plan_sha256"],
            "Original Segment1 plan changed")
    n=prod.shard_count(m["counts"]["x"],m["config"]["x_shard_size"])
    for stage in ("global_pool","v3_labels","v4_anchors","v4_tubes"):
        prod.verify_stage(old,stage,n)
    for stage,summary_file in ORIGINAL_SUMMARIES.items():
        require(sha(old/summary_file)==marker["per_stage_summaries"][summary_file],
                "Original Segment1 summary changed: "+stage)
    print("[PREDECESSOR VERIFIED] original segment_001")


def prepare(root,id):
    root=Path(root).resolve()
    require(id in SEGMENTS,"Only V3-only segments 2..7 allowed")
    if id==2:
        verify_original_predecessor(root)
    else:
        predecessor=segment_out(root,id-1)
        verify(predecessor,complete=True)
    target=segment_out(root,id)
    require(not target.exists() and not target.is_symlink(),
            "Segment already prepared; do not overwrite")
    pre_root, report, plan, segments=base.frozen_source(root)
    require(segments[id]["segment_id"]==id,
            "Wrong segment index in original immutable preflight")
    frozen=segments[id]
    local=base.segment_slice(plan,frozen)
    require(len(local["x_index"])==frozen["x_count"] and
            len(local["x_index"])<=1024 and
            np.all(np.diff(local["x_index"])>0) and
            np.array_equal(local["support"],local["valid_count"]>0),
            "Unexpected frozen segment index or Q0 support")
    cfg=prod.Config(**report["frozen_label_config"])
    cfg.validate()
    require(cfg.x_shard_size==128 and cfg.v3_q_per_x==4 and
            cfg.candidate_q_per_x==128,
            "Original V3 sampler/solver budget changed")
    require(int(local["support"].sum())==frozen["supported_x_sensor_pairs"]
            and int(local["support"].sum())*4==frozen["v3_tasks"],
            "Frozen V3 work count changed")
    urdf=REPO/"src/arm_description/urdf/Arm.urdf"
    require(urdf.is_file(),"Missing frozen robot URDF")
    m=dict(
        format=FORMAT,status="PREPARED",training_ready=False,
        segment_id=id,segment_plan=frozen,
        preflight_root=str(pre_root),
        preflight_report_sha256=sha(pre_root/"report.json"),
        preflight_x_plan_sha256=report["x_plan_sha256"],
        preflight_segments_sha256=report["segments_sha256"],
        trainprep=report["trainprep"],
        bank_cache=str(Path(report["trainprep"])/"bank_cache"),
        bank_manifest_sha256=report["provenance"]["bank_manifest_sha256"],
        source_npz_sha256=report["provenance"]["original_npz_sha256"],
        urdf=str(urdf.resolve()),urdf_sha256=sha(urdf),
        config=asdict(cfg),
        solver=asdict(prod.core.SolverConfig()),
        verify=asdict(prod.core.VerifyConfig()),
        original_algorithms=prod.code_identity(),
        v3only_code_identity=codes(),
        counts=dict(x=len(local["x_index"]),
                    train=int((local["split"]==0).sum()),
                    val=int((local["split"]==1).sum()),
                    supported_x_sensor_pairs=int(local["support"].sum()),
                    v3_tasks=int(local["support"].sum())*4),
        science=dict(original_solver=True,original_global_pool=True,
                     no_v4_tubes=True,no_fd_gradient_certification=True,
                     full_grid25_training_ready=False))
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=f".segment_{id:03d}.",dir=target.parent))
    try:
        with (tmp/"x_plan.npz").open("xb") as f:
            np.savez_compressed(f,**local)
            f.flush();os.fsync(f.fileno())
        m["x_plan_sha256"]=sha(tmp/"x_plan.npz")
        with (tmp/"manifest.json").open("x") as f:
            json.dump(m,f,indent=2,sort_keys=True)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.rename(tmp,target)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    print("[V3-ONLY SEGMENT PREPARED]",id,
          "x=",m["counts"]["x"],"v3_tasks=",m["counts"]["v3_tasks"],
          "out=",target,flush=True)


def open_segment(out):
    out=Path(out).resolve()
    m=base.read(out/"manifest.json")
    require(m.get("format")==FORMAT and m.get("status")=="PREPARED"
            and m.get("training_ready") is False
            and m.get("segment_id") in SEGMENTS,
            "Wrong V3-only segment manifest")
    require(m["v3only_code_identity"]==codes() and
            m["original_algorithms"]==prod.code_identity(),
            "V3-only or original V3 solver code changed")
    root=Path(m["preflight_root"]).parents[1]
    pre, report, full, segments=base.frozen_source(root)
    require(sha(pre/"report.json")==m["preflight_report_sha256"] and
            m["preflight_x_plan_sha256"]==report["x_plan_sha256"] and
            m["preflight_segments_sha256"]==report["segments_sha256"],
            "Grid25 preflight identity changed")
    require(m["segment_plan"]==segments[m["segment_id"]],
            "Segment 2..7 frozen plan changed")
    require(sha(out/"x_plan.npz")==m["x_plan_sha256"],
            "Local frozen segment plan changed")
    with np.load(out/"x_plan.npz",allow_pickle=False) as z:
        plan={k:z[k] for k in z.files}
    expected=base.segment_slice(full,m["segment_plan"])
    require(plan.keys()==expected.keys()
            and all(np.array_equal(plan[k],expected[k]) for k in expected),
            "Source Grid25 x identity/split drift")
    require(int(plan["support"].sum())==m["counts"]["supported_x_sensor_pairs"]
            and 4*int(plan["support"].sum())==m["counts"]["v3_tasks"],
            "Planned sensor coverage changed")
    require(sha(m["urdf"])==m["urdf_sha256"]
            and sha(Path(m["bank_cache"])/"bank_manifest.json")==
                m["bank_manifest_sha256"],
            "Robot URDF / bank source identity changed")
    bank=prod.cache_io.BankCache(m["bank_cache"])
    require(bank.manifest["source_sha256"]==m["source_npz_sha256"]
            and bank.valid.shape==(173225,500,8),
            "Q0 mmap source changed")
    return out,m,bank,plan


# The original full-production runner's functions reuse their module-global
# open_segment function to patch existing original modules. This override is
# process-local; no source files or original Segment0/1 are modified.
base.open_segment=open_segment


def verify(out,complete=False):
    out=Path(out).resolve()
    base.verify_segment(out,complete_required=complete)
    marker=out/"segment_complete.json"
    if complete:
        require(marker.is_file() and
                base.read(marker)["status"]=="SEGMENT_COMPLETE",
                "Missing V3-only segment completion marker")
        print("[V3-ONLY SEGMENT VERIFIED]",out.name,flush=True)


def status(out):
    out,m,bank,plan=open_segment(out)
    n=prod.shard_count(len(plan["x_index"]),m["config"]["x_shard_size"])
    progress={}
    for stage in STAGES:
        folder=out/stage
        progress[stage]=dict(
            published_shards=sum(int(
                (folder/f"shard_{k:04d}.npz").is_file() and
                (folder/f"shard_{k:04d}.json").is_file()
            ) for k in range(n)),
            expected_shards=n,
        )
    record=dict(segment=m["segment_id"],status="COMPLETE" if
                (out/"segment_complete.json").is_file() else "IN_PROGRESS",
                counts=m["counts"],progress=progress,
                v4_production=False,training_ready=False)
    print(json.dumps(record,indent=2),flush=True)
    return record


def work(out):
    base.run(out,workers=4)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","work","status","verify"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--segment",type=int,default=2)
    args=p.parse_args()
    if args.mode=="prepare":
        prepare(args.root,args.segment)
    else:
        out=segment_out(args.root,args.segment)
        if args.mode=="work":work(out)
        elif args.mode=="status":status(out)
        else:verify(out,complete=True)


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,KeyError,OSError,FileNotFoundError,
            IndexError,AssertionError) as e:
        raise SystemExit("[STOP] "+str(e))

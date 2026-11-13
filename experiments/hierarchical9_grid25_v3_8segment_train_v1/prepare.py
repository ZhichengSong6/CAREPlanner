#!/usr/bin/env python3
"""Freeze V3 distance-only supervised data from actual Grid25 segments 0..7.

Segment 0/1 are the original finished V3+V4 production; 2..7 are the new
V3-only production. Provenance and spatial split are never recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OLD=REPO/"experiments/hierarchical9_grid25_v4_production_runner_v1"
V3ONLY=REPO/"experiments/hierarchical9_grid25_v3only_production_v1"
TWOSEG=REPO/"experiments/hierarchical9_grid25_multisegment_train_v1"
sys.path.insert(0,str(OLD))
import runner as base

# Import the known working collect_v3 function by full path. "prepare"
# has already been imported by the original Grid25 preflight!
spec=importlib.util.spec_from_file_location(
    "grid25_verified_two_segment_v3_converter",TWOSEG/"prepare.py")
if spec is None or spec.loader is None: raise ImportError("Missing V3 converter")
converter=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=converter
spec.loader.exec_module(converter)
if not callable(getattr(converter,"collect_v3",None)):
    raise ImportError("Missing frozen V3 sample conversion function")

vspec=importlib.util.spec_from_file_location(
    "grid25_v3only_prod_verify_for_training",V3ONLY/"runner.py")
if vspec is None or vspec.loader is None: raise ImportError("Missing V3-only segment runner")
v3only=importlib.util.module_from_spec(vspec)
sys.modules[vspec.name]=v3only
vspec.loader.exec_module(v3only)

FORMAT="care_grid25_v3_eight_segment_h9_train_snapshot_v1"
ROOT=base.DEFAULT_ROOT


def require(ok,msg):
    if not ok: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def destination(root):
    return Path(root).resolve()/"derived/grid25_v3_8segment_train_v1/dataset_000_007"


def original_finished(root,i,full,segments):
    seg=base.segment_out(root,i)
    m=base.read(seg/"manifest.json")
    mark=base.read(seg/"segment_complete.json")
    require(m["segment_id"]==i and
            m["format"]=="careplanner_grid25_v4_production_segment_v1"
            and mark["status"]=="SEGMENT_COMPLETE"
            and mark["segment_id"]==i and
            mark["format"]==m["format"]
            and sha(seg/"manifest.json")==mark["manifest_sha256"]
            and m["runner_source_hashes"]==base.runner_code_hashes()
            and m["original_algorithms"]==base.prod.code_identity(),
            "Original completed V3/V4 segment identity changed")
    require(sha(seg/"x_plan.npz")==mark["x_plan_sha256"] and
            m["segment_plan"]==segments[i],
            "Original full-production preflight plan changed")
    with np.load(seg/"x_plan.npz",allow_pickle=False) as z:
        plan={k:z[k] for k in z.files}
    expected=base.segment_slice(full,segments[i])
    require(plan.keys()==expected.keys() and
            all(np.array_equal(plan[k],expected[k]) for k in plan),
            "Original production train/val x plan changed")
    nsh=base.prod.shard_count(len(plan["x_index"]),m["config"]["x_shard_size"])
    for stage in ("global_pool","v3_labels","v4_anchors","v4_tubes"):
        base.prod.verify_stage(seg,stage,nsh)
    for summary,recorded_sha in mark["per_stage_summaries"].items():
        require(sha(seg/summary)==recorded_sha,
                "Original segment summary changed: "+summary)
    return seg,m,plan


def prepare(root,out):
    root,out=Path(root).resolve(),Path(out).resolve()
    require(not out.exists(),"Eight-segment snapshot already exists; do not overwrite")
    pre,report,full,segments=base.frozen_source(root)
    require(len(segments)>=8,"Grid25 frozen plan lacks segments 0..7")
    cache=base.prod.cache_io.BankCache(Path(report["trainprep"])/"bank_cache")
    parts=[]
    sources={}
    for i in range(8):
        if i<2:
            seg,m,plan=original_finished(root,i,full,segments)
        else:
            seg=v3only.segment_out(root,i)
            v3only.verify(seg,complete=True)
            _,m,_,plan=v3only.open_segment(seg)
        require(m["segment_id"]==i and
                m["segment_plan"]==segments[i] and
                len(plan["x_index"])==1024,
                "Wrong segment index/position count")
        part,hashes=converter.collect_v3(seg,m,cache,plan)
        valid=int(part["valid"].sum())
        summary=base.read(seg/"v3_summary.json")
        require(summary["status"]=="COMPLETE" and
                valid==summary["valid_values"] and
                int(summary["tasks"])==int(m["counts"]["v3_tasks"]),
                "V3 stage and training conversion label counts disagree")
        parts.append(part)
        sources[f"segment_{i:03d}"]={
            "path":str(seg),
            "format":m["format"],
            "manifest_sha256":sha(seg/"manifest.json"),
            "complete_sha256":sha(seg/"segment_complete.json"),
            "v3_summary_sha256":sha(seg/"v3_summary.json"),
            "v3_shard_sha256":hashes,
        }
        print("[V3 TRAINING DATA]",f"segment_{i:03d}",
              "x=",len(plan["x_index"]),"valid=",valid,flush=True)
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    require(len(data["x"])==32768 and
            data["q"].shape==(32768,7) and
            data["valid"].shape==(32768,8) and
            data["target"].shape==(32768,8) and
            data["split"].shape==(32768,),
            "Unexpected total 8-segment sample shape")
    ix=data["x_index"]
    split=data["split"]
    require(np.array_equal(np.unique(ix),
                           np.unique(full["x_index"][:8192])),
            "Eight-segment x selection differs from original frozen full plan")
    require(set(ix[split==0]).isdisjoint(set(ix[split==1])),
            "Training/validation x_index overlap")
    require(np.isfinite(data["x"]).all()
            and np.isfinite(data["q"]).all()
            and np.isfinite(data["target"][data["valid"]]).all(),
            "Nonfinite training input or qualified distance label")
    counts=dict(
        segments=8,x=8192,queries=32768,
        train_x=int(len(np.unique(ix[split==0]))),
        val_x=int(len(np.unique(ix[split==1]))),
        train_queries=int((split==0).sum()),
        val_queries=int((split==1).sum()),
        valid_sensor_labels=int(data["valid"].sum()),
        train_valid_sensor_labels=int(data["valid"][split==0].sum()),
        val_valid_sensor_labels=int(data["valid"][split==1].sum()),
        qualified_per_sensor=[
            int(data["valid"][:,s].sum()) for s in range(8)],
    )
    require(counts["train_x"]>0 and counts["val_x"]>0
            and counts["train_x"]+counts["val_x"]==8192
            and counts["train_valid_sensor_labels"]>0
            and counts["val_valid_sensor_labels"]>0
            and all(x>0 for x in counts["qualified_per_sensor"]),
            "Insufficient V3 labels to train/validate")
    meta=dict(
        format=FORMAT,status="FROZEN",training_ready_full_grid25=False,
        source_segments=sources,
        preflight_report_sha256=sha(pre/"report.json"),
        preflight_x_plan_sha256=report["x_plan_sha256"],
        original_solver_sha256=report["provenance"]["original_algorithm_sha256"],
        converter_sha256=sha(TWOSEG/"prepare.py"),
        v3only_runner_sha256=sha(V3ONLY/"runner.py"),
        counts=counts,
        supervised_targets=dict(
            sensor="V3 value_valid signed best-found continuous FOV boundary distance",
            q_gradient="NOT independently FD-certified, disabled",
            union="only max-of-8 proxy when all 8 V3 labels valid; not exact SDF",
            no_v4_loss=True,
            spatial_split="original buffered Grid25 train/val; no resampling",
        ))
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=".v3_8segment.",dir=out.parent))
    try:
        with (temp/"labels.npz").open("xb") as f:
            np.savez_compressed(f,**data)
            f.flush();os.fsync(f.fileno())
        meta["labels_sha256"]=sha(temp/"labels.npz")
        with (temp/"manifest.json").open("x") as f:
            json.dump(meta,f,indent=2,sort_keys=True)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.rename(temp,out)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print("[EIGHT-SEGMENT V3 SNAPSHOT READY]",out)
    print(json.dumps(counts,indent=2),flush=True)


def verify(out):
    out=Path(out).resolve()
    m=json.loads((out/"manifest.json").read_text())
    require(m.get("format")==FORMAT and m.get("status")=="FROZEN"
            and m.get("training_ready_full_grid25") is False
            and len(m["source_segments"])==8,
            "Unexpected 8-segment dataset identity")
    require(sha(out/"labels.npz")==m["labels_sha256"],
            "Frozen 8-segment label data changed")
    with np.load(out/"labels.npz",allow_pickle=False) as z:
        q=z["q"];ix=z["x_index"];split=z["split"];mask=z["valid"]
        require(len(ix)==32768 and q.shape==(32768,7)
                and mask.shape==(32768,8)
                and int(mask.sum())==m["counts"]["valid_sensor_labels"]
                and set(ix[split==0]).isdisjoint(set(ix[split==1])),
                "Frozen 8-segment V3 snapshot malformed")
    print("[EIGHT-SEGMENT V3 SNAPSHOT VERIFIED]",out)
    print(json.dumps(m["counts"],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","verify"))
    p.add_argument("--root",type=Path,default=ROOT)
    args=p.parse_args()
    out=destination(args.root)
    (prepare(args.root,out) if args.mode=="prepare" else verify(out))


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,OSError,KeyError,FileNotFoundError,
            IndexError,AssertionError) as e:
        raise SystemExit("[STOP] "+str(e))

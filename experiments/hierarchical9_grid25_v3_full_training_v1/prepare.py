#!/usr/bin/env python3
"""Freeze a formal full eligible-Grid25 V3-only H9 distance dataset.

Sources: original complete V3+V4 segments 0,1; original V3-only 2..7;
fullscale V3-only 8..123. Original 75mm buffered spatial train/val split
and selected q0 identities are retained, never resampled.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ORIGINAL=REPO/"experiments/hierarchical9_grid25_v4_production_runner_v1"
TWOSEG=REPO/"experiments/hierarchical9_grid25_multisegment_train_v1"
OLDV3=REPO/"experiments/hierarchical9_grid25_v3only_production_v1"
FULLV3=REPO/"experiments/hierarchical9_grid25_v3only_fullscale_v1"
sys.path.insert(0,str(ORIGINAL))
import runner as base

# Explicit unique module name avoids the preflight "prepare" name collision.
spec=importlib.util.spec_from_file_location(
    "grid25_full_v3_verified_sensor_converter_v1",TWOSEG/"prepare.py")
if spec is None or spec.loader is None:
    raise ImportError("Grid25 V3 converter missing")
converter=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=converter
spec.loader.exec_module(converter)
if not callable(getattr(converter,"collect_v3",None)):
    raise ImportError("Original V3 converter missing")

FORMAT="care_grid25_full_v3only_distance_snapshot_v1"
ROOT=base.DEFAULT_ROOT
TOTAL_X=126201
LAST=123


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for b in iter(lambda:file.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def destination(root):
    return Path(root).resolve()/"derived/grid25_v3_full_training_v1/dataset_segments_000_123"


def verified_segment(root,i,full_plan,segplans):
    if i<2:
        seg=base.segment_out(root,i)
        base.verify_segment(seg,complete_required=True)
        _,m,_,plan=base.open_segment(seg)
    else:
        module=OLDV3 if i<=7 else FULLV3
        seg=(Path(root).resolve()/"derived"/
             ("grid25_v3only_production_v1" if i<=7 else
              "grid25_v3only_fullscale_v1")/f"segment_{i:03d}")
        cmd=["bash",str(module/"run.sh"),"verify",
             "--root",str(root),"--segment",str(i)]
        # Scientific runner modules patch the original preflight globals.
        # Verify in separate interpreter, never import both into this process.
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        m=base.read(seg/"manifest.json")
        with np.load(seg/"x_plan.npz",allow_pickle=False) as z:
            plan={k:z[k] for k in z.files}
    require(m["segment_id"]==i
            and m["segment_plan"]==segplans[i],
            "Source segment frozen plan differs: "+str(i))
    expected=base.segment_slice(full_plan,segplans[i])
    require(plan.keys()==expected.keys() and
            all(np.array_equal(plan[k],expected[k]) for k in expected),
            "Source x plan/split differs from frozen Grid25: "+str(i))
    require(0<len(plan["x_index"])<=1024,
            "Invalid segment capacity")
    return seg,m,plan


def build(root):
    root=Path(root).resolve()
    out=destination(root)
    require(not out.exists() and not out.is_symlink(),
            "Full V3 dataset already exists; refuse overwrite")
    batch=root/"derived/grid25_v3only_fullscale_v1"
    batch_marker=batch/"batch_complete.json"
    require(batch_marker.is_file() and
            base.read(batch_marker)["status"]=="COMPLETE" and
            base.read(batch_marker)["completed_segments"]==116,
            "All remaining 116 V3-only segments must finish before full merge")
    pre,report,full,segplans=base.frozen_source(root)
    require(len(segplans)==124 and len(full["x_index"])==TOTAL_X,
            "Unexpected eligible full Grid25 plan")
    cache=base.prod.cache_io.BankCache(Path(report["trainprep"])/"bank_cache")
    parts=[]
    sources={}
    valid_total=0
    for i in range(124):
        seg,m,plan=verified_segment(root,i,full,segplans)
        part,hashes=converter.collect_v3(seg,m,cache,plan)
        summ=base.read(seg/"v3_summary.json")
        valid=int(part["valid"].sum())
        require(summ["status"]=="COMPLETE" and
                summ["valid_values"]==valid and
                int(summ["tasks"])==int(m["counts"]["v3_tasks"]),
                f"Qualified V3 label counts inconsistent segment {i}")
        parts.append(part)
        valid_total+=valid
        sources[f"segment_{i:03d}"]=dict(
            path=str(seg),format=m["format"],
            segment_manifest_sha256=sha(seg/"manifest.json"),
            segment_complete_sha256=sha(seg/"segment_complete.json"),
            v3_summary_sha256=sha(seg/"v3_summary.json"),
            v3_shard_sha256=hashes)
        if i%8==7 or i==123:
            print("[FULL V3 CONVERSION]",f"through segment {i:03d}",
                  "eligible_x=",sum(len(a["x"])//4 for a in parts),
                  "valid_labels=",valid_total,flush=True)
    data={key:np.concatenate([p[key] for p in parts])
          for key in parts[0]}
    n=TOTAL_X*4
    require(data["x"].shape==(n,3)
            and data["q"].shape==(n,7)
            and data["valid"].shape==data["target"].shape==(n,8)
            and data["x_index"].shape==data["split"].shape==(n,)
            and data["q_slot"].shape==(n,),
            "Formal full Grid25 training snapshot dimensions wrong")
    # Exact row identity for all eligible original training/validation x:
    # every consecutive block of 4 queries refers to the same frozen x.
    require(np.array_equal(data["x_index"][::4],full["x_index"])
            and np.array_equal(data["split"][::4],full["split"])
            and np.array_equal(data["q_slot"].reshape(-1,4),
                               np.tile(np.arange(4,dtype=np.uint8),(TOTAL_X,1))),
            "Full Grid25 frozen x/split/query-slot drift")
    require(np.array_equal(data["split"],np.repeat(full["split"],4))
            and np.isin(data["split"],[0,1]).all(),
            "Full Grid25 training split not original spatial split")
    require(np.isfinite(data["x"]).all()
            and np.isfinite(data["q"]).all()
            and np.isfinite(data["target"][data["valid"]]).all()
            and np.all(data["target"][~data["valid"]]==0),
            "Invalid query or unqualified target in formal full snapshot")
    train=data["split"]==0
    val=data["split"]==1
    counts=dict(
        segments=124,eligible_x=TOTAL_X,queries=n,
        train_x=int(train.sum()//4),val_x=int(val.sum()//4),
        train_queries=int(train.sum()),val_queries=int(val.sum()),
        valid_sensor_labels=int(data["valid"].sum()),
        train_valid_sensor_labels=int(data["valid"][train].sum()),
        val_valid_sensor_labels=int(data["valid"][val].sum()),
        valid_per_sensor=[int(data["valid"][:,s].sum()) for s in range(8)])
    require(counts["train_x"]==109490 and counts["val_x"]==16711
            and counts["train_valid_sensor_labels"]>0
            and counts["val_valid_sensor_labels"]>0
            and all(x>0 for x in counts["valid_per_sensor"]),
            "Frozen train/val counts or qualified sensor coverage unexpected")
    metadata=dict(
        format=FORMAT,status="FROZEN",
        ready_for_v3_distance_training=True,
        ready_for_original_fd_h9_training=False,
        original_frozen_spatial_split=True,
        no_v4_supervision=True,
        gradient_fd_pass=False,
        union_target_exact=False,
        source_segments=sources,
        preflight_report_sha256=sha(pre/"report.json"),
        x_plan_sha256=report["x_plan_sha256"],
        label_converter_sha256=sha(TWOSEG/"prepare.py"),
        source_algorithm_identity=base.prod.code_identity(),
        counts=counts,
        label_definition="Original qualified V3 best-found signed nearest continuous FOV boundary; value_valid only; not globally certified exact SDF",
        union_definition="max of 8 V3 values only when all 8 valid; proxy, not exact union distance")
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=".full_v3.",dir=out.parent))
    try:
        with (tmp/"labels.npz").open("xb") as f:
            np.savez_compressed(f,**data)
            f.flush();os.fsync(f.fileno())
        metadata["labels_sha256"]=sha(tmp/"labels.npz")
        with (tmp/"manifest.json").open("x") as f:
            json.dump(metadata,f,indent=2,sort_keys=True)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.rename(tmp,out)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    print("[FULL GRID25 V3 DISTANCE SNAPSHOT FROZEN]",out,flush=True)
    print(json.dumps(counts,indent=2),flush=True)


def verify(root):
    out=destination(root)
    m=base.read(out/"manifest.json")
    require(m.get("format")==FORMAT and m.get("status")=="FROZEN"
            and m.get("ready_for_v3_distance_training") is True
            and m.get("ready_for_original_fd_h9_training") is False
            and len(m["source_segments"])==124,
            "Full Grid25 V3 dataset identity incorrect")
    require(sha(out/"labels.npz")==m["labels_sha256"],
            "Full Grid25 V3 frozen snapshot digest mismatch")
    with np.load(out/"labels.npz",allow_pickle=False) as z:
        split=np.asarray(z["split"])
        valid=np.asarray(z["valid"])
        require(len(split)==TOTAL_X*4 and valid.shape==(TOTAL_X*4,8)
                and int(valid.sum())==m["counts"]["valid_sensor_labels"]
                and int((split==0).sum())==m["counts"]["train_queries"]
                and int((split==1).sum())==m["counts"]["val_queries"],
                "Full V3 snapshot validation counts changed")
    print("[FULL GRID25 V3 SNAPSHOT VERIFIED]",json.dumps(m["counts"],indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","verify"))
    p.add_argument("--root",type=Path,default=ROOT)
    args=p.parse_args()
    (build(args.root) if args.mode=="prepare" else verify(args.root))


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,OSError,KeyError,FileNotFoundError,
            IndexError,AssertionError) as e:
        raise SystemExit("[STOP] "+str(e))

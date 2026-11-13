#!/usr/bin/env python3
"""Freeze Grid25 segments 0 and 1 into independent V3/V4 supervised snapshots.

Read-only conversion; original segmentation, train/val spatial split, q selection,
V3 solver labels, V4 regular tube local offsets, and all source files stay intact.
"""
from __future__ import annotations
import argparse
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
RUNNER = REPO / "experiments/hierarchical9_grid25_v4_production_runner_v1"
SINGLE = REPO / "experiments/hierarchical9_grid25_v3_v4_joint_train_v1"
sys.path.insert(0,str(RUNNER))
import runner as runner
sys.path.insert(0,str(SINGLE))
import prepare as old_tubes

FORMAT = "care_grid25_multisegment_01_v3_v4_labels_v1"
DEFAULT_ROOT = Path(
    "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):
            h.update(b)
    return h.hexdigest()


def dataset_folder(root):
    return Path(root).resolve()/"derived/grid25_multisegment_train_v1/dataset_segments_000_001"


def frozen_segments(root):
    segs=[]
    for i in (0,1):
        seg=runner.segment_out(root,i)
        runner.verify_segment(seg, complete_required=True)
        _,m,bank,plan = runner.open_segment(seg)
        require(int(m["segment_id"])==i and
                len(plan["x_index"])==1024 and
                len(plan["split"])==1024,
                "Segment must be complete and contain exactly 1024 positions")
        segs.append((seg,m,bank,plan))
    a,b=segs
    require(set(a[3]["x_index"]).isdisjoint(set(b[3]["x_index"])),
            "Segments contain repeated x_index")
    require(sha(a[1]["trainprep"]+"/splits.npz")==sha(b[1]["trainprep"]+"/splits.npz"),
            "Segment split provenance differs")
    return segs


def collect_v3(seg,m,bank,plan):
    xids=np.asarray(plan["x_index"],np.int64)
    split=np.asarray(plan["split"],np.uint8)
    nq=int(m["config"]["v3_q_per_x"])
    require(nq==4 and np.all(np.diff(xids)>0),"Expected canonical 4-q V3 segment")
    n=len(xids)*nq
    x=np.repeat(np.asarray(bank.x[xids],np.float32),nq,axis=0)
    q=np.full((n,7),np.nan,np.float32)
    target=np.zeros((n,8),np.float32)
    valid=np.zeros((n,8),bool)
    seen=np.zeros((n,8),bool)
    filled=np.zeros(n,bool)
    split_rows=np.repeat(split,nq)
    source_hashes={}
    nsh=runner.prod.shard_count(len(xids),int(m["config"]["x_shard_size"]))
    for sh in range(nsh):
        path,_=runner.prod.stage_paths(seg,"v3_labels",sh)
        source_hashes[f"v3_labels/{path.name}"]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            ix=np.asarray(z["x_index"],np.int64)
            slot=np.asarray(z["q_slot"],np.int64)
            sensor=np.asarray(z["sensor"],np.int64)
            actual_split=np.asarray(z["split"],np.uint8)
            query=np.asarray(z["q"],np.float32)
            value=np.asarray(z["new_value"],np.float32)
            value_valid=np.asarray(z["value_valid"],bool)
            reason=np.asarray(z["reason"])
            reference=np.asarray(z["reference_g_m"],np.float32)
        require(query.shape==(len(ix),7) and
                all(len(a)==len(ix) for a in (
                    slot,sensor,actual_split,value,value_valid,reason,reference)),
                "Malformed source V3 shard")
        pos=np.searchsorted(xids,ix)
        require(np.all(pos<len(xids)) and np.array_equal(xids[pos],ix)
                and np.all((slot>=0)&(slot<nq))
                and np.all((sensor>=0)&(sensor<8)),
                "Unexpected V3 x, slot or sensor")
        row=pos*nq+slot
        require(np.array_equal(split_rows[row],actual_split),
                "V3 rows have mismatched frozen spatial split")
        require(np.isfinite(query).all(), "Nonfinite query in V3 source")
        for idx, rid in enumerate(row):
            rid=int(rid)
            s=int(sensor[idx])
            require(not seen[rid,s], "Duplicate sensor label in V3 source")
            seen[rid,s]=True
            if filled[rid]:
                require(np.array_equal(q[rid],query[idx]),
                        "Selected V3 q differs for sensors at same x/slot")
            else:
                filled[rid]=True
                q[rid]=query[idx]
            if value_valid[idx]:
                require(reason[idx]=="VALID" and np.isfinite(value[idx])
                        and np.isfinite(reference[idx])
                        and value[idx]*reference[idx]>=-1e-5,
                        "Original V3 qualified signed label is inconsistent")
                valid[rid,s]=True
                target[rid,s]=value[idx]
    require(filled.all() and np.isfinite(q).all(),"Missing V3 query slots")
    require(int(valid.sum())==
            int(json.loads((seg/"v3_summary.json").read_text())["valid_values"]),
            "Per-shard V3 valid count differs from stage summary")
    require(np.all(valid.any(axis=1)),"V3 query has no supported valid sensor")
    return dict(x=x,q=q,target=target,valid=valid,
                x_index=np.repeat(xids,nq),
                split=split_rows,
                q_slot=np.tile(np.arange(nq,dtype=np.uint8),len(xids))),source_hashes


def prepare(root,out):
    root=Path(root).resolve()
    out=Path(out).resolve()
    require(not out.exists(),"Dataset already exists; no overwrite")
    segs=frozen_segments(root)
    v3=[]
    v4=[]
    fingerprints={}
    for seg,m,bank,plan in segs:
        a,hashes=collect_v3(seg,m,bank,plan)
        v3.append(a)
        v4_segment=[]
        cfg=runner.prod.Config(**m["config"])
        nsh=runner.prod.shard_count(len(plan["x_index"]),cfg.x_shard_size)
        for sh in range(nsh):
            ap,_=runner.prod.stage_paths(seg,"v4_anchors",sh)
            tp,_=runner.prod.stage_paths(seg,"v4_tubes",sh)
            hashes[f"v4_anchors/{ap.name}"]=sha(ap)
            hashes[f"v4_tubes/{tp.name}"]=sha(tp)
            v4_segment.append(old_tubes.convert_shard(bank,seg,sh,plan))
        require(sum(len(a["q"]) for a in v4_segment)==
                json.loads((seg/"v4_summary.json").read_text())["valid_tubes"],
                "Source V4 tubes incomplete")
        v4.append({k:np.concatenate([a[k] for a in v4_segment])
                   for k in v4_segment[0]})
        fingerprints[seg.name]=dict(
            segment_manifest_sha256=sha(seg/"manifest.json"),
            segment_complete_sha256=sha(seg/"segment_complete.json"),
            v3_summary_sha256=sha(seg/"v3_summary.json"),
            v4_summary_sha256=sha(seg/"v4_summary.json"),
            source_shards_sha256=hashes,
        )
        print("[SEGMENT CONVERTED]",seg.name,
              "V3 valid=",int(a["valid"].sum()),
              "V4 tubes=",len(v4[-1]["q"]),flush=True)
    arrays3={k:np.concatenate([z[k] for z in v3]) for k in v3[0]}
    arrays4={k:np.concatenate([z[k] for z in v4]) for k in v4[0]}
    require(len(arrays3["x"])==8192,"Expected 2048 x * 4 V3 query rows")
    require(int(arrays3["valid"].sum())==48455,
            "Segments 0+1 must provide exactly 48455 qualified V3 sensor labels")
    require(len(arrays4["q"])==1958335,
            "Segments 0+1 must provide exactly 1958335 valid V4 local tube rows")
    for data in (arrays3,arrays4):
        tr=set(data["x_index"][data["split"]==0])
        va=set(data["x_index"][data["split"]==1])
        require(tr.isdisjoint(va),"Train/validation spatial leakage")
    sx=set(arrays3["x_index"])
    require(set(arrays4["x_index"]).issubset(sx),
            "V4 includes x absent from V3 snapshot")
    ntrain_x=len(np.unique(arrays3["x_index"][arrays3["split"]==0]))
    nval_x=len(np.unique(arrays3["x_index"][arrays3["split"]==1]))
    require(ntrain_x==1701 and nval_x==347,"Frozen spatial x counts changed")
    counts=dict(
        x=2048,v3_query_rows=8192,
        train_x=ntrain_x,val_x=nval_x,
        train_query_rows=int((arrays3["split"]==0).sum()),
        val_query_rows=int((arrays3["split"]==1).sum()),
        valid_sensor_v3=int(arrays3["valid"].sum()),
        train_valid_sensor_v3=int(arrays3["valid"][arrays3["split"]==0].sum()),
        val_valid_sensor_v3=int(arrays3["valid"][arrays3["split"]==1].sum()),
        v4_tubes=len(arrays4["q"]),
        train_v4_tubes=int((arrays4["split"]==0).sum()),
        val_v4_tubes=int((arrays4["split"]==1).sum()))
    require(counts["train_v4_tubes"]>0 and counts["val_v4_tubes"]>0,
            "Missing tube data for a split")
    meta=dict(
        format=FORMAT,status="FROZEN",training_ready_full_grid25=False,
        original_segments=(0,1),
        original_source_identifiers=fingerprints,
        original_r1_v3_first_train_model="experiments/hierarchical9_scratch50k_r012_v1/model.py",
        counts=counts,
        labels=dict(
            v3="value_valid qualified best-found continuous FOV boundary signed distance",
            v4="REGULAR local tubes, anchor zero and signed offset consistency only",
            gradient_fd_certified=False,union_is_exact_sdf=False,
            train_val="unchanged Grid25 frozen 75mm spatial separation",
        ))
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=".seg01_train.",dir=out.parent))
    try:
        for file,data in (("v3.npz",arrays3),("v4.npz",arrays4)):
            with (temp/file).open("xb") as f:
                np.savez_compressed(f,**data)
                f.flush();os.fsync(f.fileno())
            meta[file+"_sha256"]=sha(temp/file)
        with (temp/"manifest.json").open("x") as f:
            json.dump(meta,f,indent=2,sort_keys=True)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.rename(temp,out)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print("[MULTISEG TRAINING DATA READY]",out)
    print(json.dumps(counts,indent=2),flush=True)


def verify(out):
    out=Path(out).resolve()
    m=json.loads((out/"manifest.json").read_text())
    require(m["format"]==FORMAT and m["status"]=="FROZEN"
            and m["original_segments"]==[0,1],
            "Unexpected frozen multi-segment training snapshot")
    for f in ("v3.npz","v4.npz"):
        require(sha(out/f)==m[f+"_sha256"],
                "Training snapshot SHA mismatch: "+f)
    with np.load(out/"v3.npz",allow_pickle=False) as d:
        ix=d["x_index"];split=d["split"];v=d["valid"]
        require(len(ix)==8192 and int(v.sum())==48455
                and set(ix[split==0]).isdisjoint(set(ix[split==1])),
                "Merged V3 split/data invalid")
    with np.load(out/"v4.npz",allow_pickle=False) as d:
        ix=d["x_index"];split=d["split"]
        require(len(ix)==1958335 and
                set(ix[split==0]).isdisjoint(set(ix[split==1])),
                "Merged V4 split/data invalid")
    print("[MULTISEG TRAINING SNAPSHOT VERIFIED]",out)
    print(json.dumps(m["counts"],indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("prepare","verify"))
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    out=a.out or dataset_folder(a.root)
    (prepare(a.root,out) if a.mode=="prepare" else verify(out))


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,KeyError,OSError,FileNotFoundError) as e:
        raise SystemExit("[STOP] "+str(e))

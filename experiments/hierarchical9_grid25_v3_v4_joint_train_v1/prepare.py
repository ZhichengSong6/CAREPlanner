#!/usr/bin/env python3
"""Freeze verified Grid25 segment_000 V4 tube pairs for first joint H9 training.

All V3 values come from the EXISTING, frozen v3_first_train_v1 snapshot.
The extra V4 term is local zero/signed-offset consistency, NOT certified
global distance or FD-verified q-gradient.
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
sys.path.insert(0, str(RUNNER))
import runner as r

ROOT = Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")
FORMAT = "care_grid25_segment0_v4_local_tube_training_v1"
OFFSETS = np.asarray([0., -0.005, 0.005, -0.01, 0.01, -0.02, 0.02], dtype=np.float32)


def ensure(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for block in iter(lambda: fd.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_source(root):
    seg = r.segment_out(root, 0)
    r.verify_segment(seg, complete_required=True)
    _, manifest, bank, plan = r.open_segment(seg)
    ensure(int(manifest["segment_id"]) == 0, "Only verified segment_000 is in scope")
    v3 = root / "derived/grid25_v3_first_train_v1/dataset_segment_000"
    vm = json.loads((v3/"manifest.json").read_text())
    ensure(vm.get("status") == "FROZEN_DISTANCE_ONLY"
           and vm.get("format") == "care_grid25_segment0_v3_sensor_distance_train_v1"
           and sha(v3/"labels.npz") == vm["labels_sha256"],
           "V3 baseline training snapshot missing or changed")
    ensure(vm["segment_manifest_sha256"] == sha(seg/"manifest.json")
           and vm["segment_complete_sha256"] == sha(seg/"segment_complete.json"),
           "V3 snapshot and V4 source segment provenance diverged")
    return seg, manifest, bank, plan, v3, vm


def identity(ix, sensor, slot):
    # Unique by source Grid25 x, sensor, original Q0 anchor slot.
    return (((np.asarray(ix, dtype=np.int64)*8)
             + np.asarray(sensor, dtype=np.int64))*500
            + np.asarray(slot, dtype=np.int64))


def convert_shard(bank, seg, sh, plan):
    ap, _ = r.prod.stage_paths(seg, "v4_anchors", sh)
    tp, _ = r.prod.stage_paths(seg, "v4_tubes", sh)
    with np.load(ap, allow_pickle=False) as z:
        ai = np.asarray(z["x_index"], np.int64)
        ass = np.asarray(z["sensor"], np.int64)
        asl = np.asarray(z["slot"], np.int64)
        aq = np.asarray(z["q_star"], np.float32)
        an = np.asarray(z["normal"], np.float32)
        regular = np.asarray(z["regular"], bool)
        asplit = np.asarray(z["split"], np.uint8)
    with np.load(tp, allow_pickle=False) as z:
        ti = np.asarray(z["x_index"], np.int64)
        ts = np.asarray(z["sensor"], np.int64)
        tsl = np.asarray(z["source_slot"], np.int64)
        tq = np.asarray(z["q"], np.float32)
        tn = np.asarray(z["grad"], np.float32)
        tv = np.asarray(z["value"], np.float32)
        tgm = np.asarray(z["g_m"], np.float32)
        split = np.asarray(z["split"], np.uint8)
    ensure(len(ti)>0 and tq.shape==(len(ti),7) and
           tn.shape==(len(ti),7) and aq.shape==(len(ai),7) and
           an.shape==(len(ai),7),
           "Malformed original V4 anchors or tubes")
    ensure(np.all((ts>=0)&(ts<8)) and np.all((tsl>=0)&(tsl<500)),
           "Unexpected source sensor or slot")
    keys = identity(ai, ass, asl)
    tkeys = identity(ti, ts, tsl)
    ensure(len(np.unique(keys)) == len(keys), "Nonunique original V4 anchors")
    order = np.argsort(keys)
    ordered = keys[order]
    pos = np.searchsorted(ordered, tkeys)
    ensure(np.all(pos < len(ordered)) and
           np.array_equal(ordered[pos], tkeys),
           "V4 tube lacks matching source anchor")
    src = order[pos]
    center = aq[src]
    normals = an[src]
    ensure(regular[src].all(), "Tube must originate from a regular anchor")
    ensure(np.array_equal(split, asplit[src]), "V4 tube/anchor split mismatch")
    ensure(np.isfinite(tq).all() and np.isfinite(tn).all()
           and np.isfinite(center).all() and np.isfinite(normals).all()
           and np.isfinite(tv).all() and np.isfinite(tgm).all(),
           "Nonfinite V4 tube or its source anchor")
    ensure(np.all(np.min(np.abs(tv[:,None]-OFFSETS[None,:]),axis=1)<5e-6),
           "Unexpected V4 local signed offset")
    ensure(np.allclose(tn,normals,rtol=1e-4,atol=1e-4),
           "Tube normal/source anchor disagreement")
    # Source is exactly q_star + offset * original local normal.
    residual = np.abs(tq - (center + tv[:,None]*normals)).max()
    ensure(float(residual) < 1e-4,
           "Tube perturbation does not correspond to source local normal")
    zmask = np.abs(tv) < 1e-7
    ensure(np.all(np.abs(tgm[zmask]) <= 2e-6 + 1e-7) and
           np.all(tgm[~zmask]*tv[~zmask] > 0),
           "Original tube FOV sign gate failed")
    local_pos = np.searchsorted(plan["x_index"], ti)
    ensure(np.all(local_pos < len(plan["x_index"])) and
           np.array_equal(plan["x_index"][local_pos], ti) and
           np.array_equal(plan["split"][local_pos], split),
           "Tube x/split not in frozen segment")
    return dict(
        x=np.asarray(bank.x[ti], np.float32),
        center=center,
        q=tq,
        sensor=ts.astype(np.uint8),
        offset=tv,
        split=split,
        x_index=ti,
    )


def prepare(args):
    root = args.root.resolve()
    seg, sm, bank, plan, v3, vm = load_source(root)
    out = (args.out or root/"derived/grid25_v3_v4_joint_train_v1/tube_snapshot_000").resolve()
    ensure(not out.exists(), "Tube snapshot already exists; no overwrite")
    cfg=r.prod.Config(**sm["config"])
    count=r.prod.shard_count(len(plan["x_index"]),cfg.x_shard_size)
    arrays=[]
    hashed={}
    for sh in range(count):
        ap,_ = r.prod.stage_paths(seg, "v4_anchors", sh)
        tp,_ = r.prod.stage_paths(seg, "v4_tubes", sh)
        hashed[ap.name]=sha(ap)
        hashed[tp.name]=sha(tp)
        arrays.append(convert_shard(bank,seg,sh,plan))
        print(f"[TUBE SHARD] {sh+1}/{count} rows={len(arrays[-1]['q'])}",flush=True)
    combined={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    v4 = json.loads((seg/"v4_summary.json").read_text())
    ensure(len(combined["q"]) == int(v4["valid_tubes"]) == 943493,
           "V4 tube rows no longer match verified segment 0")
    ensure(len(combined["x"]) == len(combined["q"]) and
           len(np.unique(combined["x_index"][combined["split"]==0])) <= 845 and
           len(np.unique(combined["x_index"][combined["split"]==1])) <= 179,
           "Unexpected number of tube positions")
    ensure(len(set(combined["x_index"][combined["split"]==0]) &
                   set(combined["x_index"][combined["split"]==1])) == 0,
           "Train/Val spatial leakage")
    counts=dict(
        tubes=int(len(combined["q"])),
        train_tubes=int((combined["split"]==0).sum()),
        val_tubes=int((combined["split"]==1).sum()),
        by_sensor=[int((combined["sensor"]==s).sum()) for s in range(8)],
        zero_offset=int((combined["offset"]==0).sum()),
        positive_offset=int((combined["offset"]>0).sum()),
        negative_offset=int((combined["offset"]<0).sum()),
    )
    ensure(counts["train_tubes"] > 0 and counts["val_tubes"]>0,
           "No valid tube examples in a frozen train/val partition")
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=".tube_snapshot_000.",dir=out.parent))
    try:
        with (tmp/"tubes.npz").open("xb") as f:
            np.savez_compressed(f,**combined)
            f.flush();os.fsync(f.fileno())
        meta=dict(
            format=FORMAT,status="FROZEN_LOCAL_TUBES",
            training_ready_full_grid25=False,
            segment=str(seg),
            segment_manifest_sha256=sha(seg/"manifest.json"),
            segment_complete_sha256=sha(seg/"segment_complete.json"),
            source_shards_sha256=hashed,
            v3_snapshot=str(v3),
            v3_labels_sha256=vm["labels_sha256"],
            tubes_sha256=sha(tmp/"tubes.npz"),
            counts=counts,
            supervision=dict(
                v3="Per-sensor signed continuous-boundary value, mask only VALID",
                v4="Local zero and signed-offset consistency near REGULAR boundary anchor",
                union="Max-of-all-eight V3 proxy only when all eight value labels valid",
                not_fd_certified=True,not_global_nearest_boundary=True,
                no_gradient_target=True,no_global_tube_sdf=True,
                original_spatial_split_preserved=True,
            ),
        )
        with (tmp/"manifest.json").open("x") as f:
            json.dump(meta,f,indent=2,sort_keys=True)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.rename(tmp,out)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    print("[JOINT H9 TUBE SNAPSHOT READY]",out)
    print(json.dumps(counts,indent=2),flush=True)


def verify(args):
    root=args.root.resolve()
    out=(args.out or root/"derived/grid25_v3_v4_joint_train_v1/tube_snapshot_000").resolve()
    meta=json.loads((out/"manifest.json").read_text())
    ensure(meta["format"]==FORMAT and meta["status"]=="FROZEN_LOCAL_TUBES",
           "Wrong V4 training tube snapshot")
    ensure(sha(out/"tubes.npz")==meta["tubes_sha256"],
           "Tube snapshot sha mismatch")
    v3=Path(meta["v3_snapshot"])
    v3meta=json.loads((v3/"manifest.json").read_text())
    ensure(v3meta["labels_sha256"]==meta["v3_labels_sha256"] and
           sha(v3/"labels.npz")==meta["v3_labels_sha256"],
           "Source V3 training snapshot changed")
    with np.load(out/"tubes.npz",allow_pickle=False) as z:
        n=len(z["q"])
        ensure(z["x"].shape==(n,3) and z["q"].shape==(n,7)
               and z["center"].shape==(n,7)
               and n==meta["counts"]["tubes"],
               "Tube snapshot shapes changed")
        ensure(np.isfinite(z["q"]).all()
               and np.isfinite(z["center"]).all(),
               "Snapshot contains nonfinite q")
        split=z["split"]; ix=z["x_index"]
        ensure(not (set(ix[split==0]) & set(ix[split==1])),
               "Tube train/val split overlap")
        ensure(int((split==0).sum())==meta["counts"]["train_tubes"]
               and int((split==1).sum())==meta["counts"]["val_tubes"],
               "Tube split row counts changed")
    print("[JOINT H9 TUBE SNAPSHOT VERIFIED]",out)
    print(json.dumps(meta["counts"],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","verify"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    (prepare if args.mode=="prepare" else verify)(args)


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,OSError,KeyError,FileNotFoundError) as e:
        raise SystemExit("[STOP] "+str(e))

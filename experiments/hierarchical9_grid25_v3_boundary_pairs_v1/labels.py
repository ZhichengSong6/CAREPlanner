#!/usr/bin/env python3
"""Third-round Grid25 FOV boundary-pair augmentation (NO LOS).

Use only existing qualified V3 continuous-boundary q_star endpoints, perturb
along an analytic active-joint FOV normal to opposite FOV sides, then relabel
BOTH queries with the *original* V3 solve_attempts/screen_candidates/reference_gate.
No FOV-margin targets and no invented or unverified q-gradients.

Fixed 256 train + 64 validation x (distinct spatial split), max one original
qualified anchor per (x,sensor), two new sensor-specific signed-distance rows.
Original 124 segments, full training dataset and checkpoints remain immutable.
CPU process-pool task completion is restartable and individually atomic.
"""
from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/"experiments/hierarchical9_full_dataset_v4_v1"))
import prod_common as prod  # noqa: E402
from analytic_geometry import AnalyticGeometry  # noqa: E402

ROOT=Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")
FORMAT="care_grid25_v3_boundary_pairs_relabel_v1"
TRAIN_X=256
VAL_X=64
SEED=260930
BAND_MAX=0.20
STEPS=(0.04,0.06,0.08)

_WORKER=None


def require(ok,msg):
    if not ok:raise RuntimeError("[BOUNDARY PAIRS STOP] "+msg)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def base(root):
    return Path(root).resolve()/"derived/grid25_v3_boundary_pairs_v1"


def dataset(root):
    return Path(root).resolve()/"derived/grid25_v3_full_training_v1/dataset_segments_000_123"


def segment(root,i):
    folder=("grid25_v4_production_v1" if i<2 else
            "grid25_v3only_production_v1" if i<=7 else
            "grid25_v3only_fullscale_v1")
    return Path(root)/"derived"/folder/f"segment_{i:03d}"


def write_json(path,obj):
    with Path(path).open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())


def write_npz(path,data):
    with Path(path).open("xb") as f:
        np.savez_compressed(f,**data)
        f.flush();os.fsync(f.fileno())


def load_plan(root):
    out=base(root)
    m=read(out/"plan.json")
    require(m.get("format")==FORMAT
            and m.get("status")=="PREPARED"
            and m.get("source_code_sha256")==digest(__file__)
            and m.get("plan_npz_sha256")==digest(out/"tasks.npz"),
            "Third-round frozen task plan/source changed")
    with np.load(out/"tasks.npz",allow_pickle=False) as z:
        a={k:np.asarray(z[k]) for k in z.files}
    require(a["q_star"].shape==(len(a["sensor"]),7)
            and len(a["sensor"])==m["planned_anchor_tasks"]
            and np.isin(a["split"],[0,1]).all(),
            "Corrupt immutable augmentation task plan")
    return out,m,a


def prepare(root):
    root=Path(root).resolve()
    out=base(root)
    require(not out.exists(),"Boundary pairs output already exists; do not overwrite")
    source=dataset(root)
    frozen=read(source/"manifest.json")
    require(frozen["format"]=="care_grid25_full_v3only_distance_snapshot_v1"
            and frozen["status"]=="FROZEN"
            and frozen["counts"]["train_x"]==109490
            and frozen["counts"]["val_x"]==16711,
            "Wrong frozen original Grid25 training identity")
    require(digest(source/"labels.npz")==frozen["labels_sha256"],
            "Original full V3 snapshot SHA changed")
    with np.load(source/"labels.npz",allow_pickle=False) as z:
        xids=np.asarray(z["x_index"][::4],np.int64)
        split=np.asarray(z["split"][::4],np.uint8)
    require(len(xids)==126201 and np.all(np.diff(xids)>0),
            "Unexpected original frozen x ordering")
    rng=np.random.default_rng(SEED)
    train_x=np.sort(rng.choice(xids[split==0],TRAIN_X,replace=False))
    val_x=np.sort(rng.choice(xids[split==1],VAL_X,replace=False))
    require(not (set(train_x.tolist()) & set(val_x.tolist())),
            "Spatial train/validation leakage")
    targets={int(x):0 for x in train_x}
    targets.update({int(x):1 for x in val_x})

    pre=Path(root)/"derived/grid25_v4_production_preflight_v1"
    segplans=read(pre/"segments.json")
    require(len(segplans)==124,"Original 124 segment plan missing")
    ends=[int(row["stop"]) for row in segplans]
    requested={}
    for xi,sp in targets.items():
        pos=int(np.searchsorted(xids,xi))
        require(pos<len(xids) and int(xids[pos])==xi,
                "Selected x not present in frozen Grid25 snapshot")
        segment_id=bisect.bisect_right(ends,pos)
        require(0<=segment_id<124 and
                int(segplans[segment_id]["start"])<=pos<int(segplans[segment_id]["stop"]),
                "x belongs to no source production segment")
        local=pos-int(segplans[segment_id]["start"])
        shard=local//128
        requested.setdefault((segment_id,shard),{})[xi]=sp

    candidate={}
    sources={}
    for (segid,sh),need in sorted(requested.items()):
        source_file=segment(root,segid)/"v3_labels"/f"shard_{sh:04d}.npz"
        sidecar=source_file.with_suffix(".json")
        require(source_file.is_file() and sidecar.is_file(),
                "Missing immutable V3 source shard: "+str(source_file))
        source_sha=digest(source_file)
        require(read(sidecar)["sha256"]==source_sha,
                "Original V3 source shard checksum mismatch")
        sources[f"seg_{segid:03d}_shard_{sh:04d}"]=source_sha
        with np.load(source_file,allow_pickle=False) as z:
            ix=np.asarray(z["x_index"],np.int64)
            sid=np.asarray(z["sensor"],np.int64)
            qs=np.asarray(z["q_star"],np.float32)
            valid=np.asarray(z["value_valid"],bool)
            distances=np.asarray(z["distance_rad"],np.float32)
            qslots=np.asarray(z["q_slot"],np.int64)
        require(qs.shape==(len(ix),7)
                and all(len(v)==len(ix) for v in
                        (sid,valid,distances,qslots)),
                "Malformed original V3 qualified q_star source")
        for j in range(len(ix)):
            xi=int(ix[j]);sensor_id=int(sid[j])
            if xi not in need or not valid[j]:
                continue
            key=(xi,sensor_id)
            if not np.isfinite(qs[j]).all() or not np.isfinite(distances[j]):
                continue
            # One deterministic, qualified continuous FOV boundary anchor
            # per (x,sensor), choosing the smallest original signed distance.
            row=dict(x_index=xi,split=need[xi],sensor=sensor_id,
                     q_star=qs[j].copy(),source_q_slot=int(qslots[j]),
                     original_distance=float(distances[j]))
            old=candidate.get(key)
            if old is None or row["original_distance"]<old["original_distance"]:
                candidate[key]=row

    items=sorted(candidate.values(),key=lambda v:(v["x_index"],v["sensor"]))
    require(len(items)>=1000 and
            sum(v["split"]==0 for v in items)>500 and
            sum(v["split"]==1 for v in items)>100,
            "Insufficient original qualified V3 q_star anchors for planned paired augmentation")
    tasks=dict(
        x_index=np.asarray([v["x_index"] for v in items],np.int64),
        split=np.asarray([v["split"] for v in items],np.uint8),
        sensor=np.asarray([v["sensor"] for v in items],np.uint8),
        q_star=np.asarray([v["q_star"] for v in items],np.float32),
        source_q_slot=np.asarray([v["source_q_slot"] for v in items],np.uint8),
        original_distance=np.asarray([v["original_distance"] for v in items],np.float32),
    )
    bank_path=Path(root)/"derived/grid25_v4_production_preflight_v1/report.json"
    preflight=read(bank_path)
    cache=Path(preflight["trainprep"])/"bank_cache"
    require((cache/"bank_manifest.json").is_file(),"Original q0 bank cache missing")
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=".v3_boundary_pairs.",dir=out.parent))
    try:
        write_npz(temp/"tasks.npz",tasks)
        plan=dict(
            format=FORMAT,status="PREPARED",
            source_code_sha256=digest(__file__),
            source_dataset_manifest_sha256=digest(source/"manifest.json"),
            source_dataset_labels_sha256=frozen["labels_sha256"],
            source_segments_sha256=digest(pre/"segments.json"),
            source_v3_shards_sha256=sources,
            original_bank_cache=str(cache.resolve()),
            original_bank_manifest_sha256=digest(cache/"bank_manifest.json"),
            original_urdf=str((REPO/"src/arm_description/urdf/Arm.urdf").resolve()),
            original_urdf_sha256=digest(REPO/"src/arm_description/urdf/Arm.urdf"),
            original_solver=asdict(prod.core.SolverConfig()),
            original_verify=asdict(prod.core.VerifyConfig()),
            seed=SEED,train_x=TRAIN_X,val_x=VAL_X,
            epsilon_candidates_rad=list(STEPS),
            boundary_max_distance_rad=BAND_MAX,
            planned_anchor_tasks=len(items),
            planned_train_anchors=int((tasks["split"]==0).sum()),
            planned_val_anchors=int((tasks["split"]==1).sum()),
            plan_npz_sha256=digest(temp/"tasks.npz"),
            no_los=True,no_fd_gradient_supervision=True,
        )
        write_json(temp/"plan.json",plan)
        os.rename(temp,out)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print("[BOUNDARY PAIRS PLAN FROZEN]",len(items),
          "train_anchors=",plan["planned_train_anchors"],
          "val_anchors=",plan["planned_val_anchors"],flush=True)


def task_path(out,i):
    return out/"tasks"/f"task_{i:05d}"


def _initialize(bank_path,urdf,solver,verify):
    global _WORKER
    bank=prod.cache_io.BankCache(bank_path)
    oracle=prod.RepoOracle(REPO,urdf,"cpu",
                           joint_names=bank.joints,sensor_frames=bank.sensors)
    _WORKER=(bank,oracle,prod.core.SolverConfig(**solver),
             prod.core.VerifyConfig(**verify))


def _compute(item):
    i,xi,split,s,q_star=item
    bank,oracle,solver,vc=_WORKER
    x=np.asarray(bank.x[int(xi)],np.float64)
    mask=np.asarray(bank.masks[int(s)],np.float64)
    lo=np.asarray(bank.lo,np.float64)
    hi=np.asarray(bank.hi,np.float64)
    anchor=np.asarray(q_star,np.float64)
    geo=AnalyticGeometry(oracle,x,int(s))
    h,j=geo(anchor)
    face=int(np.argmin(h))
    normal=j[face]*mask
    norm=float(np.linalg.norm(normal))
    reason="NO_VALID_OPPOSITE_SIDES"
    candidates=None
    if (np.isfinite(h).all() and np.isfinite(j).all() and
            abs(float(h[face]))<=solver.boundary_tol_m and norm>solver.min_normal):
        normal/=norm
        for eps in STEPS:
            positive=anchor+eps*normal
            negative=anchor-eps*normal
            if (np.any(positive<lo) or np.any(positive>hi)
                    or np.any(negative<lo) or np.any(negative>hi)):
                continue
            gp=float(np.min(geo(positive)[0]))
            gn=float(np.min(geo(negative)[0]))
            if gp>1e-4 and gn< -1e-4:
                candidates=[positive,negative]
                break
    if candidates is None:
        return i,dict(status="SKIP",reason=reason,
                      x_index=xi,sensor=s,split=split),None

    values=[]
    records=[]
    for side,q in zip((1,-1),candidates):
        ref=float(oracle.reference_margins(x,q[None,:])[0,int(s)])
        if (not np.isfinite(ref)
                or not ((side>0 and ref>1e-4) or (side<0 and ref< -1e-4))):
            return i,dict(status="SKIP",reason="REFERENCE_SIDE_MISMATCH",
                          x_index=xi,sensor=s,split=split),None
        bank_q=bank.sensor_bank(xi,s)
        attempts=prod.core.solve_attempts(
            q,bank_q,mask,lo,hi,geo,solver)
        labeled=prod.core.screen_candidates(
            q,mask,lo,hi,geo,attempts,solver,vc)
        labeled=prod.v3.reference_gate(
            oracle,x,s,q,labeled,solver)
        value=float(labeled["value"])
        if not (labeled["value_valid"] and
                np.isfinite(value) and
                value*ref>=0 and
                0<abs(value)<=BAND_MAX):
            return i,dict(status="SKIP",reason="UNQUALIFIED_V3_REPROJECTION",
                          x_index=xi,sensor=s,split=split,
                          failed_side=side),None
        values.append(value)
        records.append((q.astype(np.float32),ref))
    require(values[0]>0 and values[1]<0,
            "Relabeled pair has wrong signed distance orientation")
    qs=np.stack([r[0] for r in records]).astype(np.float32)
    target=np.zeros((2,8),np.float32)
    valid=np.zeros((2,8),bool)
    target[:,s]=np.asarray(values,np.float32)
    valid[:,s]=True
    data=dict(
        x=np.repeat(np.asarray(x,np.float32)[None,:],2,axis=0),
        q=qs,target=target,valid=valid,
        split=np.full(2,split,np.uint8),
        x_index=np.full(2,xi,np.int64),
        sensor=np.full(2,s,np.uint8),
        g_reference=np.asarray([r[1] for r in records],np.float32),
    )
    meta=dict(status="QUALIFIED_PAIR",reason="VALID",
              x_index=xi,sensor=s,split=split,
              signed_distances=[float(v) for v in values],
              reference_margins=[float(v[1]) for v in records])
    return i,meta,data


def commit_task(out,i,meta,data):
    final=task_path(out,i)
    require(not final.exists(),"Task already exists: "+str(final))
    final.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=".task.",dir=final.parent))
    try:
        if data is not None:
            write_npz(temp/"labels.npz",data)
            meta["labels_sha256"]=digest(temp/"labels.npz")
        write_json(temp/"complete.json",meta)
        os.rename(temp,final)
    finally:
        if temp.exists():shutil.rmtree(temp)


def work(root,workers):
    require(1<=workers<=4,"Use at most 4 CPU workers")
    out,m,tasks=load_plan(root)
    require(digest(m["original_urdf"])==m["original_urdf_sha256"]
            and digest(Path(m["original_bank_cache"])/"bank_manifest.json")==m["original_bank_manifest_sha256"],
            "Original geometry/bank source changed")
    require(not (out/"augmentation_manifest.json").exists(),
            "Augmented data already merged; refuse additional writes")
    jobs=[]
    for i in range(len(tasks["sensor"])):
        folder=task_path(out,i)
        if folder.is_dir():
            meta=read(folder/"complete.json")
            require(meta["x_index"]==int(tasks["x_index"][i])
                    and meta["sensor"]==int(tasks["sensor"][i])
                    and meta["split"]==int(tasks["split"][i])
                    and (meta["status"]=="SKIP" or
                         (meta["status"]=="QUALIFIED_PAIR" and
                          digest(folder/"labels.npz")==meta["labels_sha256"])),
                    f"Previous task {i} has changed")
            continue
        require(not folder.exists(),f"Unexpected path at task {i}")
        jobs.append((i,int(tasks["x_index"][i]),int(tasks["split"][i]),
                     int(tasks["sensor"][i]),tasks["q_star"][i].tolist()))
    print("[BOUNDARY PAIRS WORK]",len(jobs),"pending of",
          len(tasks["sensor"]),"workers",workers,flush=True)
    if not jobs:
        return
    successes=0
    begin=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=get_context("spawn"),
                             initializer=_initialize,
                             initargs=(m["original_bank_cache"],m["original_urdf"],
                                       m["original_solver"],m["original_verify"])) as pool:
        futures=[pool.submit(_compute,j) for j in jobs]
        for completed, future in enumerate(as_completed(futures),1):
            i,meta,data=future.result()
            commit_task(out,i,meta,data)
            if data is not None:successes+=1
            if completed%64==0 or completed==len(jobs):
                print("[BOUNDARY PAIRS WORK PROGRESS]",completed,"/",len(jobs),
                      "qualified_pairs_this_run=",successes,
                      "elapsed_s=",round(time.monotonic()-begin,1),flush=True)
    print("[BOUNDARY PAIRS WORK COMPLETE] newly_qualified_pairs=",successes,flush=True)


def merge(root):
    out,m,tasks=load_plan(root)
    dst=out/"augmentation_manifest.json"
    require(not dst.exists(),"Frozen augmentation already merged; refuse overwrite")
    parts={0:[],1:[]}
    reasons={}
    counts={0:0,1:0}
    for i in range(len(tasks["sensor"])):
        folder=task_path(out,i)
        require(folder.is_dir(),f"Task {i} has not completed")
        meta=read(folder/"complete.json")
        require(meta["x_index"]==int(tasks["x_index"][i])
                and meta["sensor"]==int(tasks["sensor"][i])
                and meta["split"]==int(tasks["split"][i]),
                f"Task {i} provenance differs")
        if meta["status"]=="SKIP":
            reasons[meta["reason"]]=reasons.get(meta["reason"],0)+1
            continue
        require(meta["status"]=="QUALIFIED_PAIR" and
                digest(folder/"labels.npz")==meta["labels_sha256"],
                f"Task {i} checksum invalid")
        with np.load(folder/"labels.npz",allow_pickle=False) as z:
            arrays={k:np.asarray(z[k]) for k in z.files}
        sid=int(tasks["sensor"][i]);sp=int(tasks["split"][i])
        require(arrays["q"].shape==(2,7)
                and arrays["target"].shape==(2,8)
                and arrays["valid"].shape==(2,8)
                and arrays["valid"][:,sid].all()
                and np.array_equal(arrays["valid"].sum(axis=1),[1,1])
                and arrays["target"][0,sid]>0
                and arrays["target"][1,sid]<0
                and np.max(np.abs(arrays["target"]))<=BAND_MAX
                and np.array_equal(arrays["split"],[sp,sp])
                and np.array_equal(arrays["x_index"],[int(tasks["x_index"][i])]*2),
                f"Task {i} is not a qualified V3 signed two-sided pair")
        parts[sp].append(arrays)
        counts[sp]+=1
    # No labels ever move between the original frozen x split.
    require(counts[0]>=64 and counts[1]>=16,
            "Too few qualified opposite-side labels; do not train")
    keys=("x","q","target","valid","split","x_index","sensor","g_reference")
    merged={}
    for sp in (0,1):
        merged[sp]={k:np.concatenate([p[k] for p in parts[sp]],axis=0)
                    for k in keys}
    require(not (set(merged[0]["x_index"].tolist())
                 & set(merged[1]["x_index"].tolist())),
            "Spatial train/val leakage in augmentation")
    for sp,name in ((0,"train_pairs.npz"),(1,"val_pairs.npz")):
        path=out/name
        if path.exists():
            # Safe after an interrupted merge: never overwrite, only accept
            # the exact previously constructed immutable data.
            with np.load(path,allow_pickle=False) as z:
                prior={k:np.asarray(z[k]) for k in z.files}
            require(prior.keys()==merged[sp].keys() and
                    all(np.array_equal(prior[k],merged[sp][k])
                        for k in merged[sp]),
                    "Interrupted merge created different data: "+str(path))
        else:
            write_npz(path,merged[sp])
    report=dict(
        format=FORMAT,status="FROZEN",
        plan_sha256=m["plan_npz_sha256"],
        source_dataset_labels_sha256=m["source_dataset_labels_sha256"],
        source_code_sha256=m["source_code_sha256"],
        train_pairs_sha256=digest(out/"train_pairs.npz"),
        val_pairs_sha256=digest(out/"val_pairs.npz"),
        qualified_train_pairs=counts[0],
        qualified_val_pairs=counts[1],
        train_rows=counts[0]*2,
        val_rows=counts[1]*2,
        skipped=reasons,
        no_fd_gradient_supervision=True,
        no_los=True,
        label_definition="Original qualified V3 best-found continuous FOV signed distance per sensor; two-sided analytic normal perturbations",
    )
    write_json(dst,report)
    print("[BOUNDARY PAIRS FROZEN]",json.dumps(
        {k:v for k,v in report.items() if not k.endswith("sha256")},
        indent=2),flush=True)


def status(root):
    out,m,tasks=load_plan(root)
    done=qualified=0
    for i in range(len(tasks["sensor"])):
        p=task_path(out,i)
        if p.is_dir():
            done+=1
            if read(p/"complete.json").get("status")=="QUALIFIED_PAIR":
                qualified+=1
    print("[BOUNDARY PAIRS STATUS]",json.dumps(dict(
        planned=len(tasks["sensor"]),completed=done,
        remaining=len(tasks["sensor"])-done,
        qualified=qualified,
        frozen=(out/"augmentation_manifest.json").is_file(),
    ),sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","work","merge","status"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if args.mode=="prepare":prepare(args.root)
    elif args.mode=="work":work(args.root,args.workers)
    elif args.mode=="merge":merge(args.root)
    else:status(args.root)


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,KeyError,OSError,FileNotFoundError,IndexError) as e:
        raise SystemExit(str(e))

#!/usr/bin/env python3
"""V4-A: refine legacy q0 bank points to analytic FOV boundary anchors and build local normal tubes."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,time
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
V1=REPO/"experiments/hierarchical9_offline_relabel_v1"
sys.path.insert(0,str(V1))
import cache_io
from repo_oracle import RepoOracle,sha256_file

FORMAT="careplanner_boundary_augment_v4a"
RADII=(0.005,0.01,0.02)

def digest_text(s): return hashlib.sha256(s.encode()).hexdigest()
def write_json(path,obj): cache_io.write_json(Path(path),obj)
def write_npz(path,obj): cache_io.write_npz(Path(path),obj)

def choose_points(pool,support,count,quota,rng):
    pool=rng.permutation(np.asarray(pool,np.int64))
    if count>len(pool): raise ValueError("requested x exceeds pool")
    selected=[];covered=np.zeros(8,int);used=np.zeros(len(pool),bool)
    while np.any(covered<quota):
        scores=(support[pool]*np.maximum(quota-covered,0)).sum(1);scores[used]=-1
        k=int(scores.argmax())
        if scores[k]<=0 or len(selected)>=count: raise ValueError("support quota cannot be satisfied")
        selected.append(int(pool[k]));used[k]=True;covered+=support[pool[k]]
    selected.extend(pool[~used][:count-len(selected)].tolist())
    return np.asarray(selected,np.int64)

def refine_anchor(q0,mask,lo,hi,geometry,tol=1e-6,maxiter=16,max_step=.10,backtracks=12):
    q=np.asarray(q0,np.float64).copy();mask=np.asarray(mask,np.float64);calls=0;hist=[]
    if np.any(q<lo) or np.any(q>hi) or not np.isfinite(q).all():
        return dict(success=False,reason="BAD_Q0",q_star=q,g=np.nan,iterations=0,calls=0,history=[])
    reason="MAXITER"
    for it in range(maxiter+1):
        h,j=geometry(q);calls+=1;h=np.asarray(h,float);j=np.asarray(j,float)
        face=int(np.argmin(h));g=float(h[face]);hist.append(g)
        if not np.isfinite(g) or not np.isfinite(j).all():
            reason="NONFINITE_GEOMETRY";break
        if abs(g)<=tol:
            return dict(success=True,reason="BOUNDARY",q_star=q.copy(),g=g,iterations=it,calls=calls,history=hist)
        if it==maxiter: break
        n=j[face]*mask;nn=float(n@n)
        if not math.isfinite(nn) or nn<1e-14:
            reason="DEGENERATE_NORMAL";break
        step=g*n/nn;sn=float(np.linalg.norm(step))
        if sn>max_step: step*=max_step/sn
        accepted=False
        for b in range(backtracks):
            cand=q-(.5**b)*step
            if np.any(cand<lo) or np.any(cand>hi): continue
            hc,_=geometry(cand);calls+=1;gc=float(np.min(hc))
            if math.isfinite(gc) and (abs(gc)<abs(g) or abs(gc)<=tol):
                q=cand;accepted=True;break
        if not accepted:
            reason="LINE_SEARCH_STALLED";break
    h,_=geometry(q);calls+=1
    return dict(success=False,reason=reason,q_star=q.copy(),g=float(np.min(h)),iterations=len(hist)-1,calls=calls,history=hist)

def prepare(args):
    source,m,bank,_=cache_io.open_job(args.source)
    tr,va=bank.original_split(args.val_count,args.split_seed)
    support=np.any(bank.valid,axis=1)
    excluded=set()
    if args.fresh_starts and args.fresh_starts.is_file():
        excluded={int(json.loads(x)["x_index"]) for x in args.fresh_starts.read_text().splitlines() if x.strip()}
    rng=np.random.default_rng(args.seed);chosen=[]
    for split,pool,n in ((0,tr,args.train_x),(1,va,args.val_x)):
        pool=np.asarray([int(x) for x in pool if int(x) not in excluded],np.int64)
        chosen.append(choose_points(pool,support,n,args.coverage_per_sensor,rng))
    tasks=[]
    for split,ids in enumerate(chosen):
        for xi in ids:
            for s in np.flatnonzero(support[xi]):
                slots=np.flatnonzero(np.asarray(bank.valid[xi,:,s],bool))
                take=min(args.anchors_per_sensor,len(slots))
                selected=np.sort(rng.choice(slots,take,replace=False))
                for slot in selected:
                    tasks.append((len(tasks),int(xi),int(split),int(s),int(slot)))
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(f"No overwrite: {out}")
    out.mkdir(parents=True)
    plan=dict(
      anchor_id=np.asarray([t[0] for t in tasks],np.int64),
      x_index=np.asarray([t[1] for t in tasks],np.int64),
      split=np.asarray([t[2] for t in tasks],np.uint8),
      sensor=np.asarray([t[3] for t in tasks],np.uint8),
      slot=np.asarray([t[4] for t in tasks],np.int32))
    write_npz(out/"plan.npz",plan)
    write_npz(out/"spatial_splits.npz",dict(train_x=chosen[0],val_x=chosen[1],fresh_excluded_x=np.asarray(sorted(excluded),np.int64)))
    oracle=RepoOracle(REPO,args.urdf,"cpu",joint_names=bank.joints,sensor_frames=bank.sensors)
    preflight=oracle.verify(bank,np.concatenate(chosen),np.zeros((1,7)))
    spec=dict(format=FORMAT,status="PREPARED",source=str(Path(args.source).resolve()),bank_cache=str(bank.root),
      original_data_sha256=m["original_data_sha256"],plan_sha256=sha256_file(out/"plan.npz"),
      spatial_splits_sha256=sha256_file(out/"spatial_splits.npz"),anchor_tasks=len(tasks),
      request=dict(train_x=args.train_x,val_x=args.val_x,anchors_per_sensor=args.anchors_per_sensor,
        coverage_per_sensor=args.coverage_per_sensor,seed=args.seed,split_seed=args.split_seed,val_count=args.val_count,
        radii_rad=list(RADII),fresh_starts=str(args.fresh_starts.resolve()) if args.fresh_starts else None),
      preflight=preflight,limitations=["FOV only","q0 refinement is a local analytic-boundary anchor, not a global nearest-boundary proof",
        "tube signed distance equals normal offset only in the retained regular same-face local neighborhood"])
    write_json(out/"manifest.json",spec)
    print(f"[prepared] x_train={len(chosen[0])} x_val={len(chosen[1])} anchors={len(tasks)} fresh_excluded={len(excluded)}",flush=True)

def work(args):
    out=args.output.resolve();spec=json.loads((out/"manifest.json").read_text())
    source,m,bank,_=cache_io.open_job(spec["source"])
    with np.load(out/"plan.npz",allow_pickle=False) as z: plan={k:z[k] for k in z.files}
    if sha256_file(out/"plan.npz")!=spec["plan_sha256"]: raise ValueError("plan changed")
    oracle=RepoOracle(REPO,args.urdf,"cpu",joint_names=bank.joints,sensor_frames=bank.sensors)
    anchor_rows=[];sample_rows=[];started=time.perf_counter()
    ids=np.arange(args.rank,len(plan["anchor_id"]),args.world)
    for pos,ii in enumerate(ids,1):
        aid=int(plan["anchor_id"][ii]);xi=int(plan["x_index"][ii]);split=int(plan["split"][ii]);s=int(plan["sensor"][ii]);slot=int(plan["slot"][ii])
        q0=np.asarray(bank.q[xi,slot,:,s],np.float64);mask=np.asarray(bank.masks[s],np.float64);geo=oracle.geometry(np.asarray(bank.x[xi],np.float64),s)
        t0=time.perf_counter();res=refine_anchor(q0,mask,bank.lo,bank.hi,geo);elapsed=time.perf_counter()-t0
        qstar=np.asarray(res["q_star"],float);normal=np.full(7,np.nan);face=-1;gap=np.nan;jnorm=np.nan;jmargin=np.nan;regular=False
        if res["success"]:
            h,j=geo(qstar);order=np.argsort(h);face=int(order[0]);gap=float(h[order[1]]-h[order[0]]) if len(h)>1 else math.inf
            n=np.asarray(j[face],float)*mask;jnorm=float(np.linalg.norm(n))
            active=np.flatnonzero(mask)
            jmargin=float(np.minimum(qstar-bank.lo,bank.hi-qstar)[active].min()) if len(active) else 0.
            if jnorm>1e-8: normal=n/jnorm
            regular=bool(gap>1e-4 and jnorm>1e-8 and jmargin>max(RADII)+.002)
        anchor_rows.append((aid,xi,split,s,slot,q0,qstar,normal,res["success"],regular,face,gap,jnorm,jmargin,res["g"],res["iterations"],res["calls"],elapsed,res["reason"]))
        offsets=(0.,)+tuple(v for r in RADII for v in (-r,r))
        for off in offsets:
            q=qstar+off*normal if regular else qstar.copy()
            in_limits=bool(np.isfinite(q).all() and np.all(q>=bank.lo) and np.all(q<=bank.hi))
            g=np.nan;sample_face=-1;same_face=False;sign_ok=False
            if regular and in_limits:
                h,_=geo(q);sample_face=int(np.argmin(h));g=float(np.min(h));same_face=sample_face==face
                sign_ok=abs(off)<1e-15 and abs(g)<=2e-6 or off>0 and g>0 or off<0 and g<0
            value_valid=bool(regular and in_limits and same_face and sign_ok)
            sample_rows.append((aid,xi,split,s,off,q,float(off) if value_valid else np.nan,normal if value_valid else np.full(7,np.nan),g,value_valid,same_face,in_limits))
        if pos%100==0:
            rate=pos/max(time.perf_counter()-started,1e-9)
            print(f"[rank{args.rank}] {pos}/{len(ids)} anchors rate={rate:.2f}/s",flush=True)
    shard=out/f"shard.rank{args.rank:02d}.npz"
    reasons=np.asarray([r[18] for r in anchor_rows],dtype="U32")
    write_npz(shard,dict(
      anchor_id=np.asarray([r[0] for r in anchor_rows],np.int64),x_index=np.asarray([r[1] for r in anchor_rows],np.int64),
      split=np.asarray([r[2] for r in anchor_rows],np.uint8),sensor=np.asarray([r[3] for r in anchor_rows],np.uint8),
      slot=np.asarray([r[4] for r in anchor_rows],np.int32),q0=np.asarray([r[5] for r in anchor_rows],np.float32),
      q_star=np.asarray([r[6] for r in anchor_rows],np.float32),normal=np.asarray([r[7] for r in anchor_rows],np.float32),
      refined=np.asarray([r[8] for r in anchor_rows],bool),regular=np.asarray([r[9] for r in anchor_rows],bool),
      active_face=np.asarray([r[10] for r in anchor_rows],np.int8),plane_gap_m=np.asarray([r[11] for r in anchor_rows],np.float32),
      normal_norm_m_per_rad=np.asarray([r[12] for r in anchor_rows],np.float32),joint_margin_rad=np.asarray([r[13] for r in anchor_rows],np.float32),
      boundary_g_m=np.asarray([r[14] for r in anchor_rows],np.float32),iterations=np.asarray([r[15] for r in anchor_rows],np.int16),
      geometry_calls=np.asarray([r[16] for r in anchor_rows],np.int16),elapsed_s=np.asarray([r[17] for r in anchor_rows],np.float32),reason=reasons,
      sample_anchor_id=np.asarray([r[0] for r in sample_rows],np.int64),sample_x_index=np.asarray([r[1] for r in sample_rows],np.int64),
      sample_split=np.asarray([r[2] for r in sample_rows],np.uint8),sample_sensor=np.asarray([r[3] for r in sample_rows],np.uint8),
      sample_offset_rad=np.asarray([r[4] for r in sample_rows],np.float32),sample_q=np.asarray([r[5] for r in sample_rows],np.float32),
      sample_value=np.asarray([r[6] for r in sample_rows],np.float32),sample_grad=np.asarray([r[7] for r in sample_rows],np.float32),
      sample_g_m=np.asarray([r[8] for r in sample_rows],np.float32),sample_value_valid=np.asarray([r[9] for r in sample_rows],bool),
      sample_same_face=np.asarray([r[10] for r in sample_rows],bool),sample_in_limits=np.asarray([r[11] for r in sample_rows],bool)))
    print(f"[done] rank={args.rank} anchors={len(anchor_rows)} samples={len(sample_rows)} shard={shard}",flush=True)

def merge(args):
    out=args.output.resolve();spec=json.loads((out/"manifest.json").read_text());parts=[]
    for r in range(args.world):
        p=out/f"shard.rank{r:02d}.npz"
        if not p.is_file(): raise FileNotFoundError(p)
        with np.load(p,allow_pickle=False) as z: parts.append({k:z[k] for k in z.files})
    anchor_keys=[k for k in parts[0] if not k.startswith("sample_")]
    sample_keys=[k for k in parts[0] if k.startswith("sample_")]
    a={k:np.concatenate([p[k] for p in parts]) for k in anchor_keys};s={k:np.concatenate([p[k] for p in parts]) for k in sample_keys}
    order=np.argsort(a["anchor_id"]);a={k:v[order] for k,v in a.items()}
    if not np.array_equal(a["anchor_id"],np.arange(spec["anchor_tasks"])): raise ValueError("anchor shard coverage mismatch")
    write_npz(out/"anchors.npz",a);write_npz(out/"samples.npz",s)
    summary=dict(status="COMPLETE",anchors=len(a["anchor_id"]),refined=int(a["refined"].sum()),regular=int(a["regular"].sum()),
      samples=len(s["sample_anchor_id"]),valid_value_samples=int(s["sample_value_valid"].sum()),
      elapsed_worker_seconds=float(a["elapsed_s"].sum()),median_anchor_ms=float(np.median(a["elapsed_s"])*1000),
      p95_anchor_ms=float(np.quantile(a["elapsed_s"],.95)*1000),
      median_geometry_calls=float(np.median(a["geometry_calls"])),reasons=dict(Counter(a["reason"].tolist())),
      per_sensor={f"S{i}":dict(anchors=int((a["sensor"]==i).sum()),refined=int((a["refined"]&(a["sensor"]==i)).sum()),
        regular=int((a["regular"]&(a["sensor"]==i)).sum())) for i in range(8)},
      anchors_sha256=sha256_file(out/"anchors.npz"),samples_sha256=sha256_file(out/"samples.npz"))
    write_json(out/"summary.json",summary)
    lines=["# V4-A boundary augmentation","",f"Anchors: {summary['anchors']}; refined: {summary['refined']}; regular: {summary['regular']}.",
      f"Tube samples: {summary['samples']}; valid value samples: {summary['valid_value_samples']}.",
      f"Median anchor: {summary['median_anchor_ms']:.2f} ms; p95: {summary['p95_anchor_ms']:.2f} ms; worker seconds: {summary['elapsed_worker_seconds']:.1f}.","",
      "| sensor | anchors | refined | regular |","|---|---:|---:|---:|"]
    for i in range(8):
        z=summary["per_sensor"][f"S{i}"];lines.append(f"| S{i} | {z['anchors']} | {z['refined']} | {z['regular']} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    spec.update(status="COMPLETE",anchors_sha256=summary["anchors_sha256"],samples_sha256=summary["samples_sha256"])
    write_json(out/"manifest.json",spec);print((out/"summary.md").read_text(),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("prepare","work","merge"),required=True)
    ap.add_argument("--source",type=Path);ap.add_argument("--urdf",type=Path,default=REPO/"src/arm_description/urdf/Arm.urdf")
    ap.add_argument("--fresh-starts",type=Path);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--train-x",type=int,default=256);ap.add_argument("--val-x",type=int,default=64);ap.add_argument("--anchors-per-sensor",type=int,default=4)
    ap.add_argument("--coverage-per-sensor",type=int,default=4);ap.add_argument("--seed",type=int,default=260924);ap.add_argument("--split-seed",type=int,default=0);ap.add_argument("--val-count",type=int,default=1000)
    ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=8);args=ap.parse_args()
    if args.mode=="prepare":
        if args.source is None: ap.error("--source required for prepare")
        prepare(args)
    elif args.mode=="work": work(args)
    else: merge(args)
if __name__=="__main__":main()

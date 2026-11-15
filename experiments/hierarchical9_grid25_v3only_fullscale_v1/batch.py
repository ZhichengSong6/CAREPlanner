#!/usr/bin/env python3
"""Single-job-at-a-time bulk V3-only label production for segments 8..123.

One Slurm allocation handles <=4 segments and returns exit status 42 for
continuation (chain.sh). Every later execution scans completed manifests and
resumes the first incomplete segment. Never auto-trains or overwrites labels.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

import runner

FORMAT="care_grid25_v3only_fullscale_batch_v1"
FIRST=8
STOP=124
PER_ALLOCATION=4


def require(ok,msg):
    if not ok: raise RuntimeError(msg)


def work_root(root):
    return Path(root).resolve()/"derived/grid25_v3only_fullscale_v1"


def atomic_json(path,value):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as out:
            json.dump(value,out,indent=2,sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def finished(root,id):
    """Fast, content-hashed restart guard; full shard checks on each new predecessor."""
    out=runner.segment_out(root,id)
    marker=out/"segment_complete.json"
    if not marker.is_file():return False
    m=runner.base.read(out/"manifest.json")
    c=runner.base.read(marker)
    require(c.get("status")=="SEGMENT_COMPLETE"
            and c.get("segment_id")==id
            and c.get("format")==runner.FORMAT
            and m.get("segment_id")==id
            and m.get("format")==runner.FORMAT
            and c["manifest_sha256"]==runner.sha(out/"manifest.json")
            and c["x_plan_sha256"]==runner.sha(out/"x_plan.npz")
            and m["v3only_code_identity"]==runner.codes()
            and m["original_algorithms"]==runner.prod.code_identity(),
            f"Frozen completed segment {id} changed")
    for file,digest in c["per_stage_summaries"].items():
        require(runner.sha(out/file)==digest,
                f"Frozen completed segment {id}: stage summary changed")
    require(set(c["per_stage_summaries"])==
            set(runner.base.STAGE_SUMMARIES.values()),
            "Unexpected complete stage list")
    return True


def frozen(root):
    pre,report,plan,segments=runner.base.frozen_source(root)
    require(len(segments)==124 and
            report["total_work"]["manual_segments"]==124 and
            len(plan["x_index"])==126201,
            "Unexpected full Grid25 frozen plan; expected 124 segments / 126201 x")
    require(all(segments[i]["segment_id"]==i for i in range(124)),
            "Unexpected frozen segment ordering")
    return pre,report,plan,segments


def advance_record(root,state,segment=None):
    data=dict(
        format=FORMAT,state=state,
        last_completed_segment=segment,
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        slurm_restart_count=os.environ.get("SLURM_RESTART_COUNT","0"),
        hostname=socket.gethostname(),
        scope=list(range(FIRST,STOP))[:1]+[STOP-1],
        total_new_segments=STOP-FIRST,
    )
    atomic_json(work_root(root)/"batch_progress.json",data)


def preview(root):
    frozen(root)
    work_root(root).mkdir(parents=True,exist_ok=True)
    ready=[]
    first=None
    for id in range(FIRST,STOP):
        if finished(root,id):
            ready.append(id)
        elif first is None:
            first=id
    require(ready==list(range(FIRST,FIRST+len(ready))),
            "Noncontiguous completed segment markers; refuse ambiguous bulk resume")
    if first is None:first=STOP
    info=dict(
        completed=len(ready),remaining=STOP-first,
        first_incomplete=first if first<STOP else None,
        range="8..123",total=STOP-FIRST,
        production_root=str(work_root(root)),
        finished_all=(first==STOP))
    return info


def preflight(root):
    info=preview(root)
    if info["completed"]==0:
        runner.verify_initial_predecessor(root)
    print("[FULLSCALE V3 PREFLIGHT]",json.dumps(info,indent=2),flush=True)
    return info


def complete_all(root):
    info=preview(root)
    require(info["finished_all"],"Not all 116 V3-only segments have completion markers")
    for id in range(FIRST,STOP):
        runner.verify(runner.segment_out(root,id),complete=True)
    record=dict(
        format=FORMAT,status="COMPLETE",completed_segments=STOP-FIRST,
        range=[FIRST,STOP-1],
        production_root=str(work_root(root)),
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_data_ready_for_merge=True,
        full_h9_training_finished=False,
        note="V3 label stages only; no certified FD gradients or V4 tubes")
    marker=work_root(root)/"batch_complete.json"
    if marker.exists():
        require(runner.base.read(marker)["format"]==FORMAT
                and runner.base.read(marker)["status"]=="COMPLETE",
                "Fullscale batch completion marker altered")
        print("[FULLSCALE BATCH ALREADY VERIFIED]",flush=True)
    else:
        with marker.open("x") as f:
            json.dump(record,f,indent=2,sort_keys=True);f.write("\n")
            f.flush();os.fsync(f.fileno())
    advance_record(root,"COMPLETE",STOP-1)
    print("[FULLSCALE ALL 116 V3 SEGMENTS VERIFIED COMPLETE]",flush=True)


def work(root):
    require(os.environ.get("SLURM_JOB_ID"),
            "Bulk work must run inside a Slurm batch allocation")
    info=preflight(root)
    basedir=work_root(root)
    with (basedir/".bulk.lock").open("a+") as fd:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Bulk production already running; no overlap") from e
        count=0
        for id in range(FIRST,STOP):
            if finished(root,id):
                continue
            if count>=PER_ALLOCATION:
                advance_record(root,"NEEDS_CONTINUATION",id-1)
                print("[FULLSCALE NEXT ALLOCATION NEEDED] resume_at=",id,
                      "slurm_job=",os.environ["SLURM_JOB_ID"],flush=True)
                # This deliberate exit code is handled by worker.sbatch,
                # which submits exactly one afterok child or requeues this job.
                raise SystemExit(42)
            out=runner.segment_out(root,id)
            if not out.exists():
                runner.prepare(root,id)
            else:
                require(out.is_dir() and (out/"manifest.json").is_file(),
                        f"Unexpected unfinished segment path: {out}")
                runner.open_segment(out)
                print("[FULLSCALE RESUME]",id,flush=True)
            print("[FULLSCALE WORK BEGIN]",id,flush=True)
            runner.work(out)
            runner.verify(out,complete=True)
            require(finished(root,id),"Segment verification marker not published")
            count+=1
            advance_record(root,"RUNNING",id)
            print("[FULLSCALE WORK COMPLETE]",id,"this_allocation=",count,flush=True)
        complete_all(root)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("preflight","status","work","verify-all"))
    p.add_argument("--root",type=Path,default=runner.DEFAULT_ROOT)
    args=p.parse_args()
    root=args.root.resolve()
    if args.mode=="status":
        print("[FULLSCALE PROGRESS]",json.dumps(preview(root),indent=2),flush=True)
    elif args.mode=="preflight":
        preflight(root)
    elif args.mode=="verify-all":
        complete_all(root)
    else:
        work(root)


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,OSError,KeyError,FileNotFoundError,
            IndexError,AssertionError) as exc:
        raise SystemExit("[STOP] "+str(exc))

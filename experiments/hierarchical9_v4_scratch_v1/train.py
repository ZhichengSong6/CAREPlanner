#!/usr/bin/env python3
"""Four-GPU random-init R1/private-tail scratch training on the full V4 supervision cache."""
from __future__ import annotations
import argparse,hashlib,json,os,random,sys,time,importlib.util
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import cache as cachelib
import objective as obj

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
models=_load("v4scratch_model",HERE/"model.py")
FORMAT="care_h9_v4_scratch_v1"

def log(*x):
    if not dist.is_initialized() or dist.get_rank()==0:print(*x,flush=True)
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def atomic_save(x,p):
    p=Path(p);tmp=p.with_name(p.name+f".tmp.{os.getpid()}");torch.save(x,tmp);os.replace(tmp,p)

def sample_batches(cache,args,split,step,rank,world,device):
    prefix="" if split=="train" else "val_"
    sizes={k:getattr(args,prefix+"batch_"+k) for k in ("global","v3","v4","boundary")}
    out={}
    for kind,n in sizes.items():
        if n%world:raise ValueError(f"{split}/{kind} batch must divide world")
        ids=cache.indices(split,kind,n//world,args.stream_seed,step,rank)
        out[kind]=cache.batch(split,kind,ids,device)
    return out

def run_objective(ddp,batches,w,training=True):
    world=dist.get_world_size()
    with torch.enable_grad(),torch.autocast("cuda",enabled=False):
        loss,stats=obj.combined_loss(ddp if training else ddp.module,batches,w,world,training=training)
    if training:loss.backward()
    metrics=obj.reduce_stats(stats,world)
    metrics["loss"]=(w.sign_sensor*metrics["sign_sensor"]+w.sign_union*metrics["sign_union"]+w.consistency*metrics["consistency"]
      +w.v3_sensor_sdf*metrics["v3_sensor_sdf"]+w.v3_union_sdf*metrics["v3_union_sdf"]+w.v4_sensor_sdf*metrics["v4_sensor_sdf"]
      +w.boundary_grad*metrics["boundary_grad"]+w.boundary_eikonal*metrics["boundary_eikonal"]+w.boundary_tension*metrics["boundary_tension"])
    return loss,metrics

def checkpoint(out,ddp,opt,sched,args,w,cache,step,stats,final=False):
    if dist.get_rank()!=0:return
    state=dict(format=FORMAT,completed=bool(final),step=int(step),initialization="random_from_scratch",
      architecture=ddp.module.architecture(),model_state=ddp.module.state_dict(),optimizer_state=opt.state_dict(),
      scheduler_state=sched.state_dict(),args={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},
      weights=asdict(w),cache_identity=cache.identity,cache_root=str(cache.root),code_sha=args.code_sha,stats=stats)
    atomic_save(state,out/("final.pt" if final else "latest.pt"))

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--code-sha",default="unknown")
    p.add_argument("--steps",type=int,default=50000);p.add_argument("--seed",type=int,default=0);p.add_argument("--stream-seed",type=int,default=261009)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--batch-global",type=int,default=8192);p.add_argument("--batch-v3",type=int,default=512);p.add_argument("--batch-v4",type=int,default=8192);p.add_argument("--batch-boundary",type=int,default=2048)
    p.add_argument("--val-batch-global",type=int,default=8192);p.add_argument("--val-batch-v3",type=int,default=2048);p.add_argument("--val-batch-v4",type=int,default=8192);p.add_argument("--val-batch-boundary",type=int,default=2048)
    p.add_argument("--log-every",type=int,default=100);p.add_argument("--val-every",type=int,default=1000);p.add_argument("--save-every",type=int,default=5000)
    for k,v in asdict(obj.LossWeights()).items():p.add_argument("--weight-"+k.replace("_","-"),dest="weight_"+k,type=float,default=v)
    p.add_argument("--resume",type=Path,default=None)
    a=p.parse_args()
    for k in ("steps","batch_global","batch_v3","batch_v4","batch_boundary","val_batch_global","val_batch_v3","val_batch_v4","val_batch_boundary","log_every","val_every","save_every"):
        if getattr(a,k)<=0:p.error(k+" must be positive")
    if not np.isfinite(a.lr) or a.lr<=0:p.error("lr must be positive")
    return a

def main():
    args=parse()
    if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:raise RuntimeError("use torchrun --nproc_per_node=4")
    local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=60));rank=dist.get_rank();world=dist.get_world_size()
    try:
        if world!=4:raise RuntimeError("exactly four GPUs required")
        args.cache=args.cache.resolve();args.out=args.out.resolve()
        ok=torch.tensor(1,device=device)
        if rank==0:
            try:cachelib.TrainingCache(args.cache,verify_hashes=True)
            except Exception as e:print("[cache-error]",repr(e),flush=True);ok.zero_()
        dist.broadcast(ok,0)
        if not ok.item():raise RuntimeError("cache verification failed")
        cache=cachelib.TrainingCache(args.cache,verify_hashes=False)
        for sp in ("train","val"):
            for kind in ("global","v3","v4","boundary"):
                if cache.size(sp,kind)<=0:raise RuntimeError(f"empty {sp}/{kind}")
        out=args.out;fresh=args.resume is None
        if rank==0:
            if fresh and out.exists() and any(out.iterdir()):raise RuntimeError(f"nonempty output {out}")
            out.mkdir(parents=True,exist_ok=True)
        dist.barrier();seed_all(args.seed)
        model=models.build_model("R1").to(device=device,dtype=torch.float32)
        if not all(p.requires_grad for p in model.parameters()):raise RuntimeError("unexpected frozen parameter")
        ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
        opt=torch.optim.Adam(ddp.parameters(),lr=args.lr)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=5000,threshold=.01,threshold_mode="rel",eps=1e-4)
        w=obj.LossWeights(**{k:getattr(args,"weight_"+k) for k in asdict(obj.LossWeights())});w.validate()
        start=0
        if args.resume:
            cp=torch.load(args.resume,map_location="cpu",weights_only=False)
            if cp.get("format")!=FORMAT or cp.get("cache_identity")!=cache.identity or cp.get("code_sha")!=args.code_sha:raise ValueError("resume identity mismatch")
            ddp.module.load_state_dict(cp["model_state"]);opt.load_state_dict(cp["optimizer_state"]);sched.load_state_dict(cp["scheduler_state"]);start=int(cp["step"])
        if start>=args.steps:raise ValueError("requested steps must exceed resume step")
        run_args={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
        if rank==0:
            (out/"run.json").write_text(json.dumps(dict(status="RUNNING",format=FORMAT,initialization="random_from_scratch",args=run_args,
              weights=asdict(w),cache_identity=cache.identity,cache_counts=cache.manifest["counts"],architecture=model.architecture(),code_sha=args.code_sha),indent=2)+"\n")
        log("[start] V4 scratch R1/private-tail",model.architecture())
        log("[cache]",cache.manifest["counts"])
        log("[batches]",{k:getattr(args,"batch_"+k) for k in ("global","v3","v4","boundary")},"weights",asdict(w))
        last={}
        for step in range(start+1,args.steps+1):
            t0=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
            batches=sample_batches(cache,args,"train",step,rank,world,device)
            opt.zero_grad(set_to_none=True)
            loss,metrics=run_objective(ddp,batches,w,True)
            finite=torch.isfinite(loss.detach()).int()
            grads=[p.grad for p in ddp.parameters() if p.grad is not None]
            if not grads:raise RuntimeError("no parameter gradients")
            finite=finite*torch.stack([torch.isfinite(g).all() for g in grads]).all().int()
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():raise RuntimeError("non-finite loss/gradients")
            opt.step();del batches,loss
            sched.step(metrics["loss"]);torch.cuda.synchronize(device)
            peak=torch.tensor(torch.cuda.max_memory_allocated(device)/1024**3,device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
            metrics.update(step=step,seconds=time.perf_counter()-t0,lr=opt.param_groups[0]["lr"],peak_gib=float(peak))
            last={"train":metrics}
            if step==1 or step%args.log_every==0 or step==args.steps:
                log("[train]",json.dumps(metrics,sort_keys=True))
                if rank==0:
                    with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps({"split":"train",**metrics},allow_nan=False)+"\n")
            if step==1 or step%args.val_every==0 or step==args.steps:
                ddp.eval();vb=sample_batches(cache,args,"val",0,rank,world,device)
                _,vm=run_objective(ddp,vb,w,False);del vb;ddp.train();last["val"]=vm
                log("[val]",step,json.dumps(vm,sort_keys=True))
                if rank==0:
                    with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps({"split":"val","step":step,**vm},allow_nan=False)+"\n")
            if step==1 or step%args.save_every==0 or step==args.steps:checkpoint(out,ddp,opt,sched,args,w,cache,step,last,False)
        checkpoint(out,ddp,opt,sched,args,w,cache,args.steps,last,True)
        if rank==0:
            run=json.loads((out/"run.json").read_text());run.update(status="COMPLETE",successful_updates=args.steps,final_sha256=sha(out/"final.pt"));(out/"run.json").write_text(json.dumps(run,indent=2)+"\n")
        log("[done]",out/"final.pt")
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=="__main__":main()

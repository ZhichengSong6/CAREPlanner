#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=ARMS,required=True);ap.add_argument("--r012-root",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--steps",type=int,default=400)
    ap.add_argument("--batch-size",type=int,default=64);ap.add_argument("--lr",type=float,default=2e-5);ap.add_argument("--seed",type=int,default=20260923)
    ap.add_argument("--eval-every",type=int,default=20);args=ap.parse_args()
    if args.steps<1 or args.batch_size<1 or args.eval_every<1 or not math.isfinite(args.lr) or args.lr<=0:ap.error("bad numeric option")
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda",0);torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.output.resolve()
    if out.exists():raise FileExistsError(f"No overwrite/resume: {out}")
    out.mkdir(parents=True);label,use_grad=arm_spec(args.arm);cache=PairedCache(args.cache)
    model,_,parent_sha=load_r1(args.r012_root,device,True);parent,_,_=load_r1(args.r012_root,device,False);parent.eval().requires_grad_(False)
    torch.manual_seed(args.seed);np.random.seed(args.seed);opt=torch.optim.Adam(model.parameters(),lr=args.lr);stream=hashlib.sha256()
    initial={"train":evaluate(model,cache,cache.train,device,parent),"val":evaluate(model,cache,cache.val,device,parent)};write_json(out/"initial.json",initial)
    argj=vars(args).copy();argj.update(r012_root=str(args.r012_root),cache=str(args.cache),output=str(out))
    write_json(out/"run.json",{"format":FORMAT,"status":"RUNNING","arm":args.arm,"label_set":label,"use_verified_grad":use_grad,"args":argj,
      "parent_sha256":parent_sha,"cache_index_sha256":cache.identity,"fairness":"same R1 init/cache/masks/Adam/updates/deterministic row stream"})
    for step in range(1,args.steps+1):
        ids=cache.train[stream_indices(len(cache.train),args.seed,step,args.batch_size)];stream.update(np.asarray(ids,np.int64).tobytes())
        b=cache.batch(ids,device,label);model.train();opt.zero_grad(set_to_none=True);loss,parts=paired_loss(model,b,label,use_grad,True)
        if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError("nonfinite parameter grad")
        opt.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            rec={"step":step,"loss":float(loss.detach()),**parts,"val":evaluate(model,cache,cache.val,device,parent),"train":evaluate(model,cache,cache.train,device,parent)}
            with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(rec,allow_nan=False)+"\n")
            print(f"[train] {args.arm} {step}/{args.steps} loss={rec['loss']:.6f} val_new={rec['val']['targets']['new']['sensor_mae_mean']:.6f} val_old={rec['val']['targets']['old']['sensor_mae_mean']:.6f} sign={rec['val']['analytic_sensor_sign_accuracy']:.5f}",flush=True)
    final={"train":evaluate(model,cache,cache.train,device,parent),"val":evaluate(model,cache,cache.val,device,parent),"stream_sha256":stream.hexdigest()}
    state={"format":FORMAT,"arm":args.arm,"completed":True,"parent_sha256":parent_sha,"cache_index_sha256":cache.identity,"steps":args.steps,
      "label_set":label,"use_verified_grad":use_grad,"args":argj,"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"final_metrics":final}
    torch.save(state,out/"final.pt");write_json(out/"final_metrics.json",final);run=json.loads((out/"run.json").read_text())
    run.update(status="COMPLETE",stream_sha256=stream.hexdigest(),final_sha256=sha256(out/"final.pt"));write_json(out/"run.json",run)
    print(f"[done] {args.arm} sha={run['final_sha256']}",flush=True)
if __name__=="__main__":main()

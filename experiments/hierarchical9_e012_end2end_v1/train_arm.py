#!/usr/bin/env python3
"""Train one E0/E1/E2 end-to-end arm on one GPU from the same P0 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import e012_protocol as proto
from e2e_model import build_from_p0, function_equivalence
import routed_objective as routed
from replay import ReplayBuffer, mine_projection_candidates

old=proto.old


def sample_global_indices_single(dataset,split,global_batch_x,device):
    pool=dataset.train_indices_cpu if split=="train" else dataset.val_indices_cpu if split=="val" else None
    if pool is None: raise ValueError(split)
    ridx=torch.randint(0,len(pool),(global_batch_x,),dtype=torch.long)
    return pool[ridx].to(device=device,non_blocking=True)


def sample_shared_q_single(dataset,batch_q,device):
    q_min,q_max=dataset.q_limits(device=device)
    u=torch.rand((batch_q,dataset.J),device=device)
    return q_min[None,:]+u*(q_max-q_min)[None,:]


def prepare_batch_single(api,dataset,oracle,args,device,split,baseline,obj):
    bx=args.global_batch_x if split=="train" else args.val_global_batch_x
    bq=args.batch_q if split=="train" else args.val_batch_q
    micro=args.microbatch_x if split=="train" else args.val_microbatch_x
    saved=baseline.rng_state() if split=="val" else None
    try:
        if split=="val": baseline.seed_everything(args.seed+100003)
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            indices=sample_global_indices_single(dataset,split,bx,device)
            q=sample_shared_q_single(dataset,bq,device)
            x,qlib,valid=api["materialize_local_x"](dataset,indices,0,1,device)
            masks=dataset.sensor_masks(device=device)
            batches=[]; counts9=torch.zeros(9,dtype=torch.float32,device=device)
            for start in range(0,len(x),micro):
                end=min(start+micro,len(x))
                ds,dgrad,has=api["decode_per_sensor_distance_and_grad"](
                    qlib=qlib[start:end],valid=valid[start:end],q_query=q,
                    sensor_masks=masks,x_chunk=args.decode_x_chunk)
                _,sign=oracle.signed_fov_margins(x[start:end],q)
                target,target_grad,mask=api["per_sensor_signed_targets"](ds,dgrad,sign,has)
                mask=mask.reshape(-1,8).contiguous()
                batches.append((api["make_input_pairs"](x[start:end],q).float(),
                    target.reshape(-1,8).float().contiguous(),
                    target_grad.reshape(-1,8,7).float().contiguous(),mask))
                counts9+=obj.counts_from_mask(mask)
            if counts9[0]<=0: raise RuntimeError("No supervised rows")
            identity=hashlib.sha256()
            identity.update(np.ascontiguousarray(indices.detach().cpu().numpy()).tobytes())
            identity.update(np.ascontiguousarray(q.detach().cpu().float().numpy()).tobytes())
            return batches,counts9,identity.digest()
    finally:
        if saved is not None: baseline.restore_rng(saved)


def amp_context(device,amp):
    return torch.autocast(device_type=device.type,enabled=amp!="off",
                          dtype=torch.float16 if amp=="fp16" else torch.bfloat16)


def validate(model,batches,counts9,weights,obj):
    device=counts9.device
    total=torch.zeros((9,len(obj.STAT_NAMES)),dtype=torch.float64,device=device)
    model.eval()
    for batch in batches:
        with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
            loss,st=obj.loss_for_microbatch(model,*batch,counts9,weights,world_size=1,training=False)
        if not torch.isfinite(loss.detach()) or not torch.isfinite(st).all():
            raise RuntimeError("Nonfinite validation")
        total+=st
    return obj.summarize(total,weights)


def _accumulate(dst,grads):
    for i,g in enumerate(grads):
        if g is not None: dst[i].add_(g.detach())


def _finite_train_grads(model):
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    return bool(grads) and all(bool(torch.isfinite(g).all().item()) for g in grads)


def hard_alpha(args,step):
    return args.hard_weight*min(float(step)/max(1,args.hard_warmup),1.0)


def run_update(model,batches,counts9,weights,args,obj,optimizer,scaler,step,replay_buffer=None):
    device=counts9.device; selected=proto.selected_shared_sensor(step)
    early=list(model.early.parameters())
    replay_batches={}
    if args.arm=="E2" and replay_buffer is not None:
        replay_batches=replay_buffer.sample(args.replay_sample_per_sensor,seed=args.seed,step=step,device=device)
    alpha=hard_alpha(args,step) if args.arm=="E2" else 0.0

    for attempt in range(args.max_amp_retries+1):
        optimizer.zero_grad(set_to_none=True)
        stats=torch.zeros((9,len(obj.STAT_NAMES)),dtype=torch.float64,device=device)
        early_aux=[torch.zeros_like(p) for p in early]
        model.train()
        for batch in batches:
            if args.arm=="E0":
                with torch.enable_grad(),amp_context(device,args.amp):
                    loss,st=obj.loss_for_microbatch(model,*batch,counts9,weights,world_size=1,training=True)
            else:
                with model.sensor_early_parameter_frozen():
                    with torch.enable_grad(),amp_context(device,args.amp):
                        loss,st=obj.loss_for_microbatch(model,*batch,counts9,weights,world_size=1,training=True)
                with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
                    shared_loss=routed.selected_sensor_shared_objective(model,*batch,counts9,weights,selected)
                    sg=torch.autograd.grad(shared_loss,early,allow_unused=False)
                _accumulate(early_aux,sg)
            if not torch.isfinite(loss.detach()) or not torch.isfinite(st).all():
                raise RuntimeError("Nonfinite uniform loss")
            scaler.scale(loss).backward(); stats+=st

        hard_stats={"count":0,"fp":0,"fn":0,"outside":0,"inside":0,"alpha":alpha,
                    "field_margin":args.hard_field_margin}
        if args.arm=="E2" and replay_batches and alpha>0:
            with torch.enable_grad(),amp_context(device,args.amp):
                hard_private,hs=routed.replay_private_loss(
                    model,replay_batches,args.hard_fn_ratio,args.hard_field_margin)
            scaler.scale(alpha*hard_private).backward(); hard_stats.update(hs)
            with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
                hard_shared=routed.replay_shared_sensor_loss(
                    model,replay_batches,selected,args.hard_fn_ratio,args.hard_field_margin)
                hg=torch.autograd.grad(alpha*hard_shared,early,allow_unused=False)
            _accumulate(early_aux,hg)

        scaler.unscale_(optimizer)
        if args.arm in ("E1","E2"):
            for p,g in zip(early,early_aux):
                if p.grad is None: p.grad=g.clone()
                else: p.grad.add_(g)
        if _finite_train_grads(model):
            scaler.step(optimizer); scaler.update(); break
        if args.amp!="fp16" or attempt==args.max_amp_retries:
            raise RuntimeError("Nonfinite parameter gradients")
        scaler.update(new_scale=scaler.get_scale()/2.0)

    return {"global":obj.summarize(stats,weights),"selected_shared_sensor":selected,
            "hard":hard_stats,"amp_retries":attempt,"grad_scale":scaler.get_scale()}


def weight_drift(model,initial_state):
    groups={"early":[],"union":[],"sensor_tails":[],"sensor_heads":[]}
    for name,p in model.state_dict().items():
        if not torch.is_floating_point(p): continue
        key=("early" if name.startswith("early.") else "union" if name.startswith(("union_tail.","union_head."))
             else "sensor_tails" if name.startswith("sensor_tails.") else "sensor_heads")
        base=initial_state[name].to(p.device,dtype=p.dtype); diff=(p-base).double(); groups[key].append((diff,base.double()))
    out={}
    for key,rows in groups.items():
        d2=sum(float((d*d).sum()) for d,_ in rows); b2=sum(float((b*b).sum()) for _,b in rows); max_abs=max(float(d.abs().max()) for d,_ in rows)
        out[key]={"relative_l2":math.sqrt(d2/max(b2,1e-30)),"max_abs":max_abs}
    return out


def save(path,model,optimizer,scaler,args,step,validation,p0_sha,cache_identity,stream_sha,
         equivalence,initial_state,replay_buffer,last_mining,baseline,*,final=False):
    state=dict(format=proto.FORMAT,arm=args.arm,mode=args.mode,completed=final,parent="P0",
        parent_sha256=p0_sha,parent_updates=52000,extra_updates=step,total_updates=52000+step,
        cache_manifest_sha256=cache_identity,
        optimizer_policy="fresh_Adam_param_groups_constant_lr_no_scheduler_no_clipping",
        original_global_objective=True,all_model_parameters_trainable=True,
        arm_definition=proto.arm_definition(args.arm),args=vars(args),architecture=model.architecture(),
        initial_equivalence=equivalence,training_stream_sha256=stream_sha,
        model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict(),
        validation=validation,weight_drift_from_initial=weight_drift(model,initial_state),
        replay=None if replay_buffer is None else replay_buffer.summary(),last_mining=last_mining,
        training_source_sha256=proto.training_fingerprints(),source_sha256=old.source_fingerprints())
    if final: proto.assert_checkpoint(state,_GLOBAL_P0,p0_sha,cache_identity,require_pilot=args.mode=="pilot")
    baseline.atomic_save(state,Path(path))


_GLOBAL_P0=None


def main():
    global _GLOBAL_P0
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--reference-root",type=Path,required=True)
    ap.add_argument("--arm",choices=proto.ARMS,required=True); ap.add_argument("--mode",choices=("smoke","pilot"),required=True); a0=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("One CUDA GPU required")
    device=torch.device("cuda",0); torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False

    root=a0.reference_root.resolve(); cache,p0,p0_sha=proto.load_reference_root(root); _GLOBAL_P0=p0
    out=proto.output_dir(root,a0.arm,a0.mode); args=proto.make_args(p0,out,a0.arm,a0.mode)
    if out.exists(): raise FileExistsError(f"No overwrite/resume: {out}")
    out.mkdir(parents=True)
    baseline=old.module("train",old.SCRATCH); obj=old.module("objective",old.SCRATCH); weights=obj.LossWeights()
    api=baseline.load_repo_api(old.REPO); artifact=Path(args.artifact_root)
    dataset=api["VisibilityQ0Dataset"](str(artifact/old.DATA_REL),val_count=1000,seed=0); cache.verify_dataset(dataset,artifact/old.DATA_REL)
    oracle=api["PinocchioFOVOracle"](urdf_path=str(old.URDF),joint_names=api["DEFAULT_JOINT_NAMES"],
        sensor_frames=api["DEFAULT_SENSOR_FRAMES"],horizontal_fov_deg=50.,vertical_fov_deg=66.,z_min=.2,z_max=.7,delta=.01)
    projection=old.module("compare_scalar_vs_per_sensor_apples_to_apples",old.SCRIPTS); lo,hi=dataset.q_limits(device)

    baseline.seed_everything(args.seed)
    p0_model=proto.p0_model(p0,device).eval().requires_grad_(False); model=build_from_p0(p0_model).to(device=device,dtype=torch.float32)
    initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.optimizer_groups(lr_early=args.lr_early,lr_union=args.lr_union,lr_sensor=args.lr_sensor))
    scaler=torch.amp.GradScaler("cuda",enabled=args.amp=="fp16",init_scale=1024.)
    replay_buffer=ReplayBuffer(args.replay_capacity_per_sensor) if args.arm=="E2" else None

    val_batches,val_counts9,_=prepare_batch_single(api,dataset,oracle,args,device,"val",baseline,obj)
    equivalence=function_equivalence(p0_model,model,val_batches[0][0][:32])
    if equivalence["max_abs_value_error"]>2e-6 or equivalence["max_abs_q_gradient_error"]>2e-6: raise RuntimeError(f"Not P0-equivalent: {equivalence}")
    validation=validate(model,val_batches,val_counts9,weights,obj); proto.write_json(out/"initial_validation.json",validation)
    proto.write_json(out/"run.json",dict(status="RUNNING",arm=args.arm,mode=args.mode,args=vars(args),parent="P0",parent_sha256=p0_sha,
        cache_manifest_sha256=cache.identity,architecture=model.architecture(),initial_equivalence=equivalence,training_source_sha256=proto.training_fingerprints()))
    print(f"[e012] arm={args.arm} mode={args.mode} P0={p0_sha} params={model.parameter_count()} equivalence={equivalence} routing={proto.arm_definition(args.arm)}",flush=True)

    stream=hashlib.sha256(); last_mining=None
    for step in range(1,args.steps+1):
        started=time.perf_counter(); baseline.seed_everything(proto.seed_for_update(args.seed,step))
        batches,counts9,identity=prepare_batch_single(api,dataset,oracle,args,device,"train",baseline,obj); stream.update(identity); torch.cuda.reset_peak_memory_stats(device)
        result=run_update(model,batches,counts9,weights,args,obj,optimizer,scaler,step,replay_buffer)
        if args.arm=="E2" and step%args.mine_every==0:
            last_mining=mine_projection_candidates(model,dataset,oracle,projection,args,step,device,lo,hi,replay_buffer); result["mining"]=last_mining
        torch.cuda.synchronize(device); result.update(update=step,seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        if step==1 or step%args.log_every==0 or step==args.steps:
            h=result["global"]["heads"]
            print(f"[train] {args.arm} {step}/{args.steps} loss={result['global']['loss']:.6f} Ucos={h['union']['grad_cosine']:.4f} "
                  f"S0cos={h['s0']['grad_cosine']:.4f} S6cos={h['s6']['grad_cosine']:.4f} sel={result['selected_shared_sensor']} hardN={result['hard']['count']} sec={result['seconds']:.2f}",flush=True)
            with (out/"metrics.jsonl").open("a") as f: f.write(json.dumps(result,allow_nan=False)+"\n")
        if step==1 or step%args.val_every==0 or step==args.steps:
            validation=validate(model,val_batches,val_counts9,weights,obj); drift=weight_drift(model,initial_state)
            with (out/"validation.jsonl").open("a") as f: f.write(json.dumps({"update":step,"validation":validation,"weight_drift":drift,"replay":None if replay_buffer is None else replay_buffer.summary()},allow_nan=False)+"\n")
            vh=validation["heads"]
            print(f"[val] {args.arm} update={step} loss={validation['loss']:.6f} Ucos={vh['union']['grad_cosine']:.4f} S0cos={vh['s0']['grad_cosine']:.4f} "
                  f"S6cos={vh['s6']['grad_cosine']:.4f} early_drift={drift['early']['relative_l2']:.4e}",flush=True)
            save(out/"latest.pt",model,optimizer,scaler,args,step,validation,p0_sha,cache.identity,stream.hexdigest(),equivalence,initial_state,replay_buffer,last_mining,baseline)
        del batches

    save(out/"final.pt",model,optimizer,scaler,args,args.steps,validation,p0_sha,cache.identity,stream.hexdigest(),equivalence,initial_state,replay_buffer,last_mining,baseline,final=True)
    run=json.loads((out/"run.json").read_text()); run.update(status="COMPLETE",successful_updates=args.steps,training_stream_sha256=stream.hexdigest(),
        final_sha256=proto.sha256(out/"final.pt"),weight_drift_from_initial=weight_drift(model,initial_state),replay=None if replay_buffer is None else replay_buffer.summary())
    proto.write_json(out/"run.json",run); print(f"[done] e012_arm_complete arm={args.arm} mode={args.mode} updates={args.steps} output={out}",flush=True)


if __name__=="__main__": main()

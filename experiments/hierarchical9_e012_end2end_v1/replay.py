"""Deterministic runtime-like projection mining and per-sensor replay buffers for E2."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

import e012_protocol as proto


class ReplayBuffer:
    def __init__(self, capacity_per_sensor: int):
        if capacity_per_sensor <= 0:
            raise ValueError(capacity_per_sensor)
        self.capacity = int(capacity_per_sensor)
        self.rows = [deque(maxlen=self.capacity) for _ in range(8)]
        self.total_added = [0]*8

    def add(self, sensor: int, x: torch.Tensor, q: torch.Tensor, labels: torch.Tensor, g: torch.Tensor):
        x=x.detach().cpu().float(); q=q.detach().cpu().float()
        labels=labels.detach().cpu().float(); g=g.detach().cpu().float()
        if x.shape != (len(q),3) or q.ndim != 2 or q.shape[1] != 7 or labels.shape != (len(q),) or g.shape != (len(q),):
            raise ValueError("Invalid replay rows")
        for i in range(len(q)):
            self.rows[sensor].append((x[i].clone(),q[i].clone(),labels[i].clone(),g[i].clone()))
            self.total_added[sensor]+=1

    def sample(self, count_per_sensor: int, *, seed: int, step: int, device: torch.device):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step,proto.MINER_RNG_TAG,991]))
        result={}
        for s,rows in enumerate(self.rows):
            n=min(int(count_per_sensor),len(rows))
            if n <= 0: continue
            ids=rng.choice(len(rows),n,replace=False)
            chosen=[rows[int(i)] for i in ids]
            x=torch.stack([r[0] for r in chosen]).to(device)
            q=torch.stack([r[1] for r in chosen]).to(device)
            result[s]={
                "inputs":torch.cat((x,q),1),
                "labels":torch.stack([r[2] for r in chosen]).to(device),
                "g_m":torch.stack([r[3] for r in chosen]).to(device),
            }
        return result

    def summary(self):
        return {"size_by_sensor":[len(r) for r in self.rows],"total_added_by_sensor":list(self.total_added),
                "capacity_per_sensor":self.capacity}


class SensorScalarView(nn.Module):
    def __init__(self, model: nn.Module, sensor: int):
        super().__init__()
        object.__setattr__(self,"_model",model)
        self.sensor=int(sensor)
    def forward(self, inputs):
        model=object.__getattribute__(self,"_model")
        return model.forward_sensor(inputs,self.sensor)[:,None]


@contextmanager
def parameter_frozen(model: nn.Module):
    state=[p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters(): p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(model.parameters(),state): p.requires_grad_(flag)


def matched_sensor_g(oracle, x: torch.Tensor, q_pair: torch.Tensor, sensor: int):
    """Actual conservative FOV margin for matched x rows and per-row q batches."""
    bx,bq,_=q_pair.shape
    out=torch.empty((bx,bq),device=q_pair.device,dtype=torch.float32)
    with torch.no_grad():
        for i in range(bx):
            raw,_=oracle.signed_fov_margins(x[i:i+1],q_pair[i])
            out[i]=raw[0,:,sensor].float()-float(oracle.delta)
    return out


def mine_projection_candidates(model,dataset,oracle,baseline,args,step,device,lo,hi,buffer:ReplayBuffer):
    """Mine projection candidates from actual-outside starts using isolated RNG.

    This deliberately reuses the established projection routine and the exact
    analytic FOV oracle.  It is a runtime-aligned training signal, not a claim to
    reproduce the full Sparse-SCP/VBC/GCDF execution stack.
    """
    rng=np.random.default_rng(np.random.SeedSequence([args.seed,step,proto.MINER_RNG_TAG]))
    train_ids=np.asarray(getattr(dataset,"train_indices_np",dataset.train_indices_cpu.numpy()))
    lo_np=lo.detach().cpu().numpy(); hi_np=hi.detach().cpu().numpy()
    report={"attempted":0,"initial_outside":0,"accepted":0,"fp_at_mine":0,"fn_at_mine":0,"by_sensor":{}}
    was_training=model.training
    model.eval()
    with parameter_frozen(model):
        for s in range(8):
            ids=rng.choice(train_ids,args.mine_x_per_sensor,replace=True)
            x=dataset.x_cpu[ids].to(device=device,dtype=torch.float32)
            q=torch.tensor(rng.uniform(lo_np,hi_np,(len(x),args.mine_q_per_x,7)),device=device,dtype=torch.float32)
            g0=matched_sensor_g(oracle,x,q,s)
            outside=g0 < -args.mine_ambiguous_g_m
            report["attempted"]+=outside.numel(); report["initial_outside"]+=int(outside.sum())
            view=SensorScalarView(model,s)
            qp=baseline._projection(view,"scalar",x,q,lo,hi,args.mine_projection_iters,
                                    args.mine_projection_damping,args.mine_projection_max_step)
            with torch.no_grad():
                bx,bq,_=qp.shape
                xf=x[:,None,:].expand(bx,bq,3).reshape(-1,3)
                pred=model.forward_sensor(torch.cat((xf,qp.reshape(-1,7)),1),s).reshape(bx,bq)
            g=matched_sensor_g(oracle,x,qp,s)
            valid=outside & (g.abs()>args.mine_ambiguous_g_m) & torch.isfinite(g) & torch.isfinite(pred)
            labels=torch.where(g>=0,torch.ones_like(g),-torch.ones_like(g))
            fp=valid & (labels<0) & (pred>=0)
            fn=valid & (labels>0) & (pred<0)
            keep=torch.where(valid.reshape(-1))[0]
            if len(keep):
                xf=x[:,None,:].expand(len(x),q.shape[1],3).reshape(-1,3)[keep]
                qf=qp.reshape(-1,7)[keep]
                lf=labels.reshape(-1)[keep]; gf=g.reshape(-1)[keep]
                buffer.add(s,xf,qf,lf,gf)
            report["accepted"]+=len(keep); report["fp_at_mine"]+=int(fp.sum()); report["fn_at_mine"]+=int(fn.sum())
            report["by_sensor"][f"s{s}"]={"outside":int(outside.sum()),"accepted":len(keep),
                                              "fp":int(fp.sum()),"fn":int(fn.sum())}
    model.train(was_training)
    report["buffer"]=buffer.summary()
    return report

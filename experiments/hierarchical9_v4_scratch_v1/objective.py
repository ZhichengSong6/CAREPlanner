"""Three-mask objective for full-scale V4 scratch training."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class LossWeights:
    sign_sensor:float=1.0
    sign_union:float=1.0
    sign_temperature:float=0.1
    v3_sensor_sdf:float=5.0
    v3_union_sdf:float=5.0
    v4_sensor_sdf:float=5.0
    boundary_grad:float=0.1
    boundary_eikonal:float=0.01
    boundary_tension:float=0.01
    consistency:float=0.1
    def validate(self):
        for k,v in asdict(self).items():
            if not math.isfinite(v) or v<0:raise ValueError(f"bad weight {k}={v}")
        if self.sign_temperature<=0 or self.v3_sensor_sdf<=0 or self.v4_sensor_sdf<=0:raise ValueError("positive core weights required")

def _logistic(y,sign,t):
    return t*F.softplus(-sign.float()*y.float()/t)

def global_sign_loss(model,b,w):
    pred=model(b["inputs"]).float();ss=b["sensor_sign"].float();us=b["union_sign"].float()
    ls=_logistic(pred[:,1:],ss,w.sign_temperature).mean()
    lu=_logistic(pred[:,0],us,w.sign_temperature).mean()
    sensor_max=pred[:,1:].max(dim=1).values
    cons=(pred[:,0]-sensor_max).square().mean()
    loss=w.sign_sensor*ls+w.sign_union*lu+w.consistency*cons
    stats=dict(sign_sensor=ls.detach(),sign_union=lu.detach(),consistency=cons.detach(),
      sensor_sign_acc=((pred[:,1:]>=0)==(ss>=0)).float().mean().detach(),
      union_sign_acc=((pred[:,0]>=0)==(us>=0)).float().mean().detach())
    return loss,stats

def _head_balanced(local_sum,local_count,world):
    count=local_count.clone();torch.distributed.all_reduce(count)
    active=count>0
    if not active.any():return local_sum.sum()*0.0,count
    val=(local_sum[active]/count[active]).sum()/active.sum()
    return val*float(world),count

def v3_value_loss(model,b,w,world):
    pred=model(b["inputs"]).float();mask=b["sensor_value_mask"].bool();target=b["sensor_value"].float()
    sums=torch.zeros(8,device=pred.device);counts=mask.sum(0).float()
    for s in range(8):
        if mask[:,s].any():sums[s]=(pred[mask[:,s],s+1]-target[mask[:,s],s]).square().sum()
    sensor,global_counts=_head_balanced(sums,counts,world)
    um=b["union_value_mask"].bool();uc=um.sum().float();torch.distributed.all_reduce(uc)
    if uc.item()>0:
        us=((pred[um,0]-b["union_value"][um].float()).square().sum()*float(world)/uc)
    else:us=pred.sum()*0.0
    return w.v3_sensor_sdf*sensor+w.v3_union_sdf*us,dict(v3_sensor_sdf=sensor.detach(),v3_union_sdf=us.detach(),v3_value_count=global_counts.sum().detach(),v3_union_count=uc.detach())

def v4_value_loss(model,b,w,world):
    pred=model(b["inputs"]).float();sensor=b["sensor"].long();row=torch.arange(len(sensor),device=pred.device)
    err=(pred[row,sensor+1]-b["value"].float()).square()
    sums=torch.zeros(8,device=pred.device);counts=torch.bincount(sensor,minlength=8).float()
    sums.scatter_add_(0,sensor,err)
    sensor_loss,global_counts=_head_balanced(sums,counts,world)
    return w.v4_sensor_sdf*sensor_loss,dict(v4_sensor_sdf=sensor_loss.detach(),v4_count=global_counts.sum().detach())

def boundary_loss(model,b,w,world,training=True):
    q=b["inputs"][:,3:].detach().float().clone().requires_grad_(True);x=b["inputs"][:,:3].detach().float()
    pred=model(torch.cat((x,q),1)).float();sensor=b["sensor"].long();row=torch.arange(len(sensor),device=pred.device)
    y=pred[row,sensor+1]
    g=torch.autograd.grad(y.sum(),q,create_graph=True,retain_graph=True)[0].float()
    target=b["grad"].float()
    cos=F.cosine_similarity(g,target,dim=-1,eps=1e-6);grad_term=1-cos
    norm=torch.linalg.vector_norm(g,dim=-1);eik=(norm-1).abs()
    if w.boundary_tension>0:
        hv=torch.autograd.grad(g.sum(),q,create_graph=training,retain_graph=True)[0].float()
        tension=hv.square().sum(dim=1)
    else:tension=y*0
    counts=torch.bincount(sensor,minlength=8).float()
    def balanced(v):
        sums=torch.zeros(8,device=v.device);sums.scatter_add_(0,sensor,v)
        return _head_balanced(sums,counts,world)[0]
    gl=balanced(grad_term);el=balanced(eik);tl=balanced(tension)
    loss=w.boundary_grad*gl+w.boundary_eikonal*el+w.boundary_tension*tl
    return loss,dict(boundary_grad=gl.detach(),boundary_cos=(1-gl).detach(),boundary_eikonal=el.detach(),boundary_tension=tl.detach(),boundary_grad_norm=balanced(norm).detach())

def reduce_stats(stats,world):
    out={}
    for d in stats:
        for k,v in d.items():
            t=v.detach().float().clone();torch.distributed.all_reduce(t)
            out[k]=float((t/world).item())
    return out

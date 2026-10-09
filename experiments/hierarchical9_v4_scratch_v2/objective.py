"""Class-balanced masked full-V4 objective: one FP32 forward/backward per update."""
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

def _logistic(y,sign,t):return t*F.softplus(-sign.float()*y.float()/t)

def _class_cells(y,sign,t):
    """Pure local sufficient statistics, columns x {negative,positive}."""
    if y.shape!=sign.shape or y.ndim!=2:raise ValueError("expected matching [N,C]")
    loss=_logistic(y,sign,t);positive=sign>=0
    c=y.shape[1];sums=torch.zeros((c,2),device=y.device,dtype=torch.float32)
    counts=torch.zeros((c,2),device=y.device,dtype=torch.float32)
    correct=torch.zeros((c,2),device=y.device,dtype=torch.float32)
    pred_positive=y>=0
    for j in range(c):
        for cls in (0,1):
            m=positive[:,j] if cls else ~positive[:,j]
            counts[j,cls]=m.sum()
            if m.any():
                sums[j,cls]=loss[m,j].sum()
                correct[j,cls]=(pred_positive[m,j]==positive[m,j]).float().sum()
    return sums,counts,correct

def _balanced_binary(y,sign,t,world):
    sums,counts,correct=_class_cells(y,sign,t)
    gc=counts.clone();gcorrect=correct.clone()
    torch.distributed.all_reduce(gc);torch.distributed.all_reduce(gcorrect)
    active=gc>0
    if not active.any():raise RuntimeError("no active sign class")
    objective=(sums[active]/gc[active]).mean()*float(world)
    balanced_acc=(gcorrect[active]/gc[active]).mean()
    raw_correct=gcorrect.sum();raw_total=gc.sum()
    raw_acc=raw_correct/raw_total
    positive_fraction=gc[:,1].sum()/raw_total
    return objective,dict(raw_acc=raw_acc.detach(),balanced_acc=balanced_acc.detach(),
                          positive_fraction=positive_fraction.detach(),class_cells=active.sum().detach())

def _balanced(local_sum,local_count,world):
    count=local_count.clone();torch.distributed.all_reduce(count)
    active=count>0
    if not active.any():return local_sum.sum()*0.0,count
    local=(local_sum[active]/count[active]).sum()/active.sum()
    return local*float(world),count

def combined_loss(model,b,w,world,training=True):
    order=("global","v3","v4","boundary")
    sizes={k:len(b[k]["inputs"]) for k in order};starts={};n=0
    for k in order:starts[k]=(n,n+sizes[k]);n+=sizes[k]
    inp=torch.cat([b[k]["inputs"].float() for k in order],dim=0)
    q=inp[:,3:10].detach().clone().requires_grad_(True);x=inp[:,:3].detach()
    pred=model(torch.cat((x,q),dim=1)).float();comp={};stats={}
    # Global analytic signs: every (sensor,class) cell has equal objective weight.
    a,z=starts["global"];pg=pred[a:z];ss=b["global"]["sensor_sign"].float();us=b["global"]["union_sign"].float()
    comp["sign_sensor"],sm=_balanced_binary(pg[:,1:],ss,w.sign_temperature,world)
    comp["sign_union"],umet=_balanced_binary(pg[:,0:1],us[:,None],w.sign_temperature,world)
    comp["consistency"]=(pg[:,0]-pg[:,1:].max(dim=1).values).square().mean()
    stats.update(sensor_sign_acc=sm["raw_acc"],sensor_sign_bal_acc=sm["balanced_acc"],
                 sensor_positive_fraction=sm["positive_fraction"],sensor_sign_class_cells=sm["class_cells"],
                 union_sign_acc=umet["raw_acc"],union_sign_bal_acc=umet["balanced_acc"],
                 union_positive_fraction=umet["positive_fraction"],union_sign_class_cells=umet["class_cells"])
    # V3 continuous values; independent sensor masks. Strict union only if all 8 valid.
    a,z=starts["v3"];pv=pred[a:z];mask=b["v3"]["sensor_value_mask"].bool();target=b["v3"]["sensor_value"].float()
    sums=torch.zeros(8,device=pred.device);counts=mask.sum(0).float()
    for s in range(8):
        if mask[:,s].any():sums[s]=(pv[mask[:,s],s+1]-target[mask[:,s],s]).square().sum()
    comp["v3_sensor_sdf"],v3counts=_balanced(sums,counts,world)
    uvm=b["v3"]["union_value_mask"].bool();uc=uvm.sum().float();torch.distributed.all_reduce(uc)
    comp["v3_union_sdf"]=(pv[uvm,0]-b["v3"]["union_value"][uvm].float()).square().sum()*float(world)/uc if uc.item()>0 else pv.sum()*0
    stats["v3_value_count"]=v3counts.sum().detach();stats["v3_union_count"]=uc.detach()
    # V4 local signed values.
    a,z=starts["v4"];p4=pred[a:z];sensor=b["v4"]["sensor"].long();row=torch.arange(len(sensor),device=pred.device)
    ve=(p4[row,sensor+1]-b["v4"]["value"].float()).square()
    sums=torch.zeros(8,device=pred.device);counts=torch.bincount(sensor,minlength=8).float();sums.scatter_add_(0,sensor,ve)
    comp["v4_sensor_sdf"],v4counts=_balanced(sums,counts,world);stats["v4_count"]=v4counts.sum().detach()
    # Zero-offset regular V4 boundary normals only.
    a,z=starts["boundary"];pb=pred[a:z];bs=b["boundary"]["sensor"].long();br=torch.arange(len(bs),device=pred.device)
    by=pb[br,bs+1]
    qall_grad=torch.autograd.grad(by.sum(),q,create_graph=True,retain_graph=True)[0].float()
    g=qall_grad[a:z];targetg=b["boundary"]["grad"].float()
    grad_term=1-F.cosine_similarity(g,targetg,dim=-1,eps=1e-6)
    norm=torch.linalg.vector_norm(g,dim=-1);eik=(norm-1).abs()
    if w.boundary_tension>0:
        hvall=torch.autograd.grad(g.sum(),q,create_graph=training,retain_graph=True)[0].float()
        tension=hvall[a:z].square().sum(dim=1)
    else:tension=by*0
    bcounts=torch.bincount(bs,minlength=8).float()
    def bb(v):
        sums=torch.zeros(8,device=pred.device);sums.scatter_add_(0,bs,v)
        return _balanced(sums,bcounts,world)[0]
    comp["boundary_grad"]=bb(grad_term);comp["boundary_eikonal"]=bb(eik);comp["boundary_tension"]=bb(tension)
    stats["boundary_cos"]=(1-comp["boundary_grad"]).detach();stats["boundary_grad_norm"]=bb(norm).detach()
    total=(w.sign_sensor*comp["sign_sensor"]+w.sign_union*comp["sign_union"]+w.consistency*comp["consistency"]
          +w.v3_sensor_sdf*comp["v3_sensor_sdf"]+w.v3_union_sdf*comp["v3_union_sdf"]+w.v4_sensor_sdf*comp["v4_sensor_sdf"]
          +w.boundary_grad*comp["boundary_grad"]+w.boundary_eikonal*comp["boundary_eikonal"]+w.boundary_tension*comp["boundary_tension"])
    stats.update({k:v.detach() for k,v in comp.items()})
    return total,stats

def reduce_stats(stats,world):
    out={}
    for k,v in stats.items():
        t=v.detach().float().clone();torch.distributed.all_reduce(t)
        out[k]=float((t/world).item())
    return out

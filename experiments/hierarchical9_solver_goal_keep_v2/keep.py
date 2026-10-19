"""Soft preservation on successful training trajectories; not a success guarantee."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from k2_common import read, stage, require

def actions(q, y, grad, mask, lo, hi):
    """Same individual capped projection/ascent formulas as the unchanged runtime."""
    g=grad*mask;g2=g.square().sum(-1,keepdim=True)
    raw=y[:,None]*g/g2.clamp_min(1e-8)
    cap=(.25/raw.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
    proj=-.5*raw*cap
    proj=torch.where(g2>=1e-12,proj,torch.zeros_like(proj))
    norm=g.norm(dim=-1,keepdim=True)
    asc=.05*g/norm.clamp_min(1e-8)
    asc=torch.where(norm>=1e-8,asc,torch.zeros_like(asc))
    clamp=lambda z:torch.maximum(torch.minimum(z,hi),lo)
    return clamp(q+proj)-q,clamp(q+asc)-q

def predict(model,x,q,s,mask,lo,hi,training):
    q=q.detach().clone().requires_grad_(True)
    with torch.enable_grad(),torch.autocast(q.device.type,enabled=False):
        y=model.forward_sensor(torch.cat((x.float(),q.float()),1),s).float()
        g=torch.autograd.grad(y.sum(),q,create_graph=training,retain_graph=training)[0]
        p,a=actions(q,y,g,mask,lo,hi)
    return y,p,a

def soft_errors(y,p,a,ty,tp,ta,cfg):
    """Dead zones allow useful drift; each quantity has its own unit/scale."""
    value=F.relu((y-ty).abs()-cfg['score_deadzone'])/ty.abs().clamp_min(cfg['score_scale_floor'])
    proj=F.relu((p-tp).norm(dim=-1)-cfg['action_deadzone_rad'])/cfg['action_scale_rad']
    asc=F.relu((a-ta).norm(dim=-1)-cfg['action_deadzone_rad'])/cfg['action_scale_rad']
    error=torch.stack((value,proj,asc),1)
    return F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none').mean(1),error

def objective(model,batch,masks,lo,hi,cfg,training=True):
    losses=[];metrics=[]
    for s,b in enumerate(batch):
        y,p,a=predict(model,b['x'],b['q'],s,masks[s],lo,hi,training)
        loss,err=soft_errors(y,p,a,b['score'],b['project'],b['ascent'],cfg)
        require(torch.isfinite(loss).all(),'Nonfinite keep loss')
        losses.append(loss.mean())
        metrics.append(torch.cat((loss.detach().mean().reshape(1),err.detach().mean(0))))
    return torch.stack(losses).mean(),torch.stack(metrics)

class ReplayBank:
    """Sample trajectories first, then visits: long solver paths get no extra weight."""
    def __init__(self,folder,expected_teacher):
        from pathlib import Path
        self.folder=Path(folder);m=stage(self.folder)
        require(m['teacher_sha256']==expected_teacher,'Wrong replay teacher')
        self.manifest=m;self.a={};self.groups={}
        for split in ('train','val'):
            with np.load(self.folder/(split+'.npz'),allow_pickle=False) as z:
                a={k:z[k] for k in z.files}
            require(a['q'].shape[1:]==(7,) and a['ptr'][0]==0 and a['ptr'][-1]==len(a['q']),'Bad replay shape/ptr')
            require(np.all(np.diff(a['ptr'])>0),'Empty replay trajectory')
            require(all(np.isfinite(a[k]).all() for k in ('x','q','score','project','ascent')),'Nonfinite replay values')
            require(len(a['ptr'])==len(a['sensor'])+1 and np.isin(a['sensor'],np.arange(8)).all() and np.isin(a['cohort'],[0,1]).all(),'Bad replay trajectory labels')
            self.a[split]=a
            cell=a['sensor']*2+a['cohort']
            self.groups[split]=[np.flatnonzero(cell==c) for c in range(16)]
            require(all(len(g) for g in self.groups[split]),f'Empty success stratum {split}')

    def draw(self,split,step,rank,cfg,device):
        a=self.a[split];batch=[];k=cfg['replay_per_cell_per_rank']
        for sensor in range(8):
            rng=np.random.default_rng(np.random.SeedSequence([cfg['replay_seed'],step,rank,sensor,split=='val']))
            tids=[];rows=[]
            for cohort in (0,1):
                pool=self.groups[split][sensor*2+cohort]
                chosen=pool[rng.integers(len(pool),size=k)]
                for tid in chosen:
                    tids.append(tid);rows.append(rng.integers(a['ptr'][tid],a['ptr'][tid+1]))
            def t(v):return torch.as_tensor(np.asarray(v).copy(),device=device,dtype=torch.float32)
            batch.append(dict(x=t(a['x'][tids]),q=t(a['q'][rows]),score=t(a['score'][rows]),
                              project=t(a['project'][rows]),ascent=t(a['ascent'][rows])))
        return batch

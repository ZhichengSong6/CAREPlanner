"""Original R1 optimizer step + a separately reduced task-outcome gradient.

The auxiliary functional parameters are cloned leaves, so autograd.grad never
enters the DDP reducer. Only one optimizer/scaler update occurs per base batch.
Both finite-gradient retry and checkpoint counters count successful updates.
"""
from __future__ import annotations
import contextlib
import math
import torch
import torch.distributed as dist
from torch.func import functional_call
from rollout import outcome_loss

METRICS = ('geom_loss','outside_fov_pass','positive_fov_retained','outside_margin_m',
           'positive_margin_m','target_margin_pass','initial_network_accepted',
           'valid_rows','rejected_witness_fraction')

def coefficient(step, cfg):
    return cfg['weight'] * max(0., min(1., (step-cfg['warmup'])/cfg['ramp']))

def functional_outcome(model, batch, geometry, masks, lo, hi, *, training=True):
    if list(model.buffers()):raise ValueError('R1 should not have mutable buffers')
    pars={k:p.detach().clone().requires_grad_(True) for k,p in model.named_parameters()}
    x,qm,qp,s=batch
    xx,ss=torch.cat([x,x]),torch.cat([s,s])
    def score(q):
        return functional_call(model,pars,(torch.cat([xx,q],1),),strict=True).gather(1,(ss+1)[:,None])[:,0]
    loss, stats=outcome_loss(score,geometry,x,qm,qp,s,masks,lo,hi,training=training)
    return loss,stats,pars

def reduce_aux(loss,stats,pars,weight,scale,world):
    count=stats[7].detach().clone()
    if world>1:dist.all_reduce(count)
    multiplier=weight*world*stats[7]/count.clamp_min(1)
    grads=torch.autograd.grad(loss*multiplier*scale,tuple(pars.values()),allow_unused=True)
    flat=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                    for p,g in zip(pars.values(),grads)])
    if world>1:
        dist.all_reduce(flat);flat/=world
    return flat

def add_flat(parameters,flat):
    offset=0
    for p in parameters:
        g=flat[offset:offset+p.numel()].view_as(p)
        if p.grad is None:p.grad=g.clone()
        else:p.grad.add_(g)
        offset+=p.numel()
    if offset!=flat.numel():raise ValueError('Gradient layout mismatch')

class Step:
    def __init__(self, original, base, sampler, geometry, masks, lo, hi, cfg, start=0):
        self.original,self.base,self.sampler=original,base,sampler
        self.geometry,self.masks,self.lo,self.hi=geometry,masks,lo,hi
        self.cfg,self.step=cfg,start

    def observe(self,model,batch,device,world,weight):
        loss,stats,pars=functional_outcome(model,batch,self.geometry,self.masks,self.lo,self.hi,training=False)
        result=stats.detach().double()
        n=result[7].clone();num=result[:7]*n
        tail=torch.stack((n,result[8]))
        if world>1:dist.all_reduce(num);dist.all_reduce(tail)
        vals=(num/tail[0].clamp_min(1)).tolist()
        return dict(zip(METRICS[:7],vals),valid_rows=float(tail[0]),
                    rejected_witness_fraction=float(tail[1]/world),weight=weight)

    def __call__(self,ddp,batches,counts,weights,args,optimizer=None,scaler=None):
        world=dist.get_world_size();rank=dist.get_rank();device=counts.device
        if optimizer is None:
            # Original validation objective is never replaced by new geometry loss.
            result=self.original(ddp,batches,counts,weights,args)
            batch=self.sampler.batch('val',0,rank,device,self.cfg['pairs_per_sensor_per_rank'])
            result['outcome']=self.observe(ddp.module,batch,device,world,coefficient(self.step,self.cfg))
            return result
        self.step+=1
        weight=coefficient(self.step,self.cfg)
        if weight==0:
            return self.original(ddp,batches,counts,weights,args,optimizer,scaler)
        batch=self.sampler.batch('train',self.step,rank,device,self.cfg['pairs_per_sensor_per_rank'])
        base=self.base
        for attempt in range(args.max_amp_retries+1):
            optimizer.zero_grad(set_to_none=True);ddp.train(True)
            totals=torch.zeros((9,len(base.STAT_NAMES)),dtype=torch.float64,device=device)
            for i,b in enumerate(batches):
                context=ddp.no_sync() if i<len(batches)-1 else contextlib.nullcontext()
                with context, torch.enable_grad():
                    with base.amp_context(args.amp):
                        loss,stats=base.loss_for_microbatch(ddp,*b,counts,weights,world_size=world,training=True)
                    base.synchronized_check(torch.isfinite(loss.detach()) & torch.isfinite(stats).all(),
                                            'Nonfinite original R1 objective',device)
                    scaler.scale(loss).backward()
                totals+=stats;del loss,stats
            with torch.enable_grad(),torch.autocast(device.type,enabled=False):
                aux,aux_stats,pars=functional_outcome(ddp.module,batch,self.geometry,self.masks,self.lo,self.hi)
                base.synchronized_check(torch.isfinite(aux.detach()) & torch.isfinite(aux_stats).all(),
                                        'Nonfinite candidate geometry objective',device)
                scale=float(scaler.get_scale())
                flat=reduce_aux(aux,aux_stats,pars,weight,scale,world)
                aux_norm=float(flat.norm().detach())/scale
                add_flat(list(ddp.module.parameters()),flat)
            del flat,aux,pars
            scaler.unscale_(optimizer)
            grads=[p.grad for p in ddp.parameters() if p.grad is not None]
            finite=torch.stack([torch.isfinite(g).all() for g in grads]).all().int()
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if finite.item():
                scaler.step(optimizer);scaler.update();break
            if args.amp!='fp16' or attempt==args.max_amp_retries:
                raise FloatingPointError('Nonfinite combined gradients; no false final checkpoint')
            scaler.update(new_scale=scaler.get_scale()/2)
            base.log(f'[outcome AMP retry] step={self.step} attempt={attempt+1}; same inputs')
        dist.all_reduce(totals)
        if not torch.equal(totals[:,0].float(),counts):raise ValueError('Original R1 denominator mismatch')
        result=base.summarize(totals,weights)
        if not math.isfinite(result['loss']):raise FloatingPointError('Reduced original loss not finite')
        result.update(amp_retries=attempt,grad_scale=scaler.get_scale())
        if self.step==1 or self.step%args.log_every==0 or self.step==args.steps:
            aux_stats=aux_stats.detach().double();n=aux_stats[7].clone()
            nums=aux_stats[:7]*n;tail=torch.stack((n,aux_stats[8]))
            dist.all_reduce(nums);dist.all_reduce(tail)
            result['outcome']=dict(zip(METRICS[:7],(nums/tail[0].clamp_min(1)).tolist()),
                 valid_rows=float(tail[0]),rejected_witness_fraction=float(tail[1]/world),
                 weight=weight,aux_parameter_grad_norm=aux_norm)
            base.log(f'[outcome] step={self.step} '+str(result['outcome']))
        return result

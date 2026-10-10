"""Boundary-first losses; all quantities use the original raw-radian joint metric."""
from __future__ import annotations
import torch
import torch.distributed as dist
import torch.nn.functional as F


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def cell_reduce(values: torch.Tensor, cell: torch.Tensor, ncells: int):
    """Return globally normalized differentiable column losses and sufficient stats.

    Multiply LOCAL numerators by world size because parameter gradients are later
    averaged across ranks. Invalid observations must be removed BEFORE this call.
    """
    if values.ndim == 1:
        values = values[:, None]
    sums = values.new_zeros((ncells, values.shape[1])).index_add(0, cell.long(), values)
    counts = torch.bincount(cell.long(), minlength=ncells).to(values.dtype)
    packed = torch.cat([counts[:, None], sums.detach()], 1)
    if dist.is_initialized():
        dist.all_reduce(packed)
    counts = packed[:, 0]
    active = counts > 0
    per_cell = packed[:, 1:] / counts.clamp_min(1)[:, None]
    if active.any():
        loss = world_size()*(sums[active]/counts[active, None]).mean(0)
    else:
        loss = sums.sum(0)*0
    return loss, per_cell, counts


def macro(values, counts):
    active = counts > 0
    return values[active].mean(0) if active.any() else values.sum(0)*0


def balanced_sign(y, target, temperature):
    cols = y.shape[1]
    sign = target.float()
    classes = (sign > 0).long()
    ids = (torch.arange(cols, device=y.device)[None, :]*2 + classes).reshape(-1)
    logistic = temperature*F.softplus(-sign*y/temperature)
    correct = ((y >= 0) == (sign > 0)).float().detach()
    loss, means, count = cell_reduce(torch.stack([logistic, correct], -1).reshape(-1, 2), ids, 2*cols)
    total = count.sum().clamp_min(1)
    pos, neg = count[1::2] > 0, count[::2] > 0
    metrics = {
        "balanced_accuracy": macro(means[:, 1:2], count)[0],
        "raw_accuracy": (means[:, 1]*count).sum()/total,
        "positive_recall": means[1::2, 1][pos].mean() if pos.any() else y.new_tensor(float("nan")),
        "negative_recall": means[::2, 1][neg].mean() if neg.any() else y.new_tensor(float("nan")),
        "positive_fraction": count[1::2].sum()/total,
        "active_class_cells": (count > 0).sum(),
    }
    return loss[0], metrics


def huber(z):
    return F.smooth_l1_loss(z, torch.zeros_like(z), beta=1., reduction="none")


def loss_and_metrics(model, batches: dict, cfg: dict, *, training: bool):
    # Scalar samples do not need input-gradient graphs.
    scalar_kinds = ("global", "v3", "v4")
    sizes = [len(batches[k]["inputs"]) for k in scalar_kinds]
    scalar = model(torch.cat([batches[k]["inputs"] for k in scalar_kinds], 0))
    pg, pv, pt = torch.split(scalar, sizes)
    metrics, components = {}, {}
    components["sign_sensor"], sm = balanced_sign(pg[:, 1:], batches["global"]["sensor_sign"], cfg["sign_temperature"])
    components["sign_union"], um = balanced_sign(pg[:, :1], batches["global"]["union_sign"][:, None], cfg["sign_temperature"])
    metrics.update({"sensor_"+k: v for k, v in sm.items()})
    metrics.update({"union_"+k: v for k, v in um.items()})

    # V3: valid sensor distances only, not the unverified gradient or union max.
    b = batches["v3"]
    row, sensor = torch.where(b["sensor_value_mask"].bool())
    err = pv[row, sensor+1] - b["sensor_value"][row, sensor].float()
    ll, means, count = cell_reduce(torch.stack([huber(err/cfg["v3_scale_rad"]),
                                              err.detach().abs(), err.detach().square()], 1), sensor, 8)
    components["v3"] = ll[0]
    mm = macro(means, count)
    metrics.update(v3_mae_rad=mm[1], v3_rmse_rad=mm[2].sqrt(), v3_valid_values=count.sum())

    # V4: equal weight for each (sensor, offset), including offset zero.
    b = batches["v4"]
    row = torch.arange(len(b["inputs"]), device=pt.device)
    yp = pt[row, b["sensor"].long()+1]
    target = b["value"].float()
    err = yp-target
    ll, tm, tc = cell_reduce(torch.stack([huber(err/cfg["local_scale_rad"]), err.detach().abs(),
                        err.detach().square(), target.square(), ((yp >= 0) == (target > 0)).float().detach()], 1), b["cell"], 56)
    components["local"] = ll[0]
    tmacro = macro(tm, tc)
    metrics.update(local_mae_rad=tmacro[1], local_rmse_rad=tmacro[2].sqrt(),
                   local_zero_predictor_mse=tmacro[3])
    nz = torch.arange(56, device=pt.device) % 7 != 3
    metrics["local_nonzero_sign_accuracy"] = macro(tm[nz, 4:5], tc[nz])[0]
    metrics["local_mse_over_zero_baseline"] = tmacro[2]/tmacro[3].clamp_min(1e-12)
    metrics["local_cell_rmse_rad"] = tm[:, 2].sqrt()

    # Paired V4: exact same (x, sensor, q-star, source-slot) and radius.
    # Both q's are original, separately geometry-checked production rows.
    b = batches["pair"]
    pn = len(b["sensor"])
    pp = model(torch.cat([b["plus_inputs"], b["minus_inputs"]], dim=0))
    row = torch.arange(pn, device=pp.device)
    sid = b["sensor"].long()
    fplus = pp[row, sid+1]
    fminus = pp[row+pn, sid+1]
    radius = torch.tensor([.005, .01, .02], device=pp.device, dtype=pp.dtype)[b["radius_id"].long()]
    slope = (fplus-fminus)/(2*radius)
    err_plus, err_minus = fplus-radius, fminus+radius
    local_scale = cfg["local_scale_rad"]
    side_loss = .5*(huber(err_plus/local_scale)+huber(err_minus/local_scale))
    slope_loss = huber(slope-1)
    temperature = cfg["pair_sign_temperature"]
    margin = .5*radius
    margin_loss = .5*(F.softplus((margin-fplus)/temperature)+
                     F.softplus((margin+fminus)/temperature))
    both_sign = ((fplus > 0) & (fminus < 0)).float().detach()
    pair_cell = sid*3+b["radius_id"].long()
    pl, pm, pc = cell_reduce(torch.stack([side_loss,slope_loss,margin_loss,
                      (slope.detach()-1).abs(),both_sign,
                      .5*(err_plus.detach().square()+err_minus.detach().square()),
                      slope.detach()],dim=1),pair_cell,24)
    components["pair_side"],components["pair_slope"],components["pair_margin"] = pl[:3]
    pmean = macro(pm,pc)
    metrics.update(pair_slope_mae=pmean[3],pair_both_sign_accuracy=pmean[4],
                   pair_side_rmse_rad=pmean[5].sqrt(),pair_slope_mean=pmean[6],
                   pair_cell_slope_mae=pm[:,3],pair_cell_both_sign_accuracy=pm[:,4],
                   pair_count=pc.sum())

    # Boundary: zero AND full seven-joint normal on the SAME sampled anchors.
    b = batches["boundary"]
    q = b["q"].float().detach().clone().requires_grad_(True)
    pred = model(torch.cat([b["inputs"][:, :3].detach(), q], 1))
    sid = b["sensor"].long()
    val = pred[torch.arange(len(sid), device=q.device), sid+1]
    grad = torch.autograd.grad(val.sum(), q, create_graph=training, retain_graph=training)[0]
    truth = b["grad"].float()
    vector_error = (grad-truth).square().sum(1)
    gnorm = grad.detach().norm(dim=1)
    cos = F.cosine_similarity(grad.detach(), truth, dim=1, eps=1e-8)
    ll, bm, bc = cell_reduce(torch.stack([huber(val/cfg["local_scale_rad"]), vector_error,
              val.detach().abs(), val.detach().square(), cos, gnorm, (gnorm-1).abs()], 1), sid, 8)
    components["boundary_zero"], components["normal"] = ll[0], ll[1]
    mm = macro(bm, bc)
    metrics.update(boundary_zero_mae_rad=mm[2], boundary_zero_rmse_rad=mm[3].sqrt(),
                   boundary_cos=mm[4], boundary_grad_norm=mm[5], boundary_eikonal_diagnostic=mm[6],
                   boundary_sensor_zero_mae_rad=bm[:, 2], boundary_sensor_cos=bm[:, 4])
    # No explicit Eikonal or Hessian/tension penalties; vector supervision is local.
    total = sum(cfg["weights"][key]*value for key, value in components.items())
    detached = torch.stack([components[k].detach() for k in components])
    if dist.is_initialized():
        dist.all_reduce(detached)
        detached /= world_size()
    for k, value in zip(components, detached):
        metrics["loss_"+k] = value
    metrics["selection_score"] = sum(cfg["weights"][k]*metrics["loss_"+k] for k in components)
    return total, metrics


def average_parameter_gradients(model) -> None:
    """Explicit synchronous data parallelism, after input-derivative backward.

    Avoid DDP reducer hooks around autograd.grad. Every rank performs one same-
    ordered reduction, including zeros for parameters absent on a local rank.
    """
    params = list(model.parameters())
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    if dist.is_initialized():
        dist.all_reduce(flat)
        flat /= world_size()
    if not torch.isfinite(flat).all():
        raise FloatingPointError("Nonfinite reduced parameter gradients")
    cursor = 0
    for p in params:
        p.grad = flat[cursor:cursor+p.numel()].view_as(p).clone()
        cursor += p.numel()

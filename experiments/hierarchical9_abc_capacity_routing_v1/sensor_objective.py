"""Original hierarchical9 per-sensor objective, routed only to sensor-specific params.

This deliberately excludes union supervision and union/sensor consistency from the
training signal.  The frozen union path is still evaluated by the unchanged global
objective during validation and final evaluation.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

STAT_NAMES = ("count", "sdf", "grad_loss", "eikonal", "tension", "abs_error", "sign_correct", "grad_norm")
COUNT, SDF, GRAD, EIK, TENSION, ABS, SIGN, NORM = range(len(STAT_NAMES))


def counts_from_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[1] != 8:
        raise ValueError("Expected sensor mask [N,8]")
    return mask.sum(dim=0).to(torch.float32)


def loss_for_microbatch(model, inputs, sensor_target, sensor_grad, sensor_mask,
                        global_counts, weights, *, training: bool):
    if inputs.ndim != 2 or inputs.shape[1] != 10:
        raise ValueError("Expected [N,10] inputs")
    if global_counts.shape != (8,):
        raise ValueError("Expected eight sensor denominators")
    mask = sensor_mask.bool()
    target = torch.where(mask, sensor_target.float(), 0.0)
    target_grad = torch.where(mask[..., None], sensor_grad.float(), 0.0)
    q = inputs[:, 3:].detach().float().clone().requires_grad_(True)
    x = inputs[:, :3].detach().float()
    pred = model.forward_sensors(torch.cat((x, q), dim=1))
    if pred.shape != target.shape:
        raise RuntimeError(f"Prediction {pred.shape} != target {target.shape}")

    with torch.autocast(device_type=inputs.device.type, enabled=False):
        pred = pred.float()
        denom = global_counts.to(device=pred.device, dtype=torch.float32)
        active = int((denom > 0).sum().item())
        if active == 0:
            raise RuntimeError("No globally supervised sensor")
        stats = torch.zeros((8, len(STAT_NAMES)), dtype=torch.float64, device=pred.device)
        total = (pred * 0.0).sum()
        for s in range(8):
            valid = mask[:, s]
            if denom[s] <= 0 or not valid.any():
                continue
            y = pred[:, s]
            value_error = y[valid] - target[valid, s]
            sdf_sum = value_error.square().sum(dtype=torch.float32)
            need_second = weights.tension > 0
            grad_q = torch.autograd.grad(
                y.sum(), q, create_graph=(training or need_second), retain_graph=True
            )[0].float()
            cosine = F.cosine_similarity(grad_q[valid], target_grad[valid, s], dim=-1, eps=1e-6)
            norm = torch.linalg.vector_norm(grad_q[valid], dim=-1)
            grad_sum = (1.0 - cosine).sum(dtype=torch.float32)
            eik_sum = (norm - 1.0).abs().sum(dtype=torch.float32)
            if need_second:
                hessian_vector = torch.autograd.grad(
                    grad_q.sum(), q, create_graph=training, retain_graph=True
                )[0].float()
                tension_sum = hessian_vector[valid].square().sum(dtype=torch.float32)
            else:
                tension_sum = y.sum() * 0.0
            objective = (weights.sdf * sdf_sum + weights.grad * grad_sum
                         + weights.eikonal * eik_sum + weights.tension * tension_sum)
            total = total + objective / denom[s] / float(active)
            with torch.no_grad():
                stats[s] = torch.stack((
                    valid.sum().float(), sdf_sum.detach(), grad_sum.detach(), eik_sum.detach(),
                    tension_sum.detach(), value_error.abs().sum(),
                    ((y[valid] >= 0) == (target[valid, s] >= 0)).float().sum(), norm.sum(),
                )).double()
        return total, stats


def summarize(stats: torch.Tensor, weights) -> dict:
    values = stats.detach().cpu().double()
    heads, objectives = {}, []
    for s in range(8):
        count = int(values[s, COUNT].item())
        if not count:
            heads[f"s{s}"] = {"count": 0}
            continue
        mean = values[s] / count
        objective = float(weights.sdf * mean[SDF] + weights.grad * mean[GRAD]
                          + weights.eikonal * mean[EIK] + weights.tension * mean[TENSION])
        heads[f"s{s}"] = dict(
            count=count,
            mae=float(mean[ABS]),
            rmse=math.sqrt(max(0.0, float(mean[SDF]))),
            sdf_loss=float(mean[SDF]),
            grad_loss=float(mean[GRAD]),
            grad_cosine=1.0-float(mean[GRAD]),
            eikonal_loss=float(mean[EIK]),
            tension_loss=float(mean[TENSION]),
            grad_norm=float(mean[NORM]),
            sign_accuracy=float(mean[SIGN]),
            objective=objective,
        )
        objectives.append(objective)
    return {"heads": heads, "loss": sum(objectives)/max(1, len(objectives))}

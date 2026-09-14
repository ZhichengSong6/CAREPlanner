"""Auxiliary objectives for E1/E2 shared routing and E2 hard replay.

The original hierarchical9 global objective remains authoritative and is still
used for every uniform batch.  These helpers do two narrowly defined things:
1) provide an unbiased one-sensor estimate of the original *sensor* objective
   for the shared-early trunk in E1/E2, avoiding simultaneous sensor-sensor
   gradients on that trunk;
2) provide asymmetric value-only replay on runtime-mined actual-FOV candidates.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch.nn import functional as F


def selected_sensor_shared_objective(
    model,
    inputs: torch.Tensor,
    sensor_target: torch.Tensor,
    sensor_grad: torch.Tensor,
    sensor_mask: torch.Tensor,
    global_counts9: torch.Tensor,
    weights,
    sensor_id: int,
) -> torch.Tensor:
    """Full original objective for one sensor, normalized as an unbiased estimator.

    The standard objective contributes mean_s L_s to shared parameters. Selecting
    one sensor uniformly/round-robin and using L_s (not L_s/8) has that same
    expectation while preventing same-step sensor-sensor interference.
    """
    if not 0 <= sensor_id < 8:
        raise ValueError(sensor_id)
    denom = global_counts9[sensor_id + 1].float()
    if denom <= 0:
        return inputs.sum() * 0.0
    valid = sensor_mask[:, sensor_id].bool()
    if not valid.any():
        return inputs.sum() * 0.0

    q = inputs[:, 3:].detach().float().clone().requires_grad_(True)
    x = inputs[:, :3].detach().float()
    y = model.forward_sensor(torch.cat((x, q), 1), sensor_id, freeze_sensor_early=False).float()
    target = sensor_target[:, sensor_id].float()
    target_grad = sensor_grad[:, sensor_id].float()
    value_error = y[valid] - target[valid]
    sdf = value_error.square().sum(dtype=torch.float32)
    grad_q = torch.autograd.grad(y.sum(), q, create_graph=True, retain_graph=True)[0].float()
    cosine = F.cosine_similarity(grad_q[valid], target_grad[valid], dim=-1, eps=1e-6)
    grad = (1.0 - cosine).sum(dtype=torch.float32)
    norm = torch.linalg.vector_norm(grad_q[valid], dim=-1)
    eik = (norm - 1.0).abs().sum(dtype=torch.float32)
    if weights.tension > 0:
        hvec = torch.autograd.grad(grad_q.sum(), q, create_graph=True, retain_graph=True)[0].float()
        tension = hvec[valid].square().sum(dtype=torch.float32)
    else:
        tension = y.sum() * 0.0
    objective = (weights.sdf*sdf + weights.grad*grad
                 + weights.eikonal*eik + weights.tension*tension) / denom
    return weights.sensor_objective * objective


def _asymmetric_values(pred: torch.Tensor, labels: torch.Tensor, fn_ratio: float):
    if not math.isfinite(fn_ratio) or fn_ratio < 0:
        raise ValueError("fn_ratio must be finite and nonnegative")
    outside = labels < 0
    inside = labels > 0
    if torch.any(~(outside | inside)):
        raise ValueError("Replay labels must be +/-1")
    fp = torch.relu(pred[outside]).square().sum() if outside.any() else pred.sum()*0.0
    fn = torch.relu(-pred[inside]).square().sum() if inside.any() else pred.sum()*0.0
    count = max(1, int(len(pred)))
    loss = (fp + fn_ratio*fn) / float(count)
    with torch.no_grad():
        predicted_positive = pred >= 0
        fp_n = int((outside & predicted_positive).sum().item())
        fn_n = int((inside & ~predicted_positive).sum().item())
    return loss, fp_n, fn_n, int(outside.sum()), int(inside.sum())


def replay_private_loss(model, replay_batches: Mapping[int, dict], fn_ratio: float):
    """Average replay loss across active sensors; block only early parameter grads."""
    losses = []
    stats = {"count": 0, "fp": 0, "fn": 0, "outside": 0, "inside": 0}
    for s in sorted(replay_batches):
        batch = replay_batches[s]
        inp, labels = batch["inputs"], batch["labels"]
        if not len(inp):
            continue
        pred = model.forward_sensor(inp, s, freeze_sensor_early=True).float()
        loss, fp, fn, outside, inside = _asymmetric_values(pred, labels, fn_ratio)
        losses.append(loss)
        stats["count"] += len(inp); stats["fp"] += fp; stats["fn"] += fn
        stats["outside"] += outside; stats["inside"] += inside
    if not losses:
        device = next(model.parameters()).device
        return torch.zeros((), device=device), stats
    return torch.stack(losses).mean(), stats


def replay_shared_sensor_loss(model, replay_batches: Mapping[int, dict], sensor_id: int, fn_ratio: float):
    """One-sensor replay objective for shared early params; unbiased across sensors."""
    batch = replay_batches.get(sensor_id)
    if batch is None or not len(batch["inputs"]):
        return next(model.parameters()).sum() * 0.0
    pred = model.forward_sensor(batch["inputs"], sensor_id, freeze_sensor_early=False).float()
    loss, *_ = _asymmetric_values(pred, batch["labels"], fn_ratio)
    return loss

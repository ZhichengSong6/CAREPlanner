"""Exact globally normalized baseline objective, with FP32 loss arithmetic.

Keep the existing scientific objective, including absolute (NOT squared)
Eikonal and directional-Hessian tension. These are not new boundary losses.
All normalization counts and loss reductions stay in FP32 under AMP.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    sdf: float = 5.0
    grad: float = 0.1
    eikonal: float = 0.01
    tension: float = 0.01
    union_objective: float = 1.0
    sensor_objective: float = 1.0
    consistency: float = 0.1

    def validate(self) -> None:
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if self.sdf <= 0 or self.union_objective <= 0 or self.sensor_objective <= 0:
            raise ValueError("Both union and sensor heads require value supervision")


# Nine rows: U,S0,...,S7. The last two sums are used only in the union row.
STAT_NAMES = (
    "count", "sdf", "grad_loss", "eikonal", "tension", "abs_error",
    "sign_correct", "grad_norm", "consistency", "winner_correct",
)
COUNT, SDF, GRAD, EIK, TENSION, ABS, SIGN, NORM, CONS, WIN = range(10)


def counts_from_mask(sensor_mask: torch.Tensor) -> torch.Tensor:
    """Local counts. SUM these across ranks before using as denominators."""
    if sensor_mask.ndim != 2 or sensor_mask.shape[1] != 8:
        raise ValueError("Expected sensor mask [N,8]")
    return torch.cat((sensor_mask.any(dim=1).sum().reshape(1),
                      sensor_mask.sum(dim=0))).to(torch.float32)


def supervised_targets(
    sensor_target: torch.Tensor,
    sensor_grad: torch.Tensor,
    sensor_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use GT winner for union supervision; mask unavailable sensors BEFORE max."""
    if sensor_target.ndim != 2 or sensor_target.shape[1] != 8:
        raise ValueError("Expected sensor_target:[N,8]")
    n = sensor_target.shape[0]
    if sensor_grad.shape != (n, 8, 7) or sensor_mask.shape != (n, 8):
        raise ValueError("Expected sensor_grad:[N,8,7], sensor_mask:[N,8]")
    mask = sensor_mask.bool()
    # Invalid targets may be placeholders; NEVER include them in arithmetic.
    target = torch.where(mask, sensor_target.float(), 0.0)
    grad = torch.where(mask[..., None], sensor_grad.float(), 0.0)
    row_valid = mask.any(dim=1)
    winner = target.masked_fill(~mask, -torch.inf).argmax(dim=1)
    union = target.gather(1, winner[:, None]).squeeze(1)
    union_grad = grad.gather(1, winner[:, None, None].expand(-1, 1, 7)).squeeze(1)
    union = torch.where(row_valid, union, 0.0)
    union_grad = torch.where(row_valid[:, None], union_grad, 0.0)
    return (
        torch.cat((union[:, None], target), dim=1),
        torch.cat((union_grad[:, None, :], grad), dim=1),
        torch.cat((row_valid[:, None], mask), dim=1),
        winner,
    )


def loss_for_microbatch(
    model: nn.Module,
    inputs: torch.Tensor,
    sensor_target: torch.Tensor,
    sensor_grad: torch.Tensor,
    sensor_mask: torch.Tensor,
    global_counts: torch.Tensor,
    weights: LossWeights,
    *,
    world_size: int = 1,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call within optional autocast; call returned_loss.backward() OUTSIDE it.

    Accumulate across microbatches WITHOUT dividing by number of microbatches.
    Counts refer to the WHOLE global optimizer batch. DDP averages parameter
    gradients; multiplication by world_size recovers the desired global sum.
    autograd.grad here is with respect to INPUT q, not DDP parameters. Parameter
    gradients are produced by the final backward() for DDP synchronization.
    """
    if inputs.ndim != 2 or inputs.shape[1] != 10:
        raise ValueError("Expected input [N,10]")
    if global_counts.shape != (9,) or global_counts[0] <= 0:
        raise ValueError("Expected positive union count and 9 global counts")
    target, target_grad, mask, gt_winner = supervised_targets(
        sensor_target, sensor_grad, sensor_mask
    )
    q = inputs[:, 3:10].detach().float().clone().requires_grad_(True)
    x = inputs[:, :3].detach().float()
    pred = model(torch.cat((x, q), dim=-1))
    if pred.shape != target.shape:
        raise RuntimeError(f"Prediction {pred.shape} != target {target.shape}")

    # Critical AMP fix: counts can reach 400,000; FP16 max is only 65,504.
    # Casting predictions ALSO prevents a half-precision sum overflow.
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        pred = pred.float()
        denom = global_counts.to(device=pred.device, dtype=torch.float32)
        active_sensor_n = int((denom[1:] > 0).sum().item())
        if active_sensor_n == 0:
            raise RuntimeError("No globally supervised sensor")
        stats = torch.zeros((9, len(STAT_NAMES)), dtype=torch.float64, device=pred.device)
        # Connect every output even when a rank has no valid rows for a head.
        # This avoids unused-parameter mismatches in DDP; do not detach this zero.
        total = (pred * 0.0).sum()
        for head in range(9):
            valid = mask[:, head]
            if denom[head] <= 0 or not valid.any():
                continue
            y = pred[:, head]
            value_error = y[valid] - target[valid, head]
            sdf_sum = value_error.square().sum(dtype=torch.float32)
            need_second = weights.tension > 0
            grad_q = torch.autograd.grad(
                y.sum(), q, create_graph=(training or need_second), retain_graph=True
            )[0].float()
            cosine = F.cosine_similarity(
                grad_q[valid], target_grad[valid, head], dim=-1, eps=1e-6
            )
            norm = torch.linalg.vector_norm(grad_q[valid], dim=-1)
            grad_sum = (1.0 - cosine).sum(dtype=torch.float32)
            eik_sum = (norm - 1.0).abs().sum(dtype=torch.float32)
            if need_second:
                # EXACT upstream tension: ||H_q^T 1||^2, not ||H_q||_F^2.
                hessian_vector = torch.autograd.grad(
                    grad_q.sum(), q, create_graph=training, retain_graph=True
                )[0].float()
                tension_sum = hessian_vector[valid].square().sum(dtype=torch.float32)
            else:
                tension_sum = y.sum() * 0.0
            objective = (weights.sdf * sdf_sum + weights.grad * grad_sum
                         + weights.eikonal * eik_sum + weights.tension * tension_sum)
            factor = (weights.union_objective if head == 0 else
                      weights.sensor_objective / active_sensor_n)
            total = total + factor * objective / denom[head]
            with torch.no_grad():
                stats[head] = torch.stack((
                    valid.sum().float(), sdf_sum.detach(), grad_sum.detach(),
                    eik_sum.detach(), tension_sum.detach(), value_error.abs().sum(),
                    ((y[valid] >= 0) == (target[valid, head] >= 0)).float().sum(),
                    norm.sum(), y.new_zeros(()), y.new_zeros(()),
                )).double()

        # Select valid rows BEFORE max to avoid (-inf) arithmetic on empty rows.
        row_valid = mask[:, 0]
        if row_valid.any():
            ranked = pred[row_valid, 1:].masked_fill(~mask[row_valid, 1:], -torch.inf)
            sensor_max, pred_winner = ranked.max(dim=1)
            cons_sum = (pred[row_valid, 0] - sensor_max).square().sum(dtype=torch.float32)
            total = total + weights.consistency * cons_sum / denom[0]
            stats[0, CONS] = cons_sum.detach().double()
            stats[0, WIN] = (pred_winner == gt_winner[row_valid]).sum().double()
        return total * float(world_size), stats


def summarize(stats: torch.Tensor, weights: LossWeights) -> dict:
    """Input must already be SUM-reduced across all ranks/microbatches."""
    values = stats.detach().cpu().double()
    result: dict = {"heads": {}}
    sensor_objectives: list[float] = []
    union_objective = 0.0
    for h in range(9):
        name = "union" if h == 0 else f"s{h - 1}"
        count = int(values[h, COUNT].item())
        if not count:
            result["heads"][name] = {"count": 0}
            continue
        mean = values[h] / count
        objective = float(weights.sdf * mean[SDF] + weights.grad * mean[GRAD]
                          + weights.eikonal * mean[EIK] + weights.tension * mean[TENSION])
        result["heads"][name] = {
            "count": count, "mae": float(mean[ABS]),
            "rmse": math.sqrt(max(0.0, float(mean[SDF]))),
            "sdf_loss": float(mean[SDF]), "grad_loss": float(mean[GRAD]),
            "grad_cosine": 1.0 - float(mean[GRAD]), "sign_accuracy": float(mean[SIGN]),
            "grad_norm": float(mean[NORM]), "eikonal_loss": float(mean[EIK]),
            "tension_loss": float(mean[TENSION]), "objective": objective,
        }
        if h == 0:
            union_objective = objective
        else:
            sensor_objectives.append(objective)
    count_u = max(1.0, float(values[0, COUNT]))
    consistency = float(values[0, CONS]) / count_u
    sensor_objective = sum(sensor_objectives) / max(1, len(sensor_objectives))
    result.update({
        "union_objective": union_objective, "sensor_objective": sensor_objective,
        "consistency_loss": consistency, "winner_accuracy": float(values[0, WIN]) / count_u,
        "loss": weights.union_objective * union_objective
                + weights.sensor_objective * sensor_objective + weights.consistency * consistency,
    })
    return result

#!/usr/bin/env python3
"""
CAREPlanner Visibility CDF: budget-conditioned oracle visibility ceiling.

Estimate, for matched samples from an existing equal-motion-budget evaluation,

    g_B^*(x, q0) = max g(x, q)
                    s.t. ||q - q0||_2 <= B
                         q_min <= q <= q_max

under the current FOV-only oracle and joint limits.

Why the endpoint ball is equivalent to the path-length budget here
-------------------------------------------------------------------
The equal-motion evaluator constrains the joint-space path length

    sum_k ||q_{k+1} - q_k||_2 <= B.

With only box joint limits (no collision/dynamics constraints), every endpoint
q satisfying ||q-q0||_2 <= B is reachable by the straight segment from q0 to q.
The joint-limit box is convex, so that segment remains within joint limits.
Therefore the endpoint-constrained maximum above is the exact geometric oracle
problem corresponding to the same motion budget.

Numerical search
----------------
The true global optimum is not solved analytically.  This script reports a
numerical LOWER-BOUND estimate g_B_hat using:
  1) random candidates in the feasible L2 ball, including many shell samples;
  2) oracle-gradient seeds from q0;
  3) top-K multi-start finite-difference oracle refinement;
  4) projected monotone backtracking, so accepted refinement never decreases g;
  5) nested-budget seeding, so the estimated ceiling is non-decreasing in B.

The script reuses the already validated standalone FOV/URDF backend in
`evaluate_direct_vs_projection_ascent.py`.

Matching to previous equal-budget results
-----------------------------------------
`evaluate_equal_motion_budget.py` did not save q0 explicitly.  This evaluator
reconstructs the exact q0 samples by replaying the original RNG sequence using
its saved config (seed/x_source/num_trials/batch_q).  It then verifies both:
  * reconstructed x_trial == saved x_trial
  * oracle g(x,q0) == saved init_g
for every selected sample.  If reconstruction does not match, it aborts rather
than silently comparing different samples.

Recommended first run: 2,000-5,000 matched initial-outside samples.  The output
contains a search-quality diagnostic: if learned Direct/Hybrid frequently beat
our g_B_hat by more than the tolerance, increase random candidates / top-K /
refinement iterations before interpreting the normalized progress ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    distribution_stats,
    load_npz_data,
    oracle_visibility_g,
    sample_q,
    sample_x,
    threshold_key,
)


def parse_float_list(text: str) -> List[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return vals


def safe_rate(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask))


def stat(values: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(values)
    return distribution_stats(values, mask)


def budget_tag(value: float) -> str:
    return threshold_key(value).replace("-", "m").replace(".", "p")


def paired_oracle_g(
    x_pair: torch.Tensor,
    q_pair: torch.Tensor,
    oracle: PinocchioFOVOracle,
) -> torch.Tensor:
    """Exact FOV oracle for paired rows (x_i, q_i), without a Bx x Bq cross product."""
    if x_pair.ndim != 2 or x_pair.shape[-1] != 3:
        raise ValueError(f"x_pair must be [N,3], got {tuple(x_pair.shape)}")
    if q_pair.ndim != 2 or q_pair.shape[-1] != len(oracle.joint_names):
        raise ValueError(
            f"q_pair must be [N,{len(oracle.joint_names)}], got {tuple(q_pair.shape)}"
        )
    if x_pair.shape[0] != q_pair.shape[0]:
        raise ValueError("x_pair and q_pair must have the same first dimension.")

    x_pair = x_pair.to(device=q_pair.device, dtype=q_pair.dtype)
    ax = math.tan(math.radians(oracle.horizontal_fov_deg) * 0.5)
    ay = math.tan(math.radians(oracle.vertical_fov_deg) * 0.5)
    nx = math.sqrt(1.0 + ax * ax)
    ny = math.sqrt(1.0 + ay * ay)

    sensor_margins = []
    for chain in oracle._chains(q_pair.device, q_pair.dtype):
        transform = oracle._fk_sensor_batch(chain, q_pair)
        rotation = transform[:, :3, :3]
        translation = transform[:, :3, 3]

        diff_world = x_pair - translation
        point_sensor = torch.einsum("qji,qj->qi", rotation, diff_world)
        xs = point_sensor[:, 0]
        ys = point_sensor[:, 1]
        zs = point_sensor[:, 2]

        planes = torch.stack(
            [
                (xs + zs * ax) / nx,
                (-xs + zs * ax) / nx,
                (ys + zs * ay) / ny,
                (-ys + zs * ay) / ny,
                zs - oracle.z_min,
                oracle.z_max - zs,
            ],
            dim=-1,
        )
        sensor_margins.append(torch.min(planes, dim=-1).values)

    raw = torch.stack(sensor_margins, dim=-1)
    return torch.max(raw - oracle.delta, dim=-1).values


def project_to_feasible_ball(
    q: torch.Tensor,
    q0: torch.Tensor,
    budget: float,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project to joint box, then radially into the q0-centered L2 ball.

    q0 is inside the joint box.  After box clipping, the radial segment from q0
    to q_box stays in the convex box, so the second projection remains feasible.
    """
    q_box = torch.maximum(torch.minimum(q, q_max), q_min)
    delta = q_box - q0
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(budget / torch.clamp(norm, min=eps), max=1.0)
    return q0 + delta * scale


def finite_difference_paired_gradient(
    x_pair: torch.Tensor,
    q_pair: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Joint-limit-aware finite-difference gradient of paired oracle g."""
    grads = []
    for j in range(q_pair.shape[1]):
        qp = q_pair.clone()
        qm = q_pair.clone()
        qp[:, j] += eps
        qm[:, j] -= eps
        qp = torch.maximum(torch.minimum(qp, q_max), q_min)
        qm = torch.maximum(torch.minimum(qm, q_max), q_min)

        gp = paired_oracle_g(x_pair, qp, oracle)
        gm = paired_oracle_g(x_pair, qm, oracle)
        denom = qp[:, j] - qm[:, j]
        gj = torch.where(
            torch.abs(denom) > 1e-12,
            (gp - gm) / denom,
            torch.zeros_like(gp),
        )
        grads.append(gj)
    return torch.stack(grads, dim=-1)


def merge_topk(
    top_g: torch.Tensor,
    top_q: torch.Tensor,
    new_g: torch.Tensor,
    new_q: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge [B,K] current top candidates with [B,C] new candidates."""
    all_g = torch.cat([top_g, new_g], dim=1)
    all_q = torch.cat([top_q, new_q], dim=1)
    kk = min(k, all_g.shape[1])
    best_g, idx = torch.topk(all_g, kk, dim=1, largest=True, sorted=True)
    idx_q = idx.unsqueeze(-1).expand(-1, -1, all_q.shape[-1])
    best_q = torch.gather(all_q, 1, idx_q)
    return best_g, best_q


def random_ball_topk(
    x: torch.Tensor,
    q0: torch.Tensor,
    g0: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    budget: float,
    num_candidates: int,
    candidate_chunk: int,
    shell_fraction: float,
    top_k: int,
    previous_q: Optional[torch.Tensor] = None,
    previous_g: Optional[torch.Tensor] = None,
    use_gradient_seeds: bool = True,
    fd_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random + oracle-gradient seed search inside the feasible ball."""
    bsz, dof = q0.shape
    device = q0.device

    # q0 is always a valid candidate.  Previous-budget best is also valid because
    # budgets are processed in ascending order.
    seed_q = [q0[:, None, :]]
    seed_g = [g0[:, None]]
    if previous_q is not None and previous_g is not None:
        seed_q.append(previous_q[:, None, :])
        seed_g.append(previous_g[:, None])

    # Add several deterministic seeds along the exact oracle gradient at q0.
    if use_gradient_seeds:
        grad0 = finite_difference_paired_gradient(
            x, q0, oracle, q_min, q_max, fd_eps
        )
        gn = torch.linalg.norm(grad0, dim=-1, keepdim=True)
        direction = grad0 / torch.clamp(gn, min=1e-12)
        valid = gn.squeeze(-1) > 1e-10
        for frac in (0.25, 0.5, 1.0):
            cand = q0 + (budget * frac) * direction
            cand = project_to_feasible_ball(cand, q0, budget, q_min, q_max)
            cg = paired_oracle_g(x, cand, oracle)
            # For zero gradient, replace with q0 to avoid arbitrary seed values.
            cand = torch.where(valid[:, None], cand, q0)
            cg = torch.where(valid, cg, g0)
            seed_q.append(cand[:, None, :])
            seed_g.append(cg[:, None])

    top_q = torch.cat(seed_q, dim=1)
    top_g = torch.cat(seed_g, dim=1)
    if top_g.shape[1] > top_k:
        top_g, top_q = merge_topk(
            top_g[:, :0], top_q[:, :0], top_g, top_q, top_k
        )

    remaining = num_candidates
    while remaining > 0:
        c = min(candidate_chunk, remaining)
        direction = torch.randn((bsz, c, dof), device=device, dtype=q0.dtype)
        direction = direction / torch.clamp(
            torch.linalg.norm(direction, dim=-1, keepdim=True), min=1e-12
        )

        # Mix shell samples (r=B) with uniform-volume samples (r=B*u^(1/d)).
        shell = torch.rand((bsz, c), device=device) < shell_fraction
        u = torch.rand((bsz, c), device=device, dtype=q0.dtype)
        r_uniform = budget * torch.pow(u, 1.0 / float(dof))
        radius = torch.where(shell, torch.full_like(r_uniform, budget), r_uniform)

        cand = q0[:, None, :] + direction * radius.unsqueeze(-1)
        q0_rep = q0[:, None, :].expand(-1, c, -1)
        cand_flat = project_to_feasible_ball(
            cand.reshape(-1, dof),
            q0_rep.reshape(-1, dof),
            budget,
            q_min,
            q_max,
        )
        x_flat = x[:, None, :].expand(-1, c, -1).reshape(-1, 3)
        cg = paired_oracle_g(x_flat, cand_flat, oracle).reshape(bsz, c)
        cand = cand_flat.reshape(bsz, c, dof)

        top_g, top_q = merge_topk(top_g, top_q, cg, cand, top_k)
        remaining -= c

    random_best = top_g[:, 0].clone()
    return top_g, top_q, random_best


def refine_topk_ball(
    x: torch.Tensor,
    q0: torch.Tensor,
    top_q: torch.Tensor,
    top_g: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    budget: float,
    fd_eps: float,
    base_step_size: float,
    refine_iters: int,
    line_search_steps: int,
    min_improvement: float,
    stall_patience: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projected monotone multi-start oracle refinement within the budget ball."""
    bsz, k, dof = top_q.shape
    q = top_q.reshape(-1, dof).clone()
    g = top_g.reshape(-1).clone()
    x_rep = x[:, None, :].expand(-1, k, -1).reshape(-1, 3)
    q0_rep = q0[:, None, :].expand(-1, k, -1).reshape(-1, dof)

    active = torch.isfinite(g)
    stall = torch.zeros_like(g, dtype=torch.int32)
    accepted_count = torch.zeros_like(g, dtype=torch.int32)

    for _ in range(refine_iters):
        if not torch.any(active):
            break
        idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        qi = q[idx]
        xi = x_rep[idx]
        q0i = q0_rep[idx]
        gi = g[idx]

        grad = finite_difference_paired_gradient(
            xi, qi, oracle, q_min, q_max, fd_eps
        )
        gn = torch.linalg.norm(grad, dim=-1)
        valid_grad = gn > 1e-10
        direction = torch.zeros_like(grad)
        direction[valid_grad] = grad[valid_grad] / gn[valid_grad, None]

        alpha = torch.full_like(gi, base_step_size)
        accepted = torch.zeros_like(gi, dtype=torch.bool)
        best_cand_q = qi.clone()
        best_cand_g = gi.clone()

        for _ls in range(line_search_steps):
            pending = (~accepted) & valid_grad & (alpha > 1e-8)
            if not torch.any(pending):
                break
            p = torch.nonzero(pending, as_tuple=False).squeeze(-1)
            cand = qi[p] + alpha[p, None] * direction[p]
            cand = project_to_feasible_ball(
                cand, q0i[p], budget, q_min, q_max
            )
            cg = paired_oracle_g(xi[p], cand, oracle)
            improve = cg > (gi[p] + min_improvement)
            if torch.any(improve):
                a = p[improve]
                accepted[a] = True
                best_cand_q[a] = cand[improve]
                best_cand_g[a] = cg[improve]
            alpha[p[~improve]] *= 0.5

        q[idx] = best_cand_q
        g[idx] = best_cand_g
        accepted_count[idx] += accepted.to(torch.int32)
        stall[idx] = torch.where(
            accepted,
            torch.zeros_like(stall[idx]),
            stall[idx] + 1,
        )
        if torch.any(~valid_grad):
            stall[idx[~valid_grad]] = stall_patience
        active[idx] = stall[idx] < stall_patience

    g2 = g.reshape(bsz, k)
    q2 = q.reshape(bsz, k, dof)
    best_g, best_idx = torch.max(g2, dim=1)
    gather_idx = best_idx[:, None, None].expand(-1, 1, dof)
    best_q = torch.gather(q2, 1, gather_idx).squeeze(1)
    accepted = accepted_count.reshape(bsz, k).sum(dim=1)
    return best_g, best_q, accepted


def make_cohort_mask(init_g: np.ndarray, cohort: str, far_margin: float) -> np.ndarray:
    if cohort == "all":
        return np.ones_like(init_g, dtype=bool)
    if cohort == "initial_outside":
        return init_g < 0.0
    if cohort == "initial_far_outside":
        return init_g <= -far_margin
    raise ValueError(cohort)


def load_equal_inputs(args) -> Tuple[Dict, Mapping[str, np.ndarray]]:
    with open(args.equal_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = np.load(args.equal_npz, allow_pickle=False)
    if "config" not in meta:
        raise KeyError("equal JSON does not contain config.")
    for key in ("x_trials", "init_g", "batch_q"):
        if key not in samples.files:
            raise KeyError(f"equal NPZ missing required key: {key}")
    return meta, samples


def reconstruct_selected_q0(
    selected_global_idx: np.ndarray,
    meta: Dict,
    samples: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    device: torch.device,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Replay exact original RNG and extract q0 for selected flattened indices."""
    cfg = meta["config"]
    seed = int(cfg["seed"])
    x_source = str(cfg["x_source"])
    num_trials = int(cfg["num_trials"])
    batch_q = int(cfg["batch_q"])

    selected_global_idx = np.asarray(selected_global_idx, dtype=np.int64)
    order = np.argsort(selected_global_idx)
    idx_sorted = selected_global_idx[order]
    q_out_sorted = np.empty((len(idx_sorted), q_min.numel()), dtype=np.float32)
    x_out_sorted = np.empty((len(idx_sorted), 3), dtype=np.float32)

    x_trials_saved = np.asarray(samples["x_trials"], dtype=np.float32)
    init_g_saved = np.asarray(samples["init_g"], dtype=np.float32)
    if x_trials_saved.shape[0] != num_trials:
        raise ValueError(
            f"x_trials count {x_trials_saved.shape[0]} != config num_trials {num_trials}"
        )
    if init_g_saved.size != num_trials * batch_q:
        raise ValueError("init_g size does not match num_trials*batch_q.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    ptr = 0
    max_x_err = 0.0
    max_g_err = 0.0
    checked = 0

    for trial in range(num_trials):
        x = sample_x(data, device, x_source)
        q0 = sample_q(q_min, q_max, batch_q, device)

        x_np = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        x_err = float(np.max(np.abs(x_np - x_trials_saved[trial])))
        max_x_err = max(max_x_err, x_err)
        if x_err > tolerance:
            raise RuntimeError(
                f"RNG replay mismatch at trial {trial}: x max error {x_err:.3e} > {tolerance:.3e}."
            )

        lo = trial * batch_q
        hi = lo + batch_q
        start = ptr
        while ptr < len(idx_sorted) and idx_sorted[ptr] < hi:
            ptr += 1
        if ptr == start:
            continue

        globals_here = idx_sorted[start:ptr]
        local = globals_here - lo
        q_sel = q0[torch.as_tensor(local, device=device, dtype=torch.long)]
        x_sel = x.expand(len(local), -1)
        g_sel = paired_oracle_g(x_sel, q_sel, oracle).detach().cpu().numpy()
        g_ref = init_g_saved[globals_here]
        g_err = float(np.max(np.abs(g_sel - g_ref)))
        max_g_err = max(max_g_err, g_err)
        if g_err > tolerance:
            raise RuntimeError(
                f"RNG/oracle replay mismatch at trial {trial}: init_g max error "
                f"{g_err:.3e} > {tolerance:.3e}."
            )

        q_out_sorted[start:ptr] = q_sel.detach().cpu().numpy().astype(np.float32)
        x_out_sorted[start:ptr] = x_sel.detach().cpu().numpy().astype(np.float32)
        checked += len(local)

    if ptr != len(idx_sorted) or checked != len(idx_sorted):
        raise RuntimeError("Failed to reconstruct all selected q0 samples.")

    # Undo sorting back to caller selection order.
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    q_out = q_out_sorted[inv]
    x_out = x_out_sorted[inv]
    return q_out, x_out, {
        "max_x_replay_abs_error": max_x_err,
        "max_init_g_replay_abs_error": max_g_err,
        "checked_samples": int(checked),
    }


def learned_vs_oracle_metrics(
    init_g: np.ndarray,
    learned_g: np.ndarray,
    oracle_g: np.ndarray,
    thresholds: Sequence[float],
    search_tolerance: float,
) -> Dict:
    gain_possible = oracle_g - init_g
    gain_learned = learned_g - init_g
    valid = gain_possible > 1e-8
    progress = np.full_like(gain_possible, np.nan, dtype=np.float64)
    progress[valid] = gain_learned[valid] / gain_possible[valid]
    progress_clipped = np.clip(progress, 0.0, 1.0)
    regret = oracle_g - learned_g
    regret_fraction = np.full_like(gain_possible, np.nan, dtype=np.float64)
    regret_fraction[valid] = regret[valid] / gain_possible[valid]

    return {
        "final_g_stats": stat(learned_g),
        "oracle_ceiling_stats": stat(oracle_g),
        "oracle_gap_stats": stat(regret),
        "learned_gain_stats": stat(gain_learned),
        "oracle_possible_gain_stats": stat(gain_possible),
        "progress_ratio_raw_stats": stat(progress),
        "progress_ratio_clipped_0_1_stats": stat(progress_clipped),
        "regret_fraction_stats": stat(regret_fraction),
        "valid_progress_fraction": safe_rate(valid),
        "learned_within_search_tolerance_of_ceiling_rate": safe_rate(
            learned_g >= (oracle_g - search_tolerance)
        ),
        "learned_exceeds_estimated_ceiling_rate": safe_rate(
            learned_g > (oracle_g + search_tolerance)
        ),
        "inside_rates": {
            threshold_key(t): safe_rate(learned_g >= t) for t in thresholds
        },
        "oracle_inside_rates": {
            threshold_key(t): safe_rate(oracle_g >= t) for t in thresholds
        },
    }


def evaluate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    meta, samples = load_equal_inputs(args)
    cfg = meta["config"]
    data_path = args.data or cfg["data"]
    urdf_path = args.urdf or cfg["urdf"]
    data = load_npz_data(data_path)
    q_min = torch.as_tensor(data["q_min"], device=device, dtype=torch.float32)
    q_max = torch.as_tensor(data["q_max"], device=device, dtype=torch.float32)

    oracle = PinocchioFOVOracle(
        urdf_path=urdf_path,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=float(cfg.get("horizontal_fov_deg", 50.0)),
        vertical_fov_deg=float(cfg.get("vertical_fov_deg", 66.0)),
        z_min=float(cfg.get("z_min", 0.20)),
        z_max=float(cfg.get("z_max", 0.70)),
        delta=float(cfg.get("delta", 0.01)),
    )

    init_g_full = np.asarray(samples["init_g"], dtype=np.float32)
    far_margin = float(cfg.get("far_margin", 0.03))
    cohort_mask = make_cohort_mask(init_g_full, args.cohort, far_margin)
    available = np.flatnonzero(cohort_mask)
    if available.size == 0:
        raise RuntimeError(f"No samples in cohort {args.cohort}.")

    n_select = available.size if args.num_samples <= 0 else min(args.num_samples, available.size)
    sel_rng = np.random.default_rng(args.selection_seed)
    selected = np.sort(sel_rng.choice(available, size=n_select, replace=False))

    print("\n=== Budget-conditioned Oracle Ceiling ===")
    print(f"equal JSON:              {args.equal_json}")
    print(f"equal NPZ:               {args.equal_npz}")
    print(f"cohort:                  {args.cohort}")
    print(f"available / selected:    {available.size} / {n_select}")
    print(f"selection seed:          {args.selection_seed}")

    budgets = (
        sorted(float(v) for v in cfg["budgets"])
        if not args.budgets
        else sorted(set(parse_float_list(args.budgets)))
    )
    equal_budgets = {float(v) for v in cfg["budgets"]}
    for b in budgets:
        if not any(abs(b - eb) < 1e-9 for eb in equal_budgets):
            raise ValueError(
                f"Budget {b} is not present in equal-motion results {sorted(equal_budgets)}."
            )

    print(f"budgets:                 {budgets}")
    print(f"random candidates/sample:{args.random_candidates}")
    print(f"top-K / refine iters:    {args.top_k} / {args.refine_iters}")
    print(f"sample batch size:       {args.sample_batch_size}")
    print("==========================================\n")

    # Reconstruct exact q0 before resetting RNG for oracle search.
    q0_np, x_np, replay = reconstruct_selected_q0(
        selected, meta, samples, data, oracle, q_min, q_max,
        device, args.reconstruction_tolerance
    )
    print(
        "RNG replay verified: "
        f"x_err={replay['max_x_replay_abs_error']:.3e}, "
        f"init_g_err={replay['max_init_g_replay_abs_error']:.3e}, "
        f"n={replay['checked_samples']}"
    )

    init_sel = init_g_full[selected].astype(np.float64)

    # Load learned Direct/Hybrid results for exactly the selected samples.
    learned = {b: {} for b in budgets}
    for b in budgets:
        tag = budget_tag(b)
        for method in ("direct", "hybrid"):
            key = f"budget_{tag}_{method}_g"
            if key not in samples.files:
                raise KeyError(f"equal NPZ missing {key}")
            learned[b][method] = np.asarray(samples[key], dtype=np.float32)[selected].astype(np.float64)

    # Search RNG is independent of reconstruction RNG.
    torch.manual_seed(args.search_seed)
    np.random.seed(args.search_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.search_seed)

    n = n_select
    oracle_g_by_budget = {b: np.empty(n, dtype=np.float32) for b in budgets}
    oracle_q_by_budget = {
        b: np.empty((n, q0_np.shape[1]), dtype=np.float32) for b in budgets
    }
    random_best_by_budget = {b: np.empty(n, dtype=np.float32) for b in budgets}
    accepted_by_budget = {b: np.empty(n, dtype=np.int32) for b in budgets}

    t0 = time.time()
    num_batches = math.ceil(n / args.sample_batch_size)
    for bi, start in enumerate(range(0, n, args.sample_batch_size)):
        end = min(start + args.sample_batch_size, n)
        xb = torch.as_tensor(x_np[start:end], device=device, dtype=torch.float32)
        q0b = torch.as_tensor(q0_np[start:end], device=device, dtype=torch.float32)
        g0b = paired_oracle_g(xb, q0b, oracle)

        prev_q = None
        prev_g = None
        for b in budgets:
            top_g, top_q, random_best = random_ball_topk(
                x=xb,
                q0=q0b,
                g0=g0b,
                oracle=oracle,
                q_min=q_min,
                q_max=q_max,
                budget=b,
                num_candidates=args.random_candidates,
                candidate_chunk=args.candidate_chunk,
                shell_fraction=args.shell_fraction,
                top_k=args.top_k,
                previous_q=prev_q,
                previous_g=prev_g,
                use_gradient_seeds=args.use_gradient_seeds,
                fd_eps=args.fd_eps,
            )
            best_g, best_q, accepted = refine_topk_ball(
                x=xb,
                q0=q0b,
                top_q=top_q,
                top_g=top_g,
                oracle=oracle,
                q_min=q_min,
                q_max=q_max,
                budget=b,
                fd_eps=args.fd_eps,
                base_step_size=args.refine_step_size,
                refine_iters=args.refine_iters,
                line_search_steps=args.line_search_steps,
                min_improvement=args.min_improvement,
                stall_patience=args.stall_patience,
            )

            # Nested feasible sets imply g_B is non-decreasing with B. Previous
            # best was included as a seed; this max is an extra numerical guard.
            if prev_g is not None:
                use_prev = prev_g > best_g
                if torch.any(use_prev):
                    best_g = torch.where(use_prev, prev_g, best_g)
                    best_q = torch.where(use_prev[:, None], prev_q, best_q)

            oracle_g_by_budget[b][start:end] = best_g.detach().cpu().numpy()
            oracle_q_by_budget[b][start:end] = best_q.detach().cpu().numpy()
            random_best_by_budget[b][start:end] = random_best.detach().cpu().numpy()
            accepted_by_budget[b][start:end] = accepted.detach().cpu().numpy()
            prev_q, prev_g = best_q.detach(), best_g.detach()

        if (
            bi == 0
            or (bi + 1) % args.print_every == 0
            or bi + 1 == num_batches
        ):
            elapsed = time.time() - t0
            last = budgets[-1]
            print(
                f"batch {bi+1:04d}/{num_batches} samples={end}/{n} "
                f"Bmax median={np.median(oracle_g_by_budget[last][:end]):+.5f} "
                f"elapsed={elapsed:.1f}s"
            )

    thresholds = list(args.margin_thresholds)
    results_b = {}
    print("\n=== Matched oracle-ceiling summary ===")
    print("budget method   learned_g  oracle_g   progress  gap      learned>oracle")

    for b in budgets:
        bg = oracle_g_by_budget[b].astype(np.float64)
        rb = random_best_by_budget[b].astype(np.float64)
        per_method = {}
        for method in ("direct", "hybrid"):
            lg = learned[b][method]
            m = learned_vs_oracle_metrics(
                init_sel, lg, bg, thresholds, args.search_tolerance
            )
            per_method[method] = m
            print(
                f"{b:6.3f} {method:7s} "
                f"{m['final_g_stats']['mean']:+.5f}   "
                f"{m['oracle_ceiling_stats']['mean']:+.5f}   "
                f"{m['progress_ratio_clipped_0_1_stats']['mean']:.4f}   "
                f"{m['oracle_gap_stats']['mean']:+.5f}   "
                f"{m['learned_exceeds_estimated_ceiling_rate']:.4f}"
            )

        results_b[threshold_key(b)] = {
            "budget": b,
            "oracle": {
                "g_B_hat_stats": stat(bg),
                "gain_from_q0_stats": stat(bg - init_sel),
                "random_pre_refine_best_stats": stat(rb),
                "refinement_gain_stats": stat(bg - rb),
                "refinement_improved_rate": safe_rate(bg > rb + args.min_improvement),
                "accepted_refinement_updates_stats": stat(
                    accepted_by_budget[b].astype(np.float64)
                ),
                "inside_rates": {
                    threshold_key(t): safe_rate(bg >= t) for t in thresholds
                },
            },
            "methods": per_method,
        }

    # Search adequacy: learned result should not systematically beat an oracle
    # lower-bound search by much.  A high rate means search settings are too weak.
    worst_exceed = max(
        results_b[threshold_key(b)]["methods"][m]["learned_exceeds_estimated_ceiling_rate"]
        for b in budgets for m in ("direct", "hybrid")
    )
    if worst_exceed > args.max_acceptable_exceed_rate:
        print(
            "\n[WARN] Learned optimizer exceeds estimated oracle ceiling too often: "
            f"max rate={worst_exceed:.4f} > {args.max_acceptable_exceed_rate:.4f}.\n"
            "       Treat g_B_hat as under-searched and rerun with more random candidates,\n"
            "       larger top-K, and/or more refinement iterations."
        )
    else:
        print(
            f"\n[OK] Search-quality diagnostic: max learned>g_B_hat rate "
            f"{worst_exceed:.4f} <= {args.max_acceptable_exceed_rate:.4f}."
        )

    elapsed = time.time() - t0
    result = {
        "config": {
            "equal_json": args.equal_json,
            "equal_npz": args.equal_npz,
            "data": data_path,
            "urdf": urdf_path,
            "device": str(device),
            "cohort": args.cohort,
            "available_cohort_samples": int(available.size),
            "num_selected_samples": int(n),
            "selection_seed": args.selection_seed,
            "search_seed": args.search_seed,
            "budgets": budgets,
            "random_candidates": args.random_candidates,
            "candidate_chunk": args.candidate_chunk,
            "shell_fraction": args.shell_fraction,
            "top_k": args.top_k,
            "use_gradient_seeds": args.use_gradient_seeds,
            "fd_eps": args.fd_eps,
            "refine_step_size": args.refine_step_size,
            "refine_iters": args.refine_iters,
            "line_search_steps": args.line_search_steps,
            "min_improvement": args.min_improvement,
            "stall_patience": args.stall_patience,
            "sample_batch_size": args.sample_batch_size,
            "margin_thresholds": thresholds,
            "search_tolerance": args.search_tolerance,
            "meaning": (
                "numerical lower-bound estimate of max oracle g within the same "
                "q0-centered L2 joint-space motion budget and joint limits"
            ),
        },
        "reconstruction_verification": replay,
        "selected_global_indices": selected.tolist(),
        "initial_g_stats": stat(init_sel),
        "budgets": results_b,
        "search_quality": {
            "max_learned_exceeds_estimated_ceiling_rate": float(worst_exceed),
            "acceptable_rate_threshold": args.max_acceptable_exceed_rate,
        },
        "elapsed_sec": elapsed,
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=True)
        print(f"saved JSON: {args.output}")

    if args.output_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
        save = {
            "selected_global_indices": selected.astype(np.int64),
            "x": x_np.astype(np.float32),
            "q0": q0_np.astype(np.float32),
            "init_g": init_sel.astype(np.float32),
        }
        for b in budgets:
            tag = budget_tag(b)
            save[f"budget_{tag}_oracle_g"] = oracle_g_by_budget[b]
            save[f"budget_{tag}_oracle_q"] = oracle_q_by_budget[b]
            save[f"budget_{tag}_random_best_g"] = random_best_by_budget[b]
            save[f"budget_{tag}_accepted_updates"] = accepted_by_budget[b]
            for method in ("direct", "hybrid"):
                save[f"budget_{tag}_{method}_g"] = learned[b][method].astype(np.float32)
        np.savez_compressed(args.output_npz, **save)
        print(f"saved NPZ:  {args.output_npz}")

    print(f"elapsed_sec: {elapsed:.2f}")
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Budget-conditioned oracle visibility ceiling on matched equal-budget samples."
    )
    p.add_argument(
        "--equal-json",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "equal_motion_budget/equal_motion_budget.json"
        ),
    )
    p.add_argument(
        "--equal-npz",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "equal_motion_budget/equal_motion_budget_samples.npz"
        ),
    )
    p.add_argument("--data", default="", help="Empty = use path stored in equal JSON.")
    p.add_argument("--urdf", default="", help="Empty = use path stored in equal JSON.")
    p.add_argument(
        "--cohort",
        choices=["all", "initial_outside", "initial_far_outside"],
        default="initial_outside",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Matched samples drawn from cohort; <=0 uses all (usually too expensive).",
    )
    p.add_argument("--selection-seed", type=int, default=12345)
    p.add_argument("--search-seed", type=int, default=2026)
    p.add_argument(
        "--budgets",
        default="",
        help="Comma list; empty = all budgets from equal-motion JSON.",
    )
    p.add_argument("--random-candidates", type=int, default=1024)
    p.add_argument("--candidate-chunk", type=int, default=256)
    p.add_argument("--shell-fraction", type=float, default=0.50)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--use-gradient-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--fd-eps", type=float, default=1e-4)
    p.add_argument("--refine-step-size", type=float, default=0.10)
    p.add_argument("--refine-iters", type=int, default=20)
    p.add_argument("--line-search-steps", type=int, default=8)
    p.add_argument("--min-improvement", type=float, default=1e-6)
    p.add_argument("--stall-patience", type=int, default=3)
    p.add_argument("--sample-batch-size", type=int, default=32)
    p.add_argument(
        "--margin-thresholds",
        type=parse_float_list,
        default=parse_float_list("0,0.005,0.01,0.03"),
    )
    p.add_argument(
        "--reconstruction-tolerance",
        type=float,
        default=5e-5,
        help="Abort if replayed x/init_g differ from saved equal-budget samples by more.",
    )
    p.add_argument(
        "--search-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance used when diagnosing learned final g > estimated oracle ceiling.",
    )
    p.add_argument(
        "--max-acceptable-exceed-rate",
        type=float,
        default=0.02,
        help="Warn if learned exceeds g_B_hat+tol on more than this fraction.",
    )
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--print-every", type=int, default=5, help="Print every N sample batches.")
    p.add_argument(
        "--output",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "budget_conditioned_oracle_ceiling/budget_conditioned_oracle_ceiling.json"
        ),
    )
    p.add_argument(
        "--output-npz",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "budget_conditioned_oracle_ceiling/budget_conditioned_oracle_ceiling_samples.npz"
        ),
    )
    args = p.parse_args()

    if args.num_samples == 0:
        p.error("--num-samples 0 is ambiguous; use a positive number or -1 for all.")
    if args.random_candidates < 0 or args.candidate_chunk <= 0:
        p.error("invalid random-candidate settings.")
    if not (0.0 <= args.shell_fraction <= 1.0):
        p.error("--shell-fraction must be in [0,1].")
    if args.top_k <= 0 or args.refine_iters < 0:
        p.error("invalid top-K/refinement settings.")
    if args.fd_eps <= 0 or args.refine_step_size <= 0:
        p.error("fd-eps and refine-step-size must be positive.")
    if args.line_search_steps <= 0 or args.stall_patience <= 0:
        p.error("line-search-steps and stall-patience must be positive.")
    if args.sample_batch_size <= 0 or args.print_every <= 0:
        p.error("sample-batch-size and print-every must be positive.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

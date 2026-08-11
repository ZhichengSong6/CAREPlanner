#!/usr/bin/env python3
"""
CAREPlanner Visibility CDF: equal-motion-budget evaluation.

Compare two learned-field optimizers on exactly the same sampled (x, q_init):

  Direct:
      q_init -> normalized learned-gradient ascent

  Hybrid:
      q_init -> learned zero-level projection -> normalized learned-gradient ascent

The comparison is controlled by an actual joint-space path-length budget B:

      sum_k ||q_{k+1} - q_k||_2 <= B

The budget is enforced per sample AFTER joint-limit clamping.  A final partial
step is shortened so that neither method can exceed the requested budget.

The hybrid method uses the projection stage first and then spends any remaining
budget on local ascent.  By default, projection for each sample stops early once
|f| < epsilon_f; pass --no-projection-stop-on-boundary to reproduce a fixed
number of projection iterations before ascent.

This evaluator intentionally reuses the already validated standalone backend in
`evaluate_direct_vs_projection_ascent.py`, so model reconstruction, sampling,
URDF FK, and the analytic FOV oracle are exactly the same as in the previous
Exp1 evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    distribution_stats,
    final_limit_contact,
    learned_ascent_step,
    learned_projection_step,
    load_npz_data,
    model_value,
    model_value_and_grad_q,
    oracle_visibility_g,
    parse_float_list,
    safe_rate,
    sample_q,
    sample_x,
    threshold_key,
    torch_load_checkpoint,
)


def parse_budget_list(text: str) -> List[float]:
    values = sorted(set(float(v.strip()) for v in text.split(",") if v.strip()))
    if not values or values[0] <= 0.0:
        raise argparse.ArgumentTypeError("Budgets must be positive floats.")
    return values


def make_cohorts(init_g: np.ndarray, far_margin: float) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones_like(init_g, dtype=bool),
        "initial_outside": init_g < 0.0,
        "initial_far_outside": init_g <= -far_margin,
    }


def apply_proposal_with_budget(
    q: torch.Tensor,
    proposed_q: torch.Tensor,
    remaining_budget: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    clamp_q: bool,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply a proposed update while respecting joint limits and path budget.

    The joint-limit projection is applied first.  The resulting segment from q
    to q_clamped is then shortened, if necessary, to exactly fit the remaining
    per-sample path budget.  Since the joint box is convex, shortening that
    segment cannot violate the joint limits.
    """
    if clamp_q:
        q_limited = torch.maximum(
            torch.minimum(proposed_q, q_max[None, :]), q_min[None, :]
        )
        joint_clipped = torch.any(torch.abs(q_limited - proposed_q) > 1e-10, dim=-1)
    else:
        q_limited = proposed_q
        joint_clipped = torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)

    delta = q_limited - q
    full_norm = torch.linalg.norm(delta, dim=-1)
    remaining_budget = torch.clamp(remaining_budget, min=0.0)

    scale = torch.ones_like(full_norm)
    nonzero = full_norm > eps
    scale[nonzero] = torch.minimum(
        torch.ones_like(full_norm[nonzero]),
        remaining_budget[nonzero] / full_norm[nonzero],
    )
    scale = torch.clamp(scale, min=0.0, max=1.0)

    actual_delta = delta * scale.unsqueeze(-1)
    q_new = q + actual_delta
    actual_norm = torch.linalg.norm(actual_delta, dim=-1)
    budget_truncated = full_norm > (remaining_budget + 1e-9)

    return q_new, {
        "actual_step_norm": actual_norm,
        "full_post_limit_step_norm": full_norm,
        "joint_clipped": joint_clipped,
        "budget_truncated": budget_truncated,
    }


def _active_subset_value_grad(x, q, model, active):
    idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return idx, None, None
    f, grad, _ = model_value_and_grad_q(x, q[idx], model)
    return idx, f.detach(), grad.detach()


def run_budgeted_direct(
    x: torch.Tensor,
    q_init: torch.Tensor,
    model: torch.nn.Module,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    budget: float,
    step_size: float,
    max_step_norm: float,
    max_iters: int,
    clamp_q: bool,
    stall_eps: float,
) -> Dict[str, torch.Tensor]:
    q = q_init.clone()
    n = q.shape[0]
    path = torch.zeros(n, device=q.device)
    max_actual_step = torch.zeros(n, device=q.device)
    clamp_any = torch.zeros(n, device=q.device, dtype=torch.bool)
    budget_truncated_any = torch.zeros(n, device=q.device, dtype=torch.bool)
    stalled = torch.zeros(n, device=q.device, dtype=torch.bool)
    updates = torch.zeros(n, device=q.device, dtype=torch.int32)

    for _ in range(max_iters):
        remaining = budget - path
        active = (remaining > 1e-8) & (~stalled)
        if not torch.any(active):
            break

        idx, _f, grad = _active_subset_value_grad(x, q, model, active)
        if idx.numel() == 0:
            break

        proposed, _ = learned_ascent_step(
            q[idx], grad, step_size=step_size, max_step_norm=max_step_norm
        )
        q_new, diag = apply_proposal_with_budget(
            q[idx], proposed, remaining[idx], q_min, q_max, clamp_q
        )

        actual = diag["actual_step_norm"]
        q[idx] = q_new
        path[idx] += actual
        max_actual_step[idx] = torch.maximum(max_actual_step[idx], actual)
        clamp_any[idx] |= diag["joint_clipped"]
        budget_truncated_any[idx] |= diag["budget_truncated"]
        updates[idx] += (actual > stall_eps).to(torch.int32)
        stalled[idx] |= actual <= stall_eps

    return {
        "q": q,
        "path": path,
        "max_actual_step": max_actual_step,
        "clamp_any": clamp_any,
        "budget_truncated_any": budget_truncated_any,
        "stalled": stalled,
        "updates": updates,
    }


def run_budgeted_hybrid(
    x: torch.Tensor,
    q_init: torch.Tensor,
    model: torch.nn.Module,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    budget: float,
    projection_iters: int,
    projection_damping: float,
    projection_max_step_norm: float,
    epsilon_f: float,
    projection_stop_on_boundary: bool,
    ascent_step_size: float,
    ascent_max_step_norm: float,
    max_ascent_iters: int,
    clamp_q: bool,
    stall_eps: float,
) -> Dict[str, torch.Tensor]:
    q = q_init.clone()
    n = q.shape[0]
    path = torch.zeros(n, device=q.device)
    projection_path = torch.zeros(n, device=q.device)
    max_actual_step = torch.zeros(n, device=q.device)
    clamp_any = torch.zeros(n, device=q.device, dtype=torch.bool)
    budget_truncated_any = torch.zeros(n, device=q.device, dtype=torch.bool)
    stalled_projection = torch.zeros(n, device=q.device, dtype=torch.bool)
    projection_done = torch.zeros(n, device=q.device, dtype=torch.bool)
    projection_updates = torch.zeros(n, device=q.device, dtype=torch.int32)

    # Stage 1: learned zero-level projection, with per-sample early stopping.
    for _ in range(projection_iters):
        remaining = budget - path
        eligible = (remaining > 1e-8) & (~stalled_projection) & (~projection_done)
        if not torch.any(eligible):
            break

        idx, f, grad = _active_subset_value_grad(x, q, model, eligible)
        if idx.numel() == 0:
            break

        if projection_stop_on_boundary:
            reached = torch.abs(f) < epsilon_f
            if torch.any(reached):
                projection_done[idx[reached]] = True
            move_mask = ~reached
        else:
            move_mask = torch.ones_like(f, dtype=torch.bool)

        if not torch.any(move_mask):
            continue

        move_idx = idx[move_mask]
        f_move = f[move_mask]
        grad_move = grad[move_mask]

        proposed, _ = learned_projection_step(
            q[move_idx],
            f_move,
            grad_move,
            damping=projection_damping,
            max_step_norm=projection_max_step_norm,
        )
        q_new, diag = apply_proposal_with_budget(
            q[move_idx],
            proposed,
            remaining[move_idx],
            q_min,
            q_max,
            clamp_q,
        )

        actual = diag["actual_step_norm"]
        q[move_idx] = q_new
        path[move_idx] += actual
        projection_path[move_idx] += actual
        max_actual_step[move_idx] = torch.maximum(max_actual_step[move_idx], actual)
        clamp_any[move_idx] |= diag["joint_clipped"]
        budget_truncated_any[move_idx] |= diag["budget_truncated"]
        projection_updates[move_idx] += (actual > stall_eps).to(torch.int32)
        stalled_projection[move_idx] |= actual <= stall_eps

    q_after_projection = q.clone()
    proj_f = model_value(x, q_after_projection, model)

    # Stage 2: spend all remaining budget on learned normalized ascent.
    stalled_ascent = torch.zeros(n, device=q.device, dtype=torch.bool)
    ascent_updates = torch.zeros(n, device=q.device, dtype=torch.int32)

    for _ in range(max_ascent_iters):
        remaining = budget - path
        active = (remaining > 1e-8) & (~stalled_ascent)
        if not torch.any(active):
            break

        idx, _f, grad = _active_subset_value_grad(x, q, model, active)
        if idx.numel() == 0:
            break

        proposed, _ = learned_ascent_step(
            q[idx], grad, step_size=ascent_step_size,
            max_step_norm=ascent_max_step_norm,
        )
        q_new, diag = apply_proposal_with_budget(
            q[idx], proposed, remaining[idx], q_min, q_max, clamp_q
        )

        actual = diag["actual_step_norm"]
        q[idx] = q_new
        path[idx] += actual
        max_actual_step[idx] = torch.maximum(max_actual_step[idx], actual)
        clamp_any[idx] |= diag["joint_clipped"]
        budget_truncated_any[idx] |= diag["budget_truncated"]
        ascent_updates[idx] += (actual > stall_eps).to(torch.int32)
        stalled_ascent[idx] |= actual <= stall_eps

    return {
        "q": q,
        "q_after_projection": q_after_projection,
        "proj_f": proj_f,
        "path": path,
        "projection_path": projection_path,
        "local_ascent_path": path - projection_path,
        "max_actual_step": max_actual_step,
        "clamp_any": clamp_any,
        "budget_truncated_any": budget_truncated_any,
        "stalled_projection": stalled_projection,
        "stalled_ascent": stalled_ascent,
        "projection_updates": projection_updates,
        "ascent_updates": ascent_updates,
    }


def method_metrics(
    cohort: np.ndarray,
    init_g: np.ndarray,
    final_g: np.ndarray,
    endpoint_distance: np.ndarray,
    path: np.ndarray,
    max_step: np.ndarray,
    clamp_any: np.ndarray,
    limit_contact: np.ndarray,
    updates: np.ndarray,
    budget: float,
    margin_thresholds: Sequence[float],
) -> Dict:
    delta_g = final_g - init_g
    efficiency = np.divide(
        delta_g,
        path,
        out=np.full_like(delta_g, np.nan, dtype=np.float64),
        where=path > 1e-8,
    )
    result = {
        "count": int(cohort.sum()),
        "final_g_stats": distribution_stats(final_g, cohort),
        "delta_g_stats": distribution_stats(delta_g, cohort),
        "endpoint_distance_stats": distribution_stats(endpoint_distance, cohort),
        "path_length_stats": distribution_stats(path, cohort),
        "budget_utilization_stats": distribution_stats(path / budget, cohort),
        "budget_exhausted_rate": safe_rate(path >= (budget - 1e-6), cohort),
        "max_actual_step_stats": distribution_stats(max_step, cohort),
        "oracle_gain_per_path_stats": distribution_stats(efficiency, cohort),
        "clamp_any_rate": safe_rate(clamp_any, cohort),
        "final_joint_limit_contact_rate": safe_rate(limit_contact, cohort),
        "update_count_stats": distribution_stats(updates.astype(np.float64), cohort),
    }
    result["oracle_inside_rate"] = {
        threshold_key(t): safe_rate(final_g >= t, cohort)
        for t in margin_thresholds
    }
    return result


def hybrid_projection_metrics(
    cohort: np.ndarray,
    proj_f: np.ndarray,
    proj_g: np.ndarray,
    projection_path: np.ndarray,
    local_path: np.ndarray,
    projection_updates: np.ndarray,
    ascent_updates: np.ndarray,
    budget: float,
    epsilon_f: float,
    epsilon_g: float,
) -> Dict:
    return {
        "count": int(cohort.sum()),
        "learned_boundary_rate": safe_rate(np.abs(proj_f) < epsilon_f, cohort),
        "oracle_boundary_rate": safe_rate(np.abs(proj_g) < epsilon_g, cohort),
        "projection_path_stats": distribution_stats(projection_path, cohort),
        "local_ascent_path_stats": distribution_stats(local_path, cohort),
        "projection_consumed_full_budget_rate": safe_rate(
            projection_path >= (budget - 1e-6), cohort
        ),
        "projection_update_count_stats": distribution_stats(
            projection_updates.astype(np.float64), cohort
        ),
        "ascent_update_count_stats": distribution_stats(
            ascent_updates.astype(np.float64), cohort
        ),
    }


def comparison_metrics(
    cohort: np.ndarray,
    direct_g: np.ndarray,
    hybrid_g: np.ndarray,
    direct_path: np.ndarray,
    hybrid_path: np.ndarray,
    margin_thresholds: Sequence[float],
) -> Dict:
    result = {
        "count": int(cohort.sum()),
        "hybrid_minus_direct_g_stats": distribution_stats(hybrid_g - direct_g, cohort),
        "hybrid_minus_direct_path_stats": distribution_stats(
            hybrid_path - direct_path, cohort
        ),
        "by_threshold": {},
    }
    for threshold in margin_thresholds:
        key = threshold_key(threshold)
        d = direct_g >= threshold
        h = hybrid_g >= threshold
        result["by_threshold"][key] = {
            "direct_success_rate": safe_rate(d, cohort),
            "hybrid_success_rate": safe_rate(h, cohort),
            "both_success_rate": safe_rate(d & h, cohort),
            "hybrid_only_success_rate": safe_rate(h & (~d), cohort),
            "direct_only_success_rate": safe_rate(d & (~h), cohort),
            "neither_success_rate": safe_rate((~d) & (~h), cohort),
        }
    return result


def evaluate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data = load_npz_data(args.data)
    q_min = torch.as_tensor(data["q_min"], device=device, dtype=torch.float32)
    q_max = torch.as_tensor(data["q_max"], device=device, dtype=torch.float32)

    ckpt = torch_load_checkpoint(args.checkpoint, device)
    model, _ = build_model_from_checkpoint(ckpt, device)

    oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        z_min=args.z_min,
        z_max=args.z_max,
        delta=args.delta,
    )

    budgets = list(args.budgets)
    thresholds = list(args.margin_thresholds)
    max_ascent_iters = args.max_ascent_iters
    if max_ascent_iters <= 0:
        max_ascent_iters = int(math.ceil(max(budgets) / args.ascent_step_size)) + 20

    print("\n=== Equal Motion Budget Evaluation ===")
    print(f"checkpoint:                 {args.checkpoint}")
    print(f"x_source:                   {args.x_source}")
    print(f"num_trials / batch_q:       {args.num_trials} / {args.batch_q}")
    print(f"budgets:                    {budgets}")
    print(f"ascent_step_size:           {args.ascent_step_size}")
    print(f"projection_iters:           {args.projection_iters}")
    print(f"projection_stop_boundary:   {args.projection_stop_on_boundary}")
    print(f"epsilon_f / epsilon_g:      {args.epsilon_f} / {args.epsilon_g}")
    print(f"max_ascent_iters:           {max_ascent_iters}")
    print(f"clamp_q:                    {args.clamp_q}")
    print("======================================\n")

    store = {
        budget: {
            "direct": {k: [] for k in [
                "g", "endpoint", "path", "max_step", "clamp_any", "limit_contact", "updates"
            ]},
            "hybrid": {k: [] for k in [
                "g", "endpoint", "path", "max_step", "clamp_any", "limit_contact",
                "updates", "proj_f", "proj_g", "proj_path", "local_path",
                "proj_updates", "ascent_updates"
            ]},
        }
        for budget in budgets
    }
    init_g_all = []
    x_trials = []

    t0 = time.time()
    for trial in range(args.num_trials):
        x = sample_x(data, device, args.x_source)
        q0 = sample_q(q_min, q_max, args.batch_q, device)
        g0 = oracle_visibility_g(x, q0, oracle)
        init_g_all.append(g0.detach().cpu().numpy())
        x_trials.append(x.squeeze(0).detach().cpu().numpy())

        for budget in budgets:
            direct = run_budgeted_direct(
                x=x,
                q_init=q0,
                model=model,
                q_min=q_min,
                q_max=q_max,
                budget=budget,
                step_size=args.ascent_step_size,
                max_step_norm=args.ascent_max_step_norm,
                max_iters=max_ascent_iters,
                clamp_q=args.clamp_q,
                stall_eps=args.stall_eps,
            )
            qd = direct["q"]
            gd = oracle_visibility_g(x, qd, oracle)
            d_endpoint = torch.linalg.norm(qd - q0, dim=-1)
            d_limit = final_limit_contact(
                qd, q_min, q_max, args.joint_limit_tolerance
            )

            hybrid = run_budgeted_hybrid(
                x=x,
                q_init=q0,
                model=model,
                q_min=q_min,
                q_max=q_max,
                budget=budget,
                projection_iters=args.projection_iters,
                projection_damping=args.projection_damping,
                projection_max_step_norm=args.projection_max_step_norm,
                epsilon_f=args.epsilon_f,
                projection_stop_on_boundary=args.projection_stop_on_boundary,
                ascent_step_size=args.ascent_step_size,
                ascent_max_step_norm=args.ascent_max_step_norm,
                max_ascent_iters=max_ascent_iters,
                clamp_q=args.clamp_q,
                stall_eps=args.stall_eps,
            )
            qh = hybrid["q"]
            gh = oracle_visibility_g(x, qh, oracle)
            gproj = oracle_visibility_g(x, hybrid["q_after_projection"], oracle)
            h_endpoint = torch.linalg.norm(qh - q0, dim=-1)
            h_limit = final_limit_contact(
                qh, q_min, q_max, args.joint_limit_tolerance
            )

            def add(method, key, tensor):
                store[budget][method][key].append(tensor.detach().cpu().numpy())

            add("direct", "g", gd)
            add("direct", "endpoint", d_endpoint)
            add("direct", "path", direct["path"])
            add("direct", "max_step", direct["max_actual_step"])
            add("direct", "clamp_any", direct["clamp_any"])
            add("direct", "limit_contact", d_limit)
            add("direct", "updates", direct["updates"])

            add("hybrid", "g", gh)
            add("hybrid", "endpoint", h_endpoint)
            add("hybrid", "path", hybrid["path"])
            add("hybrid", "max_step", hybrid["max_actual_step"])
            add("hybrid", "clamp_any", hybrid["clamp_any"])
            add("hybrid", "limit_contact", h_limit)
            add("hybrid", "updates", hybrid["projection_updates"] + hybrid["ascent_updates"])
            add("hybrid", "proj_f", hybrid["proj_f"])
            add("hybrid", "proj_g", gproj)
            add("hybrid", "proj_path", hybrid["projection_path"])
            add("hybrid", "local_path", hybrid["local_ascent_path"])
            add("hybrid", "proj_updates", hybrid["projection_updates"])
            add("hybrid", "ascent_updates", hybrid["ascent_updates"])

        if (trial + 1) % args.print_every == 0 or trial == 0 or trial + 1 == args.num_trials:
            elapsed = time.time() - t0
            print(f"trial {trial+1:04d}/{args.num_trials} elapsed={elapsed:.1f}s")

    init_g = np.concatenate(init_g_all, axis=0)
    x_trials_np = np.asarray(x_trials, dtype=np.float32)
    cohorts = make_cohorts(init_g, args.far_margin)

    for budget in budgets:
        for method in ["direct", "hybrid"]:
            for key in store[budget][method]:
                store[budget][method][key] = np.concatenate(
                    store[budget][method][key], axis=0
                )

    result_budgets = {}
    for budget in budgets:
        key_b = threshold_key(budget)
        d = store[budget]["direct"]
        h = store[budget]["hybrid"]
        result_budgets[key_b] = {
            "budget": budget,
            "methods": {"direct": {}, "hybrid": {}},
            "comparison": {},
            "hybrid_projection_stage": {},
        }
        for cohort_name, mask in cohorts.items():
            result_budgets[key_b]["methods"]["direct"][cohort_name] = method_metrics(
                mask, init_g, d["g"], d["endpoint"], d["path"], d["max_step"],
                d["clamp_any"], d["limit_contact"], d["updates"], budget, thresholds
            )
            result_budgets[key_b]["methods"]["hybrid"][cohort_name] = method_metrics(
                mask, init_g, h["g"], h["endpoint"], h["path"], h["max_step"],
                h["clamp_any"], h["limit_contact"], h["updates"], budget, thresholds
            )
            result_budgets[key_b]["comparison"][cohort_name] = comparison_metrics(
                mask, d["g"], h["g"], d["path"], h["path"], thresholds
            )
            result_budgets[key_b]["hybrid_projection_stage"][cohort_name] = hybrid_projection_metrics(
                mask, h["proj_f"], h["proj_g"], h["proj_path"], h["local_path"],
                h["proj_updates"], h["ascent_updates"], budget,
                args.epsilon_f, args.epsilon_g,
            )

    elapsed = time.time() - t0
    result = {
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_step": ckpt.get("step", None),
            "checkpoint_best_val": ckpt.get("best_val", None),
            "data": args.data,
            "urdf": args.urdf,
            "device": str(device),
            "seed": args.seed,
            "x_source": args.x_source,
            "num_trials": args.num_trials,
            "batch_q": args.batch_q,
            "total_samples": int(init_g.size),
            "budgets": budgets,
            "budget_definition": "sum L2 joint-space step lengths after joint-limit clamp",
            "ascent_step_size": args.ascent_step_size,
            "ascent_max_step_norm": args.ascent_max_step_norm,
            "max_ascent_iters": max_ascent_iters,
            "projection_iters": args.projection_iters,
            "projection_damping": args.projection_damping,
            "projection_max_step_norm": args.projection_max_step_norm,
            "projection_stop_on_boundary": args.projection_stop_on_boundary,
            "epsilon_f": args.epsilon_f,
            "epsilon_g": args.epsilon_g,
            "margin_thresholds": thresholds,
            "far_margin": args.far_margin,
            "clamp_q": args.clamp_q,
            "joint_limit_tolerance": args.joint_limit_tolerance,
            "horizontal_fov_deg": args.horizontal_fov_deg,
            "vertical_fov_deg": args.vertical_fov_deg,
            "z_min": args.z_min,
            "z_max": args.z_max,
            "delta": args.delta,
        },
        "x_trials": x_trials_np.tolist(),
        "initial": {
            name: {
                "count": int(mask.sum()),
                "g_stats": distribution_stats(init_g, mask),
                "oracle_inside_rate": safe_rate(init_g >= 0.0, mask),
            }
            for name, mask in cohorts.items()
        },
        "budgets": result_budgets,
        "elapsed_sec": elapsed,
    }

    print("\n=== Equal-budget summary: initial_outside ===")
    header = "budget  method    path    g>=0   g>=.005 g>=.01  g>=.03  clamp"
    print(header)
    for budget in budgets:
        kb = threshold_key(budget)
        for method in ["direct", "hybrid"]:
            r = result_budgets[kb]["methods"][method]["initial_outside"]
            rates = r["oracle_inside_rate"]
            print(
                f"{budget:6.3f}  {method:7s} "
                f"{r['path_length_stats']['mean']:6.3f} "
                f"{rates[threshold_key(0.0)]:7.4f} "
                f"{rates[threshold_key(0.005)]:8.4f} "
                f"{rates[threshold_key(0.01)]:7.4f} "
                f"{rates[threshold_key(0.03)]:7.4f} "
                f"{r['clamp_any_rate']:7.4f}"
            )

    print("\n=== Hybrid-only vs direct-only success at g>=0 ===")
    for budget in budgets:
        kb = threshold_key(budget)
        c = result_budgets[kb]["comparison"]["initial_outside"]["by_threshold"][threshold_key(0.0)]
        print(
            f"B={budget:.3f}: hybrid_only={c['hybrid_only_success_rate']:.4f}, "
            f"direct_only={c['direct_only_success_rate']:.4f}, "
            f"both={c['both_success_rate']:.4f}"
        )

    print(f"elapsed_sec: {elapsed:.2f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=True)
        print(f"saved JSON: {args.output}")

    if args.save_samples:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_samples)), exist_ok=True)
        save = {
            "x_trials": x_trials_np,
            "init_g": init_g,
            "batch_q": np.asarray([args.batch_q], dtype=np.int32),
        }
        for budget in budgets:
            tag = threshold_key(budget).replace(".", "p")
            for method in ["direct", "hybrid"]:
                values = store[budget][method]
                for name in ["g", "endpoint", "path", "max_step", "clamp_any", "limit_contact", "updates"]:
                    save[f"budget_{tag}_{method}_{name}"] = values[name]
            for name in ["proj_f", "proj_g", "proj_path", "local_path", "proj_updates", "ascent_updates"]:
                save[f"budget_{tag}_hybrid_{name}"] = store[budget]["hybrid"][name]
        np.savez_compressed(args.save_samples, **save)
        print(f"saved samples: {args.save_samples}")

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Equal joint-space motion-budget direct-vs-hybrid evaluator.")
    p.add_argument("--checkpoint", default="src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt")
    p.add_argument("--data", default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz")
    p.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    p.add_argument("--num-trials", type=int, default=1000)
    p.add_argument("--batch-q", type=int, default=1000)
    p.add_argument("--x-source", choices=["unit_box", "random_box", "dataset"], default="unit_box")
    p.add_argument("--budgets", type=parse_budget_list, default=parse_budget_list("0.25,0.5,0.75,1.0"))
    p.add_argument("--ascent-step-size", type=float, default=0.05)
    p.add_argument("--ascent-max-step-norm", type=float, default=0.25)
    p.add_argument("--max-ascent-iters", type=int, default=0, help="0 = auto from maximum budget and ascent step size.")
    p.add_argument("--projection-iters", type=int, default=10)
    p.add_argument("--projection-damping", type=float, default=0.5)
    p.add_argument("--projection-max-step-norm", type=float, default=0.25)
    p.add_argument("--projection-stop-on-boundary", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epsilon-f", type=float, default=0.03)
    p.add_argument("--epsilon-g", type=float, default=0.03)
    p.add_argument("--margin-thresholds", type=parse_float_list, default=parse_float_list("0,0.005,0.01,0.03"))
    p.add_argument("--far-margin", type=float, default=0.03)
    p.add_argument("--clamp-q", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--joint-limit-tolerance", type=float, default=1e-5)
    p.add_argument("--stall-eps", type=float, default=1e-9)
    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--output", default="src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/equal_motion_budget/equal_motion_budget.json")
    p.add_argument("--save-samples", default="", help="Optional compressed NPZ for later oracle-ceiling normalization.")
    args = p.parse_args()

    if args.num_trials <= 0 or args.batch_q <= 0:
        p.error("num-trials and batch-q must be positive.")
    if args.ascent_step_size <= 0 or args.ascent_max_step_norm < 0:
        p.error("invalid ascent step size/max step norm.")
    if args.projection_iters < 0 or args.projection_damping <= 0:
        p.error("invalid projection settings.")
    if args.epsilon_f <= 0 or args.epsilon_g <= 0:
        p.error("epsilon-f and epsilon-g must be positive.")
    if args.joint_limit_tolerance < 0 or args.stall_eps < 0:
        p.error("joint-limit-tolerance and stall-eps must be non-negative.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

#!/usr/bin/env python3
"""Targeted Case026 per-sensor fallback test.

This is the narrow test requested after the first hybrid integration run:
replay the *original* Case026 visibility target and measured seed that produced
the known self-occluded S4 scalar q_vis, then run the current hybrid policy:

  measured seed
      -> scalar VisCDF projection/root -> q_zero
      -> 8-head branch ranking
      -> branch-specific ascent
      -> conservative analytic FOV
      -> zero-padding primitive self-occlusion
      -> reject/blacklist and try next branch

No planner, Gazebo, GCDF, VBC trajectory certification, or confidence-map
dynamics are involved.  The sole question is whether the new representation
can escape the original self-occluded visibility mode.

The script also checks the previously observed blocked q_vis directly as a
sanity check on the primitive LOS implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
VIS_SCRIPTS = REPO / "src" / "care_visibility_cdf" / "scripts"
if str(VIS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(VIS_SCRIPTS))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    build_model_from_checkpoint,
    learned_projection_step,
    model_value_and_grad_q,
    torch_load_checkpoint,
)
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime  # noqa: E402


TARGET = np.asarray(
    [0.10000000149011612, 0.05000000074505806, 0.15000000596046448],
    dtype=np.float64,
)
MEASURED_SEED = np.asarray(
    [
        -0.23993935822373746,
        0.7577102185084499,
        -0.301829051647589,
        -1.626700836259844,
        0.19068393720573518,
        -0.18052088294007795,
        -0.3808012222262862,
    ],
    dtype=np.float64,
)
KNOWN_BLOCKED_QVIS = np.asarray(
    [
        -0.26947852969169617,
        0.750813901424408,
        -0.2667731046676636,
        -1.8708069324493408,
        0.1902204304933548,
        -0.1728929728269577,
        -0.3792421519756317,
    ],
    dtype=np.float64,
)

Q_MIN = np.asarray(
    [-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20],
    dtype=np.float64,
)
Q_MAX = np.asarray(
    [3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20],
    dtype=np.float64,
)


def fmt(v, nd=5):
    return "[" + ", ".join(f"{float(x):+.{nd}f}" for x in v) + "]"


@torch.no_grad()
def scalar_value(model, x, q):
    return float(model(torch.cat([x, q], dim=-1)).reshape(-1)[0].item())


def scalar_root_bisection(
    model,
    x,
    qa,
    fa,
    qb,
    fb,
    iters,
    tolerance,
):
    lo = qa.detach().clone()
    hi = qb.detach().clone()
    flo = float(fa)
    fhi = float(fb)
    hist = []

    best_q = lo if abs(flo) <= abs(fhi) else hi
    best_f = flo if abs(flo) <= abs(fhi) else fhi

    for k in range(1, iters + 1):
        mid = 0.5 * (lo + hi)
        fm = scalar_value(model, x, mid)
        hist.append(
            {
                "iter": k,
                "f": float(fm),
                "q": mid[0].detach().cpu().numpy().astype(float).tolist(),
            }
        )
        if abs(fm) < abs(best_f):
            best_q = mid.detach().clone()
            best_f = float(fm)
        if abs(fm) <= tolerance:
            return mid.detach(), float(fm), hist

        if flo * fm <= 0.0:
            hi = mid.detach()
            fhi = float(fm)
        else:
            lo = mid.detach()
            flo = float(fm)

    return best_q.detach(), float(best_f), hist


def scalar_project_to_zero(
    model,
    x,
    q_seed,
    q_min,
    q_max,
    projection_iters,
    damping,
    epsilon_f,
    max_step_norm,
    root_refine_iters,
    root_tolerance,
):
    q = q_seed.detach().clone()
    f_current = scalar_value(model, x, q)
    best_q = q.clone()
    best_f = float(f_current)
    history = [
        {
            "iter": 0,
            "f": float(f_current),
            "q": q[0].detach().cpu().numpy().astype(float).tolist(),
        }
    ]

    if f_current >= 0.0:
        return {
            "q_zero": q,
            "f_zero": float(f_current),
            "root_source": "initial_positive",
            "projection_history": history,
            "root_history": [],
        }
    if abs(f_current) <= epsilon_f:
        return {
            "q_zero": q,
            "f_zero": float(f_current),
            "root_source": "initial_tolerance",
            "projection_history": history,
            "root_history": [],
        }

    for k in range(1, projection_iters + 1):
        f_tensor, grad_tensor, _ = model_value_and_grad_q(x, q, model)
        q_next, diag = learned_projection_step(
            q=q,
            f=f_tensor,
            grad=grad_tensor,
            damping=damping,
            max_step_norm=max_step_norm,
            eps=1e-8,
        )
        q_next = torch.maximum(
            torch.minimum(q_next, q_max[None, :]), q_min[None, :]
        ).detach()
        f_next = scalar_value(model, x, q_next)
        history.append(
            {
                "iter": k,
                "f": float(f_next),
                "grad_norm": float(
                    torch.linalg.vector_norm(grad_tensor[0]).item()
                ),
                "raw_step_norm": float(diag["raw_step_norm"][0].item()),
                "applied_step_norm": float(
                    diag["applied_step_norm"][0].item()
                ),
                "q": q_next[0]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist(),
            }
        )

        if f_next > best_f:
            best_f = float(f_next)
            best_q = q_next.clone()

        if f_current * f_next <= 0.0 and f_current != f_next:
            q_zero, f_zero, root_hist = scalar_root_bisection(
                model,
                x,
                q,
                f_current,
                q_next,
                f_next,
                root_refine_iters,
                root_tolerance,
            )
            return {
                "q_zero": q_zero,
                "f_zero": float(f_zero),
                "root_source": "sign_crossing_bisection",
                "projection_history": history,
                "root_history": root_hist,
            }

        if abs(f_next) <= epsilon_f:
            return {
                "q_zero": q_next,
                "f_zero": float(f_next),
                "root_source": "projection_tolerance",
                "projection_history": history,
                "root_history": [],
            }

        q = q_next
        f_current = float(f_next)

    return {
        "q_zero": best_q,
        "f_zero": float(best_f),
        "root_source": "root_not_found_best_effort",
        "projection_history": history,
        "root_history": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scalar-checkpoint",
        default=str(
            REPO
            / "src/care_visibility_cdf/checkpoints/"
            "exp1_yiming_k500_fov_signed/final.pt"
        ),
    )
    ap.add_argument(
        "--per-sensor-checkpoint",
        default=str(
            REPO
            / "src/care_visibility_cdf/checkpoints/"
            "per_sensor_e2e_fullbatch_seed0/final.pt"
        ),
    )
    ap.add_argument(
        "--reference-urdf",
        default=str(REPO / "src/arm_description/urdf/Arm.urdf"),
    )
    ap.add_argument(
        "--self-filter-urdf",
        default=str(
            REPO
            / "src/arm_description/urdf/Arm_with_self_filter_collision.urdf"
        ),
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

    ap.add_argument("--projection-iters", type=int, default=10)
    ap.add_argument("--projection-damping", type=float, default=0.5)
    ap.add_argument("--projection-epsilon-f", type=float, default=0.03)
    ap.add_argument("--projection-max-step-norm", type=float, default=0.25)
    ap.add_argument("--root-refine-iters", type=int, default=12)
    ap.add_argument("--root-tolerance-f", type=float, default=0.002)

    ap.add_argument("--branch-ascent-steps", type=int, default=1)
    ap.add_argument("--branch-step-size", type=float, default=0.05)
    ap.add_argument("--branch-max-step-norm", type=float, default=0.25)
    ap.add_argument("--max-branch-attempts", type=int, default=8)
    ap.add_argument(
        "--force-first-sensor",
        type=int,
        default=4,
        help=(
            "Stress the original S4 failure first, then continue with the "
            "network ranking excluding S4. Use -1 for pure learned ranking."
        ),
    )
    ap.add_argument(
        "--output",
        default=str(
            REPO
            / "outputs/phase_e_case026_targeted_fallback/"
            "case026_targeted_per_sensor_fallback.json"
        ),
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    for p in (
        args.scalar_checkpoint,
        args.per_sensor_checkpoint,
        args.reference_urdf,
        args.self_filter_urdf,
    ):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    print("================================================================")
    print("CASE026 TARGETED PER-SENSOR FALLBACK")
    print("target             :", fmt(TARGET))
    print("measured seed      :", fmt(MEASURED_SEED))
    print("known blocked qvis :", fmt(KNOWN_BLOCKED_QVIS))
    print("forced first sensor:", args.force_first_sensor)
    print("================================================================")

    scalar_ckpt = torch_load_checkpoint(args.scalar_checkpoint, device)
    scalar_model, _ = build_model_from_checkpoint(scalar_ckpt, device)
    scalar_model.eval()

    q_min = torch.tensor(Q_MIN, device=device, dtype=torch.float32)
    q_max = torch.tensor(Q_MAX, device=device, dtype=torch.float32)
    x = torch.tensor(TARGET.reshape(1, 3), device=device, dtype=torch.float32)
    q_seed = torch.tensor(
        MEASURED_SEED.reshape(1, 7), device=device, dtype=torch.float32
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tic = time.perf_counter()
    scalar = scalar_project_to_zero(
        scalar_model,
        x,
        q_seed,
        q_min,
        q_max,
        args.projection_iters,
        args.projection_damping,
        args.projection_epsilon_f,
        args.projection_max_step_norm,
        args.root_refine_iters,
        args.root_tolerance_f,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scalar_ms = 1000.0 * (time.perf_counter() - tic)

    q_zero_np = (
        scalar["q_zero"][0].detach().cpu().numpy().astype(np.float64)
    )
    print("")
    print("=== SCALAR COARSE PROJECTION ===")
    print("root source :", scalar["root_source"])
    print("f_zero      :", f"{float(scalar['f_zero']):+.6f}")
    print("q_zero      :", fmt(q_zero_np))
    print("compute ms  :", f"{scalar_ms:.2f}")

    runtime = PerSensorVisibilityRuntime(
        checkpoint_path=args.per_sensor_checkpoint,
        reference_urdf_path=args.reference_urdf,
        self_filter_urdf_path=args.self_filter_urdf,
        device=device,
        q_min=Q_MIN,
        q_max=Q_MAX,
        projection_iters=args.projection_iters,
        projection_damping=args.projection_damping,
        projection_epsilon_f=args.projection_epsilon_f,
        projection_max_step_norm=args.projection_max_step_norm,
        root_refine_iters=args.root_refine_iters,
        root_tolerance_f=args.root_tolerance_f,
        branch_ascent_steps=args.branch_ascent_steps,
        branch_step_size=args.branch_step_size,
        branch_max_step_norm=args.branch_max_step_norm,
        branch_fallback_ascent_steps=8,
        max_branch_attempts=8,
        min_conservative_g=0.0,
        require_primitive_los=True,
    )

    # Reproduce the already-known failure using exactly the old q_vis.
    blocked_geometry = runtime._candidate_geometry(
        TARGET.reshape(1, 3), KNOWN_BLOCKED_QVIS, 4
    )
    print("")
    print("=== KNOWN OLD S4 Q_VIS SANITY ===")
    print(
        "conservative_g:",
        f"{blocked_geometry['min_conservative_g']:+.6f}",
    )
    print(
        "primitive_self_occluded:",
        int(blocked_geometry["any_primitive_self_occluded"]),
    )
    if blocked_geometry["per_point"]:
        print("primitive_hit:", blocked_geometry["per_point"][0]["primitive_hit"])

    points = torch.tensor(
        TARGET.reshape(1, 3), device=device, dtype=torch.float32
    )
    q_zero = torch.tensor(
        q_zero_np.reshape(1, 7), device=device, dtype=torch.float32
    )
    scores, learned_order = runtime.rank_sensors(points, q_zero)

    if args.force_first_sensor >= 0:
        first = int(args.force_first_sensor)
        if first < 0 or first >= 8:
            raise ValueError("--force-first-sensor must be -1 or [0,7]")
        order = [first] + [
            int(s) for s in learned_order.tolist() if int(s) != first
        ]
    else:
        order = [int(s) for s in learned_order.tolist()]

    print("")
    print("=== 8-HEAD SCORES AT SCALAR Q_ZERO ===")
    for s in range(8):
        print(f"S{s}: {scores[s]:+.6f}")
    print("learned ranking :", [int(v) for v in learned_order.tolist()])
    print("tested order    :", order[: args.max_branch_attempts])
    print("branch seed     : measured q (independent solve for every sensor)")
    print("branch solver   : projection -> root refinement -> ascent")

    attempts = []
    selected = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tic = time.perf_counter()

    for rank, sid in enumerate(
        order[: args.max_branch_attempts], start=1
    ):
        branch = runtime._optimize_branch(points, q_seed, sid)
        geom = runtime._candidate_geometry(
            TARGET.reshape(1, 3), branch["q_candidate"], sid
        )
        rec = dict(branch)
        rec["rank"] = rank
        rec["geometry"] = geom
        attempts.append(rec)

        hit = None
        if geom["per_point"]:
            hit = geom["per_point"][0]["primitive_hit"]
        print("")
        print(
            f"[ATTEMPT rank={rank} S{sid}] "
            f"score {branch['initial_score']:+.5f}->{branch['final_score']:+.5f} "
            f"root={branch['root_source']} "
            f"g={geom['min_conservative_g']:+.5f} "
            f"occ={int(geom['any_primitive_self_occluded'])} "
            f"accepted={int(geom['accepted'])}"
        )
        print("  q_start    :", fmt(branch["q_start"]))
        print("  q_zero     :", fmt(branch["q_zero"]))
        print("  q_candidate:", fmt(branch["q_candidate"]))
        print("  mode       :", branch["solution_mode"])
        print("  reject_reason:", geom["reject_reason"])
        if hit is not None:
            print("  hit:", hit)

        if geom["accepted"]:
            selected = rec
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    branch_ms = 1000.0 * (time.perf_counter() - tic)

    rejected = [
        int(a["sensor_id"])
        for a in attempts
        if not bool(a["geometry"]["accepted"])
    ]
    if selected is not None:
        verdict = "FALLBACK_SUCCESS" if rejected else "DIRECT_BRANCH_SUCCESS"
        selected_sid = int(selected["sensor_id"])
        selected_q = list(selected["q_candidate"])
    else:
        verdict = "NO_CLEAR_SENSOR_BRANCH"
        selected_sid = -1
        selected_q = None

    # The strongest intended result is specifically S4 rejected by LOS followed
    # by a different accepted sensor.  Record this separately from generic
    # branch success so the test cannot overclaim.
    s4_attempt = next(
        (a for a in attempts if int(a["sensor_id"]) == 4), None
    )
    strict_original_mode_fallback = bool(
        s4_attempt is not None
        and s4_attempt["geometry"]["any_primitive_self_occluded"]
        and selected is not None
        and int(selected["sensor_id"]) != 4
    )

    report = {
        "diagnostic": "phase_e_case026_targeted_per_sensor_fallback_projection_root_ascent",
        "target_xyz": TARGET.tolist(),
        "measured_seed_q": MEASURED_SEED.tolist(),
        "known_blocked_q_vis": KNOWN_BLOCKED_QVIS.tolist(),
        "known_blocked_s4_geometry": blocked_geometry,
        "scalar": {
            "checkpoint": args.scalar_checkpoint,
            "checkpoint_step": int(scalar_ckpt.get("step", -1)),
            "q_zero": q_zero_np.tolist(),
            "f_zero": float(scalar["f_zero"]),
            "root_source": scalar["root_source"],
            "projection_history": scalar["projection_history"],
            "root_history": scalar["root_history"],
            "compute_ms": float(scalar_ms),
        },
        "per_sensor": {
            "checkpoint": args.per_sensor_checkpoint,
            "scores_at_q_zero": [float(v) for v in scores.tolist()],
            "learned_ranking": [int(v) for v in learned_order.tolist()],
            "tested_order": order[: args.max_branch_attempts],
            "forced_first_sensor": int(args.force_first_sensor),
            "branch_seed_q": MEASURED_SEED.tolist(),
            "branch_solver": "projection_root_ascent",
            "attempts": attempts,
            "rejected_sensor_ids": rejected,
            "selected_sensor_id": selected_sid,
            "selected_q_vis": selected_q,
            "branch_compute_ms": float(branch_ms),
            "strict_original_mode_fallback": strict_original_mode_fallback,
        },
        "verdict": verdict,
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True))

    print("")
    print("================ FINAL TARGETED VERDICT ================")
    print("old S4 q_vis self-occluded :", int(
        blocked_geometry["any_primitive_self_occluded"]
    ))
    print("rejected sensors            :", rejected)
    print("selected sensor             :", selected_sid)
    print("strict S4->other fallback   :", int(strict_original_mode_fallback))
    print("verdict                     :", verdict)
    print("branch compute ms           :", f"{branch_ms:.2f}")
    print("[OUTPUT]", out)
    print("========================================================")


if __name__ == "__main__":
    main()

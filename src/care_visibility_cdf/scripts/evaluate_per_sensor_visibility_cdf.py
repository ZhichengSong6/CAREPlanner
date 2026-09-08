#!/usr/bin/env python3
"""Evaluate an 8-head per-sensor visibility CDF checkpoint.

Primary diagnostics:
  * per-sensor value/sign accuracy,
  * per-sensor gradient alignment,
  * union-envelope fidelity,
  * winner sensor accuracy / top-k recall,
  * fallback-sensor accuracy after the GT winner is removed.

The last metric directly measures the representation needed by the Case-026
failure mode: once the dominant sensor is rejected by self-occlusion, does the
network still rank the correct alternative sensor highest?
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from train_signed_visibility_cdf_pairwise_replace import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    MLP,
    VisibilityQ0Dataset,
    PinocchioFOVOracle,
    _parse_mlp_layers,
    sample_random_q,
    decode_per_sensor_distance_and_grad,
    make_input_pairs,
)
from train_per_sensor_visibility_cdf import (
    NUM_SENSORS,
    per_sensor_signed_targets,
)


def build_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    args = ckpt.get("args", {})
    semantics = ckpt.get("output_semantics", "")
    if semantics and semantics != "per_sensor_signed_visibility_cdf":
        raise RuntimeError(f"unexpected checkpoint semantics: {semantics}")

    raw_layers = args.get("mlp_layers", "1024,512,256,128,128")
    layers = _parse_mlp_layers(raw_layers)
    raw_skips = args.get("skips", "")
    skips = tuple(
        int(v.strip()) for v in str(raw_skips).split(",") if v.strip()
    )
    out_dim = int(ckpt.get("out_dim", 0))
    if out_dim <= 0:
        # Infer from the final linear layer for older experimental checkpoints.
        last_keys = [
            k for k in ckpt["model_state"]
            if k.endswith(".weight") and k.startswith("layers.")
        ]
        if not last_keys:
            raise RuntimeError("cannot infer output dimension")
        # state_dict insertion order follows module construction.
        out_dim = int(ckpt["model_state"][last_keys[-1]].shape[0])
    if out_dim != NUM_SENSORS:
        raise RuntimeError(f"checkpoint out_dim={out_dim}, expected 8")

    model = MLP(
        in_dim=10,
        out_dim=NUM_SENSORS,
        activation=args.get("activation", "relu"),
        model_arch=args.get("model_arch", "yiming"),
        mlp_layers=layers,
        skips=skips,
        nerf=bool(args.get("nerf", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "exp_per_sensor_yiming_k500_fov_signed/best.pt"
        ),
    )
    ap.add_argument(
        "--data",
        default=(
            "src/care_visibility_cdf/data/"
            "visibility_yiming_style_grid30_q20000_k500_fovonly.npz"
        ),
    )
    ap.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    ap.add_argument("--num-batches", type=int, default=20)
    ap.add_argument("--batch-x", type=int, default=128)
    ap.add_argument("--batch-q", type=int, default=64)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--decode-x-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument(
        "--output",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "exp_per_sensor_yiming_k500_fov_signed/"
            "per_sensor_evaluation.json"
        ),
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, ckpt = build_model(args.checkpoint, device)
    dataset = VisibilityQ0Dataset(
        args.data, val_count=args.val_count, seed=0
    )
    oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=50.0,
        vertical_fov_deg=66.0,
        z_min=0.20,
        z_max=0.70,
        delta=0.01,
    )

    agg = defaultdict(float)
    per = [defaultdict(float) for _ in range(NUM_SENSORS)]

    for bi in range(args.num_batches):
        x, qlib, valid, _ = dataset.sample_x_batch(
            args.batch_x, split="val", device=device
        )
        masks = dataset.sensor_masks(device)
        q = sample_random_q(dataset, args.batch_q, device)

        with torch.no_grad():
            d_s, grad_d_s, has_s = decode_per_sensor_distance_and_grad(
                qlib, valid, q, masks, x_chunk=args.decode_x_chunk
            )
            _, sign_s = oracle.signed_fov_margins(x, q)
            target, target_grad, valid_mask_3d = per_sensor_signed_targets(
                d_s, grad_d_s, sign_s, has_s
            )

        N = args.batch_x * args.batch_q
        inp = make_input_pairs(x, q)
        target = target.reshape(N, NUM_SENSORS)
        target_grad = target_grad.reshape(N, NUM_SENSORS, 7)
        valid_mask = valid_mask_3d.reshape(N, NUM_SENSORS)

        q_in = inp[:, 3:10].detach().clone().requires_grad_(True)
        model_in = torch.cat([inp[:, :3].detach(), q_in], dim=-1)
        pred = model(model_in)

        pred_grad = []
        for s in range(NUM_SENSORS):
            gs = torch.autograd.grad(
                pred[:, s],
                q_in,
                grad_outputs=torch.ones_like(pred[:, s]),
                retain_graph=(s < NUM_SENSORS - 1),
                create_graph=False,
                only_inputs=True,
            )[0]
            pred_grad.append(gs.detach())
        pred_grad = torch.stack(pred_grad, dim=1)
        pred = pred.detach()

        neg_inf = torch.full_like(pred, -float("inf"))
        pred_rank = torch.where(valid_mask, pred, neg_inf)
        gt_rank = torch.where(valid_mask, target, neg_inf)
        row_valid = valid_mask.any(dim=1)

        # Per-sensor metrics.
        for s in range(NUM_SENSORS):
            m = valid_mask[:, s]
            n = int(m.sum().item())
            if n == 0:
                continue
            err = pred[m, s] - target[m, s]
            cos = F.cosine_similarity(
                pred_grad[m, s, :], target_grad[m, s, :],
                dim=-1, eps=1e-6
            )
            per[s]["n"] += n
            per[s]["abs_err_sum"] += float(err.abs().sum().cpu())
            per[s]["sq_err_sum"] += float(err.square().sum().cpu())
            per[s]["sign_correct"] += int(
                ((pred[m,s] >= 0) == (target[m,s] >= 0)).sum().item()
            )
            per[s]["grad_cos_sum"] += float(cos.sum().cpu())
            per[s]["grad_norm_err_sum"] += float(
                (pred_grad[m,s,:].norm(dim=-1) - 1.0).abs().sum().cpu()
            )

        if torch.any(row_valid):
            idx = torch.where(row_valid)[0]
            gt_win = gt_rank[idx].argmax(dim=1)
            pred_win = pred_rank[idx].argmax(dim=1)
            nrow = len(idx)
            agg["rank_n"] += nrow
            agg["winner_correct"] += int((gt_win == pred_win).sum().item())

            for k in (2,3):
                pk = pred_rank[idx].topk(k, dim=1).indices
                hit = (pk == gt_win[:,None]).any(dim=1)
                agg[f"winner_top{k}_hit"] += int(hit.sum().item())

            # Remove the *true* best branch from both GT and prediction, exactly
            # matching the runtime situation after geometry rejects that mode.
            enough = valid_mask[idx].sum(dim=1) >= 2
            if torch.any(enough):
                idx2 = idx[enough]
                gt2 = gt_rank[idx2].clone()
                pr2 = pred_rank[idx2].clone()
                gt_best2 = gt2.argmax(dim=1)
                rr = torch.arange(gt2.shape[0], device=device)
                gt2[rr, gt_best2] = -float("inf")
                pr2[rr, gt_best2] = -float("inf")
                gt_fallback = gt2.argmax(dim=1)
                pred_fallback = pr2.argmax(dim=1)
                agg["fallback_n"] += int(gt2.shape[0])
                agg["fallback_correct"] += int(
                    (gt_fallback == pred_fallback).sum().item()
                )

            pu = pred_rank[idx].max(dim=1).values
            gu = gt_rank[idx].max(dim=1).values
            ue = pu - gu
            agg["union_n"] += nrow
            agg["union_abs_err_sum"] += float(ue.abs().sum().cpu())
            agg["union_sq_err_sum"] += float(ue.square().sum().cpu())
            agg["union_sign_correct"] += int(
                ((pu >= 0) == (gu >= 0)).sum().item()
            )

        print(
            f"[eval] batch {bi+1}/{args.num_batches} "
            f"winner_acc={safe_div(agg['winner_correct'],agg['rank_n']):.4f} "
            f"fallback_acc={safe_div(agg['fallback_correct'],agg['fallback_n']):.4f}",
            flush=True,
        )

    per_sensor = {}
    for s, frame in enumerate(DEFAULT_SENSOR_FRAMES):
        a = per[s]
        n = int(a["n"])
        per_sensor[f"S{s}"] = {
            "frame": frame,
            "count": n,
            "mae": safe_div(a["abs_err_sum"], n),
            "rmse": math.sqrt(safe_div(a["sq_err_sum"], n))
            if n else float("nan"),
            "sign_accuracy": safe_div(a["sign_correct"], n),
            "gradient_cosine_mean": safe_div(a["grad_cos_sum"], n),
            "gradient_norm_abs_error_mean": safe_div(
                a["grad_norm_err_sum"], n
            ),
        }

    un = int(agg["union_n"])
    result = {
        "config": vars(args),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_best_val": float(
            ckpt.get("best_val", float("nan"))
        ),
        "output_semantics": ckpt.get("output_semantics", "unknown"),
        "per_sensor": per_sensor,
        "ranking": {
            "count": int(agg["rank_n"]),
            "winner_top1_accuracy": safe_div(
                agg["winner_correct"], agg["rank_n"]
            ),
            "winner_top2_recall": safe_div(
                agg["winner_top2_hit"], agg["rank_n"]
            ),
            "winner_top3_recall": safe_div(
                agg["winner_top3_hit"], agg["rank_n"]
            ),
            "fallback_count": int(agg["fallback_n"]),
            "fallback_accuracy_after_gt_winner_removed": safe_div(
                agg["fallback_correct"], agg["fallback_n"]
            ),
        },
        "union": {
            "count": un,
            "mae": safe_div(agg["union_abs_err_sum"], un),
            "rmse": math.sqrt(safe_div(agg["union_sq_err_sum"], un))
            if un else float("nan"),
            "sign_accuracy": safe_div(
                agg["union_sign_correct"], un
            ),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print("\n=== PER-SENSOR EVALUATION ===")
    print(json.dumps(result["ranking"], indent=2))
    print(json.dumps(result["union"], indent=2))
    for s in range(NUM_SENSORS):
        print(f"S{s}: {json.dumps(per_sensor[f'S{s}'])}")
    print(f"[OUTPUT] {args.output}")


if __name__ == "__main__":
    main()

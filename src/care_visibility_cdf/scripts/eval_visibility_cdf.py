#!/usr/bin/env python3

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_visibility_cdf import (
    MLP,
    VisibilityCDFDataset,
    DEFAULT_P_MIN,
    DEFAULT_P_MAX,
    DEFAULT_Q_MIN,
    DEFAULT_Q_MAX,
    grad_wrt_q_physical,
)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["model_config"]
    model = MLP(
        input_dim=cfg["input_dim"],
        output_dim=1,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        activation=cfg["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def percentile_string(x):
    p = np.percentile(x, [0, 25, 50, 75, 90, 95, 99, 100])
    return " / ".join(f"{v:.6f}" for v in p)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained neural visibility CDF model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--q0", required=True)
    parser.add_argument("--metric", choices=["l2", "l1"], default=None)
    parser.add_argument("--target-clip", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-grad-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    norm = ckpt.get("normalization", {})
    metric = args.metric or norm.get("metric", "l2")
    target_clip = args.target_clip if args.target_clip is not None else float(norm.get("target_clip", 5.0))

    dataset = VisibilityCDFDataset(
        labels_path=args.labels,
        q0_path=args.q0,
        metric=metric,
        target_clip=target_clip,
        p_min=norm.get("p_min", DEFAULT_P_MIN),
        p_max=norm.get("p_max", DEFAULT_P_MAX),
        q_min=norm.get("q_min", DEFAULT_Q_MIN),
        q_max=norm.get("q_max", DEFAULT_Q_MAX),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    preds = []
    ys = []
    visibles = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
            ys.append(batch["y"].numpy())
            visibles.append(batch["visible"].numpy())

    pred = np.concatenate(preds)
    y = np.concatenate(ys)
    visible = np.concatenate(visibles).astype(bool)
    err = pred - y
    abs_err = np.abs(err)
    pred_visible = pred >= 0.0

    print("")
    print("=== Visibility CDF Eval Config ===")
    print(f"checkpoint:    {args.checkpoint}")
    print(f"labels:        {args.labels}")
    print(f"q0:            {args.q0}")
    print(f"device:        {device}")
    print(f"samples:       {len(dataset)}")
    print(f"metric:        {metric}")
    print(f"target_clip:   {target_clip}")

    print("")
    print("=== Regression Metrics ===")
    print(f"mse:           {np.mean(err ** 2):.9f}")
    print(f"rmse:          {math.sqrt(np.mean(err ** 2)):.9f}")
    print(f"mae:           {np.mean(abs_err):.9f}")
    print(f"max_abs_error: {np.max(abs_err):.9f}")
    print(f"target percentiles: {percentile_string(y)}")
    print(f"pred percentiles:   {percentile_string(pred)}")
    print(f"abs err percentiles:{percentile_string(abs_err)}")

    print("")
    print("=== Sign Metrics ===")
    print(f"sign_acc:             {np.mean(pred_visible == visible):.6f}")
    print(f"visible_count:         {np.count_nonzero(visible)}")
    print(f"invisible_count:       {np.count_nonzero(~visible)}")
    print(f"false_visible_count:   {np.count_nonzero(pred_visible & (~visible))}")
    print(f"false_visible_rate:    {np.mean(pred_visible & (~visible)):.6f}")
    print(f"false_invisible_count: {np.count_nonzero((~pred_visible) & visible)}")
    print(f"false_invisible_rate:  {np.mean((~pred_visible) & visible):.6f}")

    print("")
    print("=== Near Boundary Metrics ===")
    for band in [0.25, 0.5, 1.0, 2.0]:
        mask = np.abs(y) <= band
        if np.count_nonzero(mask) == 0:
            print(f"|cdf| <= {band:g}: count=0")
            continue
        print(
            f"|cdf| <= {band:g}: count={np.count_nonzero(mask)} "
            f"mae={np.mean(abs_err[mask]):.9f} "
            f"sign_acc={np.mean(pred_visible[mask] == visible[mask]):.6f}"
        )

    if args.num_grad_samples > 0:
        rng = np.random.default_rng(args.seed)
        n = min(args.num_grad_samples, len(dataset))
        idx = rng.choice(len(dataset), size=n, replace=False)
        x = torch.from_numpy(dataset.x[idx]).to(device).requires_grad_(True)
        grad_gt = torch.from_numpy(dataset.grad_q_gt[idx]).to(device)
        q_min = torch.tensor(norm.get("q_min", DEFAULT_Q_MIN), dtype=torch.float32, device=device)
        q_max = torch.tensor(norm.get("q_max", DEFAULT_Q_MAX), dtype=torch.float32, device=device)
        q_scale = 2.0 / (q_max - q_min)
        pred_g = model(x)
        grad_q = grad_wrt_q_physical(pred_g, x, q_scale)
        cos = torch.sum(grad_q * grad_gt, dim=1) / (
            torch.linalg.norm(grad_q, dim=1) * torch.linalg.norm(grad_gt, dim=1) + 1e-8
        )
        grad_norm = torch.linalg.norm(grad_q, dim=1)
        print("")
        print("=== Gradient Metrics ===")
        print(f"grad_samples:       {n}")
        print(f"grad_cos mean/med:  {cos.mean().item():.6f} / {cos.median().item():.6f}")
        print(f"grad_norm mean/med: {grad_norm.mean().item():.6f} / {grad_norm.median().item():.6f}")
        print(f"eikonal_mse:        {torch.mean((grad_norm - 1.0) ** 2).item():.9f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

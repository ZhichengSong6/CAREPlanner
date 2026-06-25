#!/usr/bin/env python3

import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from train_visibility_cdf import (  # noqa: E402
    MLP,
    OnlineVisibilityCDFSampler,
)


def merge_args(cli, ckpt):
    norm = ckpt["normalization"]
    out = {
        "urdf": cli.urdf or ckpt.get("urdf", ""),
        "q0": cli.q0 or ckpt.get("q0_path", ""),
        "occlusion_urdf": cli.occlusion_urdf if cli.occlusion_urdf is not None else ckpt.get("occlusion_urdf", ""),
        "base_frame": cli.base_frame,
        "joint_names": None,
        "sensor_frames": None,
        "ray_start_offset": cli.ray_start_offset,
        "point_end_offset": cli.point_end_offset,
        "min_hit_distance": cli.min_hit_distance,
        "min_ray_length": cli.min_ray_length,
        "ignore_start_inside": not cli.count_start_inside,
        "ignore_links": cli.ignore_links,
        "metric": norm.get("metric", "l2"),
        "target_clip": float(norm.get("target_clip", 5.0)),
        "p_min": norm["p_min"],
        "p_max": norm["p_max"],
        "q_min": norm["q_min"],
        "q_max": norm["q_max"],
        "horizontal_fov_deg": None,
        "vertical_fov_deg": None,
        "z_min": None,
        "z_max": None,
        "delta": None,
        "val_points": cli.val_points,
        "seed": cli.seed,
    }
    missing = [k for k in ["urdf", "q0"] if not out[k]]
    if missing:
        raise RuntimeError(f"Missing required paths: {missing}. Pass them explicitly.")
    return SimpleNamespace(**out)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate online-trained visibility CDF checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--urdf", default="")
    parser.add_argument("--q0", default="")
    parser.add_argument("--occlusion-urdf", default=None)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--batch-x", type=int, default=10)
    parser.add_argument("--batch-q", type=int, default=100)
    parser.add_argument("--val-points", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ray-start-offset", type=float, default=0.03)
    parser.add_argument("--point-end-offset", type=float, default=0.005)
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--count-start-inside", action="store_true")
    parser.add_argument("--ignore-links", nargs="*", default=[])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    eval_args = merge_args(args, ckpt)
    sampler = OnlineVisibilityCDFSampler(eval_args, device)

    model_cfg = ckpt["model_config"]
    model = MLP(
        model_cfg["input_dim"],
        1,
        model_cfg["hidden_dim"],
        model_cfg["num_layers"],
        model_cfg["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ys = []
    preds = []
    visibles = []

    for _ in range(args.val_batches):
        batch = sampler.sample_batch(args.batch_x, args.batch_q, split="val")
        pred = model(batch["x"])
        ys.append(batch["y"].detach().cpu())
        preds.append(pred.detach().cpu())
        visibles.append(batch["visible"].detach().cpu())

    y = torch.cat(ys).numpy()
    pred = torch.cat(preds).numpy()
    visible = torch.cat(visibles).numpy().astype(bool)

    err = pred - y
    pred_visible = pred >= 0.0
    invisible = ~visible

    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(err)))
    sign_acc = float(np.mean(pred_visible == visible))
    all_invisible_acc = float(np.mean(invisible))

    tp = int(np.count_nonzero(pred_visible & visible))
    fp = int(np.count_nonzero(pred_visible & invisible))
    tn = int(np.count_nonzero((~pred_visible) & invisible))
    fn = int(np.count_nonzero((~pred_visible) & visible))
    visible_recall = tp / max(tp + fn, 1)
    visible_precision = tp / max(tp + fp, 1)
    invisible_recall = tn / max(tn + fp, 1)

    print("")
    print("=== Online VCDF Eval Config ===")
    print(f"checkpoint:     {args.checkpoint}")
    print(f"q0:             {eval_args.q0}")
    print(f"urdf:           {eval_args.urdf}")
    print(f"occlusion_urdf: {eval_args.occlusion_urdf if eval_args.occlusion_urdf else '<none>'}")
    print(f"device:         {device}")
    print(f"samples:        {len(y)} ({args.val_batches} batches x {args.batch_x} p x {args.batch_q} q)")
    print(f"metric:         {eval_args.metric}")
    print(f"target_clip:    {eval_args.target_clip}")

    print("")
    print("=== Regression Metrics ===")
    print(f"mse:            {mse:.9f}")
    print(f"rmse:           {rmse:.9f}")
    print(f"mae:            {mae:.9f}")
    print(f"max_abs_error:  {float(np.max(np.abs(err))):.9f}")
    print("target percentiles:", np.percentile(y, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
    print("pred percentiles:  ", np.percentile(pred, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
    print("abs err percentiles:", np.percentile(np.abs(err), [0, 1, 5, 25, 50, 75, 95, 99, 100]))

    print("")
    print("=== Sign Metrics ===")
    print(f"visible_count:       {int(np.count_nonzero(visible))}")
    print(f"invisible_count:     {int(np.count_nonzero(invisible))}")
    print(f"visible_ratio:       {float(np.mean(visible)):.6f}")
    print(f"sign_acc:            {sign_acc:.6f}")
    print(f"all_invisible_acc:   {all_invisible_acc:.6f}")
    print(f"tp/fp/tn/fn:         {tp} / {fp} / {tn} / {fn}")
    print(f"visible_recall:      {visible_recall:.6f}")
    print(f"visible_precision:   {visible_precision:.6f}")
    print(f"invisible_recall:    {invisible_recall:.6f}")

    print("")
    print("=== Near Boundary Metrics ===")
    abs_y = np.abs(y)
    abs_err = np.abs(err)
    for band in [0.25, 0.5, 1.0, 2.0]:
        mask = abs_y <= band
        if np.count_nonzero(mask) == 0:
            print(f"|cdf| <= {band:g}: count=0")
            continue
        print(
            f"|cdf| <= {band:g}: count={np.count_nonzero(mask)} "
            f"ratio={np.mean(mask):.6f} mae={np.mean(abs_err[mask]):.9f} "
            f"sign_acc={np.mean(pred_visible[mask] == visible[mask]):.6f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

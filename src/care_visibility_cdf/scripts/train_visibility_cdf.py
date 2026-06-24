#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


DEFAULT_P_MIN = [-0.95, -0.95, 0.0]
DEFAULT_P_MAX = [0.95, 0.95, 1.15]
DEFAULT_Q_MIN = [-3.14, -2.3, -3.14, -3.14, -3.14, -3.14, -1.2]
DEFAULT_Q_MAX = [3.14, 2.3, 3.14, 3.14, 3.14, 3.14, 1.2]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        if activation == "relu":
            act = nn.ReLU
        elif activation == "softplus":
            act = lambda: nn.Softplus(beta=10.0)
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = [nn.Linear(input_dim, hidden_dim), act()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VisibilityCDFDataset(Dataset):
    def __init__(self, labels_path, q0_path, metric, target_clip, p_min, p_max, q_min, q_max):
        self.labels_path = labels_path
        self.q0_path = q0_path
        self.metric = metric
        self.target_clip = target_clip

        d = np.load(labels_path, allow_pickle=True)
        required = ["p", "q", "cdf", "visible", "sign", "q0_file_row", "nearest_q0_index"]
        missing = [key for key in required if key not in d]
        if missing:
            raise RuntimeError(f"Label file missing required keys: {missing}")

        self.p = d["p"].astype(np.float32)
        self.q = d["q"].astype(np.float32)
        self.visible = d["visible"].astype(np.bool_)
        self.sign = d["sign"].astype(np.float32)
        self.q0_file_row = d["q0_file_row"].astype(np.int64)
        self.nearest_q0_index = d["nearest_q0_index"].astype(np.int64)

        if metric == "l2" and "cdf_l2" in d:
            cdf = d["cdf_l2"].astype(np.float32)
        elif metric == "l1" and "cdf_l1" in d:
            cdf = d["cdf_l1"].astype(np.float32)
        else:
            cdf = d["cdf"].astype(np.float32)

        if target_clip > 0.0:
            cdf = np.clip(cdf, -target_clip, target_clip)
        self.cdf = cdf.astype(np.float32)

        if q0_path is None:
            raise RuntimeError("--q0 is required for yiming-style gradient/eikonal/tension losses.")
        q0 = np.load(q0_path, allow_pickle=True)
        if "q0_templates" not in q0:
            raise RuntimeError("q0 file missing q0_templates")
        self.q0_templates = q0["q0_templates"].astype(np.float32)

        self.p_min = np.asarray(p_min, dtype=np.float32)
        self.p_max = np.asarray(p_max, dtype=np.float32)
        self.q_min = np.asarray(q_min, dtype=np.float32)
        self.q_max = np.asarray(q_max, dtype=np.float32)

        self.x = self.normalize(self.p, self.q).astype(np.float32)
        self.grad_q_gt = self.make_l2_grad_targets().astype(np.float32)

    def normalize(self, p, q):
        p_norm = 2.0 * (p - self.p_min.reshape(1, 3)) / (self.p_max - self.p_min).reshape(1, 3) - 1.0
        q_norm = 2.0 * (q - self.q_min.reshape(1, 7)) / (self.q_max - self.q_min).reshape(1, 7) - 1.0
        return np.concatenate([p_norm, q_norm], axis=1)

    def make_l2_grad_targets(self):
        q0_nearest = self.q0_templates[self.q0_file_row, self.nearest_q0_index]
        diff = self.q - q0_nearest
        norm = np.linalg.norm(diff, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-6)
        return self.sign.reshape(-1, 1) * diff / norm

    def __len__(self):
        return len(self.cdf)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.tensor(self.cdf[idx], dtype=torch.float32),
            "grad_q_gt": torch.from_numpy(self.grad_q_gt[idx]),
            "visible": torch.tensor(self.visible[idx], dtype=torch.bool),
        }


def parse_float_list(value, expected_len, name):
    if len(value) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} expects {expected_len} values, got {len(value)}")
    return [float(v) for v in value]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grad_wrt_q_physical(pred, x, q_scale):
    grad_x = torch.autograd.grad(
        pred.sum(),
        x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad_x[:, 3:] * q_scale.view(1, 7)


def tension_loss_from_grad_q(grad_q, x, q_scale):
    cols = []
    for j in range(grad_q.shape[1]):
        grad2_x = torch.autograd.grad(
            grad_q[:, j].sum(),
            x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        cols.append(grad2_x[:, 3:] * q_scale.view(1, 7))
    hessian = torch.stack(cols, dim=2)
    return (hessian ** 2).mean()


def compute_losses(model, batch, q_scale, weights):
    x = batch["x"]
    if weights["eikonal"] > 0.0 or weights["tension"] > 0.0 or weights["grad"] > 0.0:
        x = x.detach().clone().requires_grad_(True)

    y = batch["y"]
    pred = model(x)
    mse = torch.mean((pred - y) ** 2)

    zero = pred.new_tensor(0.0)
    eikonal = zero
    tension = zero
    grad_loss = zero
    grad_cos = zero

    if weights["eikonal"] > 0.0 or weights["tension"] > 0.0 or weights["grad"] > 0.0:
        grad_q = grad_wrt_q_physical(pred, x, q_scale)
        grad_norm = torch.linalg.norm(grad_q, dim=1)
        eikonal = torch.mean((grad_norm - 1.0) ** 2)

        if weights["grad"] > 0.0:
            grad_gt = batch["grad_q_gt"]
            grad_cos_each = torch.sum(grad_q * grad_gt, dim=1) / (
                torch.linalg.norm(grad_q, dim=1) * torch.linalg.norm(grad_gt, dim=1) + 1e-8
            )
            grad_cos = grad_cos_each.mean()
            grad_loss = torch.mean(1.0 - grad_cos_each)

        if weights["tension"] > 0.0:
            tension = tension_loss_from_grad_q(grad_q, x, q_scale)

    total = (
        weights["cdf"] * mse
        + weights["eikonal"] * eikonal
        + weights["tension"] * tension
        + weights["grad"] * grad_loss
    )
    return {
        "total": total,
        "mse": mse.detach(),
        "eikonal": eikonal.detach(),
        "tension": tension.detach(),
        "grad": grad_loss.detach(),
        "grad_cos": grad_cos.detach(),
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0
    se = 0.0
    ae = 0.0
    sign_ok = 0
    false_visible = 0
    false_invisible = 0
    near = {0.25: [0, 0.0], 0.5: [0, 0.0], 1.0: [0, 0.0]}

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        visible = batch["visible"].to(device)
        pred = model(x)
        err = pred - y
        total += len(y)
        se += torch.sum(err ** 2).item()
        ae += torch.sum(torch.abs(err)).item()
        pred_visible = pred >= 0.0
        sign_ok += torch.count_nonzero(pred_visible == visible).item()
        false_visible += torch.count_nonzero(pred_visible & (~visible)).item()
        false_invisible += torch.count_nonzero((~pred_visible) & visible).item()
        abs_y = torch.abs(y)
        abs_err = torch.abs(err)
        for band in near:
            mask = abs_y <= band
            count = torch.count_nonzero(mask).item()
            if count > 0:
                near[band][0] += count
                near[band][1] += torch.sum(abs_err[mask]).item()

    out = {
        "mse": se / max(total, 1),
        "rmse": math.sqrt(se / max(total, 1)),
        "mae": ae / max(total, 1),
        "sign_acc": sign_ok / max(total, 1),
        "false_visible_rate": false_visible / max(total, 1),
        "false_invisible_rate": false_invisible / max(total, 1),
    }
    for band, (count, err_sum) in near.items():
        out[f"near_{band:g}_count"] = count
        out[f"near_{band:g}_mae"] = err_sum / count if count > 0 else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description="Train a neural visibility CDF model.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--q0", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", choices=["l2", "l1"], default="l2")
    parser.add_argument("--target-clip", type=float, default=5.0)
    parser.add_argument("--p-min", nargs=3, type=float, default=DEFAULT_P_MIN)
    parser.add_argument("--p-max", nargs=3, type=float, default=DEFAULT_P_MAX)
    parser.add_argument("--q-min", nargs=7, type=float, default=DEFAULT_Q_MIN)
    parser.add_argument("--q-max", nargs=7, type=float, default=DEFAULT_Q_MAX)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--activation", choices=["relu", "softplus", "silu"], default="relu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-cdf-weight", type=float, default=5.0)
    parser.add_argument("--loss-eikonal-weight", type=float, default=0.01)
    parser.add_argument("--loss-tension-weight", type=float, default=0.01)
    parser.add_argument("--loss-grad-weight", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = VisibilityCDFDataset(
        labels_path=args.labels,
        q0_path=args.q0,
        metric=args.metric,
        target_clip=args.target_clip,
        p_min=args.p_min,
        p_max=args.p_max,
        q_min=args.q_min,
        q_max=args.q_max,
    )

    val_len = int(round(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = MLP(10, 1, args.hidden_dim, args.num_layers, args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    q_min = torch.tensor(args.q_min, dtype=torch.float32, device=device)
    q_max = torch.tensor(args.q_max, dtype=torch.float32, device=device)
    q_scale = 2.0 / (q_max - q_min)

    weights = {
        "cdf": args.loss_cdf_weight,
        "eikonal": args.loss_eikonal_weight,
        "tension": args.loss_tension_weight,
        "grad": args.loss_grad_weight,
    }

    config = vars(args).copy()
    config.update({
        "device_used": str(device),
        "num_samples": len(dataset),
        "train_samples": train_len,
        "val_samples": val_len,
        "loss_formula": "cdf*MSE + eikonal*(||dc/dq||-1)^2 + tension*||d2c/dq2||^2 + grad*(1-cos(dc/dq, grad_gt))",
    })
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("")
    print("=== Visibility CDF Training Config ===")
    print(f"labels:        {args.labels}")
    print(f"q0:            {args.q0}")
    print(f"output_dir:    {args.output_dir}")
    print(f"device:        {device}")
    print(f"samples:       {len(dataset)} train={train_len} val={val_len}")
    print(f"model:         MLP input=10 hidden={args.hidden_dim} layers={args.num_layers} activation={args.activation}")
    print(f"loss weights:  cdf={weights['cdf']} eikonal={weights['eikonal']} tension={weights['tension']} grad={weights['grad']}")
    print(f"target_clip:   {args.target_clip}")

    best_val = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "mse": 0.0, "eikonal": 0.0, "tension": 0.0, "grad": 0.0, "grad_cos": 0.0}
        count = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, batch, q_scale, weights)
            losses["total"].backward()
            optimizer.step()

            bs = len(batch["y"])
            count += bs
            for key in sums:
                sums[key] += losses[key].item() * bs

        train_log = {key: sums[key] / max(count, 1) for key in sums}

        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if do_eval:
            val_log = eval_epoch(model, val_loader, device)
            scheduler.step(val_log["mse"])
            history.append({"epoch": epoch, "train": train_log, "val": val_log})
            print(
                f"[epoch {epoch:04d}] "
                f"train_total={train_log['total']:.6f} train_mse={train_log['mse']:.6f} "
                f"grad_cos={train_log['grad_cos']:.4f} "
                f"val_mse={val_log['mse']:.6f} val_mae={val_log['mae']:.6f} "
                f"sign_acc={val_log['sign_acc']:.4f} near0.5_mae={val_log['near_0.5_mae']:.6f} "
                f"elapsed={time.time() - t0:.1f}s"
            )

            if val_log["mse"] < best_val:
                best_val = val_log["mse"]
                save_checkpoint(args, model, optimizer, epoch, best_val, dataset, os.path.join(args.output_dir, "best.pt"))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args, model, optimizer, epoch, best_val, dataset, os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"))

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("")
    print(f"best_val_mse: {best_val:.9f}")
    print(f"saved:        {os.path.join(args.output_dir, 'best.pt')}")


def save_checkpoint(args, model, optimizer, epoch, best_val, dataset, path):
    ckpt = {
        "epoch": epoch,
        "best_val_mse": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_dim": 10,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "activation": args.activation,
        },
        "normalization": {
            "p_min": dataset.p_min.tolist(),
            "p_max": dataset.p_max.tolist(),
            "q_min": dataset.q_min.tolist(),
            "q_max": dataset.q_max.tolist(),
            "target_clip": args.target_clip,
            "metric": args.metric,
        },
        "labels_path": args.labels,
        "q0_path": args.q0,
    }
    torch.save(ckpt, path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

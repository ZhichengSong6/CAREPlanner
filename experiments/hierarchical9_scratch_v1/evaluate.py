#!/usr/bin/env python3
"""Read-only, same-sample FP32 evaluation of final hierarchical9 and baselines.

Reuses repository labels, FOV oracle and the EXISTING 10-step projection/ascent
benchmark. This does not run self-occlusion, Case026, trajectory certification,
or runtime integration. No checkpoint is converted, overwritten or fine-tuned.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import HierarchicalVisibilityCDF
from objective import supervised_targets

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "src/care_visibility_cdf/scripts"
FORMAT = "careplanner_hierarchical9_scratch_v1"


class HeadView(nn.Module):
    """Shape-only adapter; preserves the original input-autograd graph."""
    def __init__(self, model: HierarchicalVisibilityCDF, mode: str):
        super().__init__()
        if mode not in ("union", "sensors"):
            raise ValueError(mode)
        self.model, self.mode = model, mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "union":
            return self.model.forward_union(inputs)[:, None]
        h = self.model.shared(self.model.encode(inputs))
        return torch.cat([head(h) for head in self.model.sensor_heads], dim=1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def freeze(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)  # Input-q gradients REMAIN enabled.
    return model


def checkpoint(path: str) -> tuple[dict, dict]:
    p = Path(path).expanduser().resolve()
    if p.name != "final.pt" or not p.is_file():
        raise ValueError(f"Expected an existing final.pt (not best/latest/smoke): {p}")
    # Full training checkpoints contain optimizer/RNG pickle objects. Only load
    # locally trusted checkpoints produced by your own training scripts.
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Not a supported training checkpoint: {p}")
    return ckpt, {"path": str(p), "sha256": sha256(p),
                  "step": int(ckpt.get("step", -1)),
                  "training_args": ckpt.get("args", {}),
                  "output_semantics": ckpt.get("output_semantics", "unspecified")}


def load_hierarchical(path: str, device: torch.device):
    ckpt, meta = checkpoint(path)
    if (ckpt.get("format") != FORMAT or ckpt.get("step") != 50000
            or ckpt.get("initialization") != "random_from_scratch"
            or ckpt.get("out_dim") != 9):
        raise ValueError("Expected the completed 50,000-update scratch hierarchical9 final.pt")
    args = ckpt.get("args", {})
    for key, expected in {"seed": 0, "val_count": 1000, "horizontal_fov_deg": 50.0,
                          "vertical_fov_deg": 66.0, "z_min": 0.2, "z_max": 0.7,
                          "delta": 0.01}.items():
        if args.get(key) != expected:
            raise ValueError(f"Training/evaluation definition mismatch: {key}={args.get(key)}")
    model = HierarchicalVisibilityCDF()
    model.load_state_dict(ckpt["model_state"], strict=True)
    meta["saved_final_validation"] = ckpt.get("stats", {}).get("val", {})
    meta["training_metadata"] = ckpt.get("metadata", {})
    return freeze(model, device), meta


def load_legacy(path: str, out_dim: int, device: torch.device):
    from train_signed_visibility_cdf_pairwise_replace import MLP, _parse_mlp_layers
    ckpt, meta = checkpoint(path)
    args = ckpt.get("args", {})
    raw = args.get("skips", "")
    skips = tuple(int(v.strip()) for v in raw.split(",") if v.strip()) if isinstance(raw, str) else tuple(raw)
    model = MLP(in_dim=10, out_dim=out_dim,
                activation=args.get("activation", "relu"),
                model_arch=args.get("model_arch", "yiming"),
                mlp_layers=_parse_mlp_layers(args.get("mlp_layers", "1024,512,256,128,128")),
                skips=skips, nerf=bool(args.get("nerf", True)))
    model.load_state_dict(ckpt["model_state"], strict=True)
    if meta["step"] < 0:
        print(f"[warning] Legacy checkpoint has no step metadata: {path}", flush=True)
    for key, expected in (("seed", 0), ("val_count", 1000),
                          ("horizontal_fov_deg", 50.0), ("vertical_fov_deg", 66.0),
                          ("z_min", 0.2), ("z_max", 0.7), ("delta", 0.01)):
        if key in args and args[key] != expected:
            raise ValueError(f"Legacy {key} mismatch: {args[key]}; use a common held-out split")
    return freeze(model, device), meta


def output_gradients(model: nn.Module, inputs: torch.Tensor):
    with torch.enable_grad():
        q = inputs[:, 3:].detach().clone().float().requires_grad_(True)
        pred = model(torch.cat((inputs[:, :3].detach().float(), q), dim=1))
        grads = [torch.autograd.grad(pred[:, s].sum(), q,
                 retain_graph=s < pred.shape[1] - 1)[0].detach()
                 for s in range(pred.shape[1])]
    pred, grads = pred.detach(), torch.stack(grads, dim=1)
    if not torch.isfinite(pred).all() or not torch.isfinite(grads).all():
        raise RuntimeError("Non-finite field/input gradient; evaluation aborted")
    return pred, grads


def masked_union(pred, grad, mask):
    keep = mask.any(dim=1)
    pr = pred[keep].masked_fill(~mask[keep], -torch.inf)
    value, winner = pr.max(dim=1)  # Same first-winner tie convention as torch.max.
    selected_grad = grad[keep].gather(1, winner[:, None, None].expand(-1, 1, 7))[:, 0]
    return keep, value, selected_grad


class FieldStats:
    def __init__(self):
        self.n = self.sign = 0
        self.abs = self.sq = self.cos = self.norm_error = 0.0

    def add(self, pred, grad, target, target_grad):
        if not pred.numel():
            return
        err = (pred - target).double()
        self.n += pred.numel()
        self.abs += err.abs().sum().item()
        self.sq += err.square().sum().item()
        self.sign += ((pred >= 0) == (target >= 0)).sum().item()
        self.cos += F.cosine_similarity(grad, target_grad, dim=-1, eps=1e-6).double().sum().item()
        self.norm_error += (grad.norm(dim=-1) - target_grad.norm(dim=-1)).abs().double().sum().item()

    def result(self):
        n = self.n
        return {"count": n, "mae": self.abs / n if n else None,
                "rmse": math.sqrt(self.sq / n) if n else None,
                "sign_accuracy": self.sign / n if n else None,
                "gradient_cosine_mean": self.cos / n if n else None,
                "gradient_norm_abs_error_mean": self.norm_error / n if n else None}


class RankingStats:
    def __init__(self):
        self.n = self.fallback_n = self.fallback_ok = 0
        self.hits = [0, 0, 0]

    def add(self, pred, target, mask):
        row = mask.any(dim=1)
        if not row.any():
            return
        mask = mask[row]
        pred = pred[row].masked_fill(~mask, -torch.inf)
        target = target[row].masked_fill(~mask, -torch.inf)
        winner = target.argmax(dim=1)
        self.n += len(winner)
        self.hits[0] += (pred.argmax(dim=1) == winner).sum().item()
        for k in (2, 3):
            self.hits[k-1] += (pred.topk(k, dim=1).indices == winner[:, None]).any(dim=1).sum().item()
        enough = mask.sum(dim=1) >= 2
        p, t, win = pred[enough].clone(), target[enough].clone(), winner[enough]
        if len(win):
            rows = torch.arange(len(win), device=p.device)
            p[rows, win] = t[rows, win] = -torch.inf
            self.fallback_n += len(win)
            self.fallback_ok += (p.argmax(dim=1) == t.argmax(dim=1)).sum().item()

    def result(self):
        return {"count": self.n, **{f"winner_top{k}_accuracy_or_recall": self.hits[k-1] / self.n
                 if self.n else None for k in (1, 2, 3)}, "fallback_count": self.fallback_n,
                "fallback_accuracy_after_gt_winner_removed": self.fallback_ok / self.fallback_n
                if self.fallback_n else None}


def draw(dataset, bx, bq, rng, device, saved, key):
    """Independent CPU RNG: sample identity never depends on model construction."""
    indices = dataset.val_indices_cpu[rng.integers(0, len(dataset.val_indices_cpu), size=bx)]
    qlo, qhi = dataset.q_limits(device=torch.device("cpu"))
    unit = torch.from_numpy(rng.random((bq, 7)).astype(np.float32))
    q = (qlo[None] + unit * (qhi - qlo)[None]).to(device)
    saved[key + "_x_indices"] = indices.numpy()
    saved[key + "_q"] = q.cpu().numpy()
    x = dataset.x_cpu[indices].to(device)
    saved[key + "_x"] = x.cpu().numpy()
    return x, dataset.qlib_cpu[indices].to(device), dataset.valid_cpu[indices].to(device), q


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def markdown(result):
    def fmt(v):
        return "N/A" if v is None else f"{v:.5f}"
    lines = ["# Hierarchical9 final: paired offline evaluation", "",
             "All models use identical held-out samples and solver starts; FP32, no AMP.", "",
             "## Union field", "| Method | N | MAE | RMSE | Sign | Grad cosine |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in result["field"].items():
        lines.append(f"| {name} | {r['count']} | " + " | ".join(fmt(r[k]) for k in
                     ("mae", "rmse", "sign_accuracy", "gradient_cosine_mean")) + " |")
    lines += ["", "## Sensor gradient cosine", "| Sensor | New hierarchical9 | Old 8-head |",
              "|---|---:|---:|"]
    for s in range(8):
        new = result["per_sensor"]["hierarchical"][f"S{s}"]["gradient_cosine_mean"]
        old = result["per_sensor"].get("old8", {}).get(f"S{s}", {}).get("gradient_cosine_mean")
        lines.append(f"| S{s} | {fmt(new)} | {fmt(old)} |")
    lines += ["", "## Projection / ascent (FOV only)",
              "| Method | N | Proj abs(g)<.03 | Asc1 g>=.03 | Asc10 g>=.03 |",
              "|---|---:|---:|---:|---:|"]
    for name, r in result["planning"].items():
        lines.append(f"| {name} | {r['count']} | " + " | ".join(fmt(r[k]) for k in
                     ("proj_oracle_boundary_030", "asc1_g_ge_0p03", "asc10_g_ge_0p03")) + " |")
    lines += ["", "## Ranking (not executed fallback success)",
              "| Model | Top1 | Top2 | Top3 | GT-winner-removed accuracy |", "|---|---:|---:|---:|---:|"]
    for name, r in result["ranking"].items():
        lines.append(f"| {name} | " + " | ".join(fmt(r[k]) for k in
                     ("winner_top1_accuracy_or_recall", "winner_top2_accuracy_or_recall",
                      "winner_top3_accuracy_or_recall", "fallback_accuracy_after_gt_winner_removed")) + " |")
    lines += ["", "No self-occlusion, collision/path certification, Case026 replay or runtime switch was performed.",
              "All samples have the same seed0/1000-point validation split; these are not claimed to be",
              "the exact historical evaluator draws. Re-run baselines here rather than comparing old printed numbers."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = "src/care_visibility_cdf/checkpoints/"
    p.add_argument("--checkpoint", default=base + "hierarchical9_scratch_seed0/final.pt")
    p.add_argument("--scalar-checkpoint", default=base + "exp1_yiming_k500_fov_signed/final.pt")
    p.add_argument("--eight-checkpoint", default=base + "per_sensor_e2e_fullbatch_seed0/final.pt")
    p.add_argument("--skip-old-eight", action="store_true", help="Explicitly omit the optional old8 baseline")
    p.add_argument("--data", default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz")
    p.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--batch-x", type=int, default=128)
    p.add_argument("--batch-q", type=int, default=64)
    p.add_argument("--decode-x-chunk", type=int, default=64)
    p.add_argument("--planning-batches", type=int, default=10)
    p.add_argument("--planning-batch-x", type=int, default=8)
    p.add_argument("--planning-batch-q", type=int, default=64)
    args = p.parse_args()
    for key in ("num_batches", "batch_x", "batch_q", "decode_x_chunk", "planning_batches",
                "planning_batch_x", "planning_batch_q"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    for key in ("checkpoint", "scalar_checkpoint", "eight_checkpoint", "data", "urdf", "output_dir"):
        v = Path(getattr(args, key)).expanduser()
        setattr(args, key, str((REPO / v).resolve() if not v.is_absolute() else v.resolve()))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No allocated CUDA device. Use Slurm; no silent CPU fallback.")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation output: {out}")
    sys.path.insert(0, str(SCRIPTS))
    import compare_scalar_vs_per_sensor_apples_to_apples as baseline
    from train_per_sensor_visibility_cdf import per_sensor_signed_targets
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    new, nm = load_hierarchical(args.checkpoint, device)
    old, om = load_legacy(args.scalar_checkpoint, 1, device)
    models = {"hierarchical": new, "old_scalar": old}
    meta = {"hierarchical": nm, "old_scalar": om}
    views = {"old_scalar": (old, "scalar"),
             "hierarchical_union": (HeadView(new, "union"), "scalar"),
             "hierarchical_sensor_max": (HeadView(new, "sensors"), "eight")}
    if not args.skip_old_eight:
        eight, em = load_legacy(args.eight_checkpoint, 8, device)
        models["old8"], meta["old8"] = eight, em
        views["old8_sensor_max"] = (eight, "eight")
    for name, m in meta.items():
        print(f"[checkpoint] {name}: step={m['step']} sha256={m['sha256']} path={m['path']}", flush=True)
    trained_meta = nm["training_metadata"]
    if trained_meta.get("urdf_sha256") and trained_meta["urdf_sha256"] != sha256(Path(args.urdf)):
        raise ValueError("URDF content differs from the training checkpoint")
    if trained_meta.get("data_bytes") and trained_meta["data_bytes"] != Path(args.data).stat().st_size:
        raise ValueError("Dataset size differs from training; check DATA before evaluating")
    dataset = baseline.VisibilityQ0Dataset(args.data, val_count=1000, seed=0)
    if dataset.J != 7 or dataset.S != 8 or len(dataset.val_indices_cpu) != 1000:
        raise ValueError("Expected the same 7-joint, 8-sensor, 1000-point validation split")
    oracle = baseline.PinocchioFOVOracle(urdf_path=args.urdf, joint_names=baseline.DEFAULT_JOINT_NAMES,
                sensor_frames=baseline.DEFAULT_SENSOR_FRAMES, horizontal_fov_deg=50.0,
                vertical_fov_deg=66.0, z_min=0.2, z_max=0.7, delta=0.01)
    sensor_masks = dataset.sensor_masks(device)
    q_min, q_max = dataset.q_limits(device)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": vars(args), "checkpoints": meta, "protocol": {
        "precision": "FP32; AMP/TF32 disabled", "split_seed": 0, "val_count": 1000,
        "sampling": "with replacement; independent CPU RNG after model loading; samples.npz stores exact draws",
        "projection_iters": 10, "projection_damping": 0.5, "projection_max_step": 0.25,
        "ascent_step": 0.05, "ascent_max_step": 0.25, "ascent_snapshots": [1, 3, 5, 10],
        "root_refinement": False, "self_occlusion": False, "collision_safety": False,
        "max_mask": "per-x q0 availability (not a runtime visibility certificate)",
        "gradient_norm_error": "abs(norm(predicted gradient)-norm(label gradient))",
        "near_boundary_subset": "abs(discrete target field)<0.1 rad; diagnostic, not true mesh/FOV boundary"},
        "environment": {"torch": str(torch.__version__), "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "urdf_sha256": sha256(Path(args.urdf)),
        "data_identity": {"path": args.data, "bytes": Path(args.data).stat().st_size,
                          "mtime_ns": Path(args.data).stat().st_mtime_ns,
                          "note": "Size/mtime are not a dataset content checksum"},
        "source_sha256": {str(f.relative_to(REPO)): sha256(f) for f in
            (Path(__file__).resolve(), HERE / "model.py", HERE / "objective.py",
             Path(baseline.__file__), SCRIPTS / "train_signed_visibility_cdf_pairwise_replace.py",
             SCRIPTS / "train_per_sensor_visibility_cdf.py")}}
    dump(out / "manifest.json", manifest)
    fields = {name: FieldStats() for name in views}
    near = {name: FieldStats() for name in views}
    sensors = {name: [FieldStats() for _ in range(8)] for name in models if name != "old_scalar"}
    sensor_near = {name: [FieldStats() for _ in range(8)] for name in sensors}
    rankings = {name: RankingStats() for name in sensors}
    saved, rng = {}, np.random.default_rng(args.seed)
    started = time.perf_counter()
    for b in range(args.num_batches):
        x, qlib, valid, q = draw(dataset, args.batch_x, args.batch_q, rng, device, saved, f"field_{b:03d}")
        with torch.no_grad():
            ds, dg, has = baseline.decode_per_sensor_distance_and_grad(qlib, valid, q, sensor_masks,
                                                                         x_chunk=args.decode_x_chunk)
            _, sign = oracle.signed_fov_margins(x, q)
            st, sg, sm = per_sensor_signed_targets(ds, dg, sign, has)
            st, sg, sm = st.reshape(-1, 8), sg.reshape(-1, 8, 7), sm.reshape(-1, 8)
            t, tg, mask, _ = supervised_targets(st, sg, sm)
            inp = baseline.make_input_pairs(x, q)
        predictions = {name: output_gradients(m, inp) for name, m in models.items()}
        keep = mask[:, 0]
        values = {"old_scalar": (predictions["old_scalar"][0][keep, 0], predictions["old_scalar"][1][keep, 0]),
                  "hierarchical_union": (predictions["hierarchical"][0][keep, 0], predictions["hierarchical"][1][keep, 0])}
        for name in sensors:
            pred, grad = predictions[name]
            if name == "hierarchical":
                pred, grad = pred[:, 1:], grad[:, 1:]
            _, value, gradient = masked_union(pred, grad, sm)
            values["hierarchical_sensor_max" if name == "hierarchical" else "old8_sensor_max"] = (value, gradient)
            rankings[name].add(pred, st, sm)
            for s in range(8):
                for tracker, m in ((sensors[name][s], sm[:, s]),
                                   (sensor_near[name][s], sm[:, s] & (st[:, s].abs() < 0.1))):
                    tracker.add(pred[m, s], grad[m, s], st[m, s], sg[m, s])
        close = t[keep, 0].abs() < 0.1
        for name, (pred, grad) in values.items():
            fields[name].add(pred, grad, t[keep, 0], tg[keep, 0])
            near[name].add(pred[close], grad[close], t[keep, 0][close], tg[keep, 0][close])
        print(f"[field] {b+1}/{args.num_batches} " + " | ".join(
              f"{k} MAE={v.result()['mae']:.5f}" for k, v in fields.items()), flush=True)
    planning = {name: defaultdict(float) for name in views}
    rng = np.random.default_rng(args.seed + 1)
    for b in range(args.planning_batches):
        x, _, valid, q = draw(dataset, args.planning_batch_x, args.planning_batch_q, rng, device, saved,
                             f"planning_{b:03d}")
        available = valid.any(dim=1)
        if not available.any(dim=1).all():
            raise RuntimeError("Planning point without available q0 sensors")
        q0 = q[None].expand(len(x), -1, -1).contiguous()
        initial_g = baseline._oracle_pair_g(oracle, x, q0)
        for name, (view, mode) in views.items():
            avail = available if mode == "eight" else None
            qp = baseline._projection(view, mode, x, q0, q_min, q_max, 10, 0.5, 0.25, sensor_available=avail)
            fp = baseline._pair_value(view, x, qp, mode, sensor_available=avail)
            gp = baseline._oracle_pair_g(oracle, x, qp)
            snapshots = baseline._ascent_snapshots(view, mode, x, qp, q_min, q_max,
                         0.05, 0.25, (1, 3, 5, 10), sensor_available=avail)
            ga = {k: baseline._oracle_pair_g(oracle, x, qk) for k, qk in snapshots.items()}
            if not all(torch.isfinite(v).all().item() for v in (qp, fp, initial_g, gp, *ga.values())):
                raise RuntimeError(f"Non-finite planning result for {name}; not counted as a valid evaluation")
            baseline._accumulate_planning(planning[name], fp, initial_g, gp, ga)
            saved[f"planning_{b:03d}_{name}_q_projection"] = qp.cpu().numpy()
            saved[f"planning_{b:03d}_{name}_q_ascent1"] = snapshots[1].cpu().numpy()
        print(f"[planning] {b+1}/{args.planning_batches} " + " | ".join(
              f"{k} |g|<.03={v['proj_oracle_boundary_030']/v['n']:.4f}" for k, v in planning.items()), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"manifest": "manifest.json", "elapsed_seconds": time.perf_counter() - started,
              "field": {k: v.result() for k, v in fields.items()},
              "field_near_target_boundary": {k: v.result() for k, v in near.items()},
              "per_sensor": {k: {f"S{s}": a.result() for s, a in enumerate(v)} for k, v in sensors.items()},
              "per_sensor_near_target_boundary": {k: {f"S{s}": a.result() for s, a in enumerate(v)} for k, v in sensor_near.items()},
              "ranking": {k: v.result() for k, v in rankings.items()},
              "planning": {k: baseline._planning_summary(v) for k, v in planning.items()}}
    result = clean(result)
    np.savez_compressed(out / "samples.npz", **saved)
    dump(out / "comparison.json", result)
    report = markdown(result)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"[done] offline_evaluation_complete output={out}", flush=True)


if __name__ == "__main__":
    main()

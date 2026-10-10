"""Differentiate frozen objectives at frozen models. NEVER optimizer.step().

Every replica reconstructs the COMPLETE four-rank sampled batch and uses the
existing PS objective in single-process/global normalization, avoiding accidental
world-size rescaling. These are gradients of CURRENT PS losses at both models;
not a reconstruction of R1's historical objective, not causal/Adam-update proof.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from common import state_digest, write_json, write_npz


def draw_global(cache, pairs, train_cfg, step, device):
    batches, drawn = {}, {}
    world = train_cfg["world_size"]
    for kind in ("global", "v3", "v4", "boundary"):
        ids = np.concatenate([cache.ids("train", kind, train_cfg["batch"][kind],
             train_cfg["stream_seed"], step, rank, world) for rank in range(world)])
        batches[kind] = cache.batch("train", kind, ids, device)
        drawn[kind] = ids
    ids = np.concatenate([pairs.ids("train", train_cfg["batch"]["pair"],
             train_cfg["stream_seed"], step, rank, world) for rank in range(world)])
    batches["pair"] = pairs.batch("train", ids, device)
    drawn["pair"] = ids
    return batches, drawn


def parameter_groups(model):
    groups = {"shared_early": [], "union_private": []}
    groups.update({f"S{s}_private": [] for s in range(8)})
    cursor = 0
    for name, p in model.named_parameters():
        if name.startswith("early."):
            group = "shared_early"
        elif name.startswith(("union_head.", "union_tail.")):
            group = "union_private"
        elif name.startswith(("sensor_heads.", "sensor_tails.")):
            group = f"S{int(name.split('.')[1])}_private"
        else:
            raise ValueError(f"Unreviewed parameter layout: {name}")
        groups[group].append((cursor, cursor+p.numel()))
        cursor += p.numel()
    if any(not entries for entries in groups.values()):
        raise ValueError("An expected private/shared parameter group is missing")
    return groups


def flatten(grads, params):
    parts = [(torch.zeros_like(p) if g is None else g).detach().reshape(-1).cpu().double()
             for g, p in zip(grads, params)]
    result = torch.cat(parts)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite parameter gradient in read-only diagnostic")
    return result


def cosine(a, b):
    n = float(a.norm()*b.norm())
    return float(torch.dot(a, b)/n) if n > 1e-30 else None


def group_stats(vectors, total, weights, groups):
    report = {}
    keys = list(vectors)
    for group, spans in groups.items():
        local = {name: torch.cat([v[a:b] for a, b in spans]) for name, v in vectors.items()}
        summed = torch.cat([total[a:b] for a, b in spans])
        denom = sum(abs(weights[k])*float(local[k].norm()) for k in keys)
        report[group] = {
            "parameter_count": summed.numel(), "total_gradient_norm": float(summed.norm()),
            "weighted_sum_over_sum_norms": float(summed.norm())/denom if denom else None,
            "components": {k: {"unweighted_norm": float(v.norm()),
                "weighted_norm": abs(weights[k])*float(v.norm()),
                "cosine_to_total_raw_gradient": cosine(v, summed),
                "nonzero_parameter_elements": int(torch.count_nonzero(v))} for k, v in local.items()},
            "component_cosines": {a: {b: cosine(local[a], local[b]) for b in keys} for a in keys},
        }
    return report


def diagnose_model(model, batches, cfg, objective):
    if torch.distributed.is_initialized():
        raise RuntimeError("Run gradient diagnostic without a distributed process group")
    before = state_digest(model.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    model.eval().requires_grad_(True)
    params = list(model.parameters())
    components = list(cfg["weights"])
    vectors, values = {}, {}
    try:
        for component in components:
            one = copy.deepcopy(cfg)
            one["weights"] = {k: float(k == component) for k in components}
            loss, metrics = objective.loss_and_metrics(model, batches, one, training=True)
            values[component] = float(metrics["loss_" + component])
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            vectors[component] = flatten(grads, params)
            del loss, grads, metrics
        loss, _ = objective.loss_and_metrics(model, batches, cfg, training=True)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        actual_total = flatten(grads, params)
        expected_total = sum(cfg["weights"][key]*vectors[key] for key in components)
        rel = float((expected_total-actual_total).norm()/actual_total.norm().clamp_min(1e-30))
        if rel > 5e-4:
            raise RuntimeError(f"Component gradient sum does not reproduce real total: {rel}")
        norm = float(actual_total.norm())
        result = {
            "objective": "CURRENT_PS_OBJECTIVE_AT_FROZEN_WEIGHTS",
            "meaning": "Raw parameter gradients, not Adam-preconditioned updates; negative cosines alone do not establish a cause.",
            "weighted_component_values": {k: cfg["weights"][k]*v for k, v in values.items()},
            "component_values": values,
            "total_loss": float(loss.detach()), "total_parameter_gradient_norm": norm,
            "global_clip_factor_if_applied": min(1., cfg["gradient_clip_norm"]/max(norm, 1e-30)),
            "component_sum_relative_error": rel,
            "groups": group_stats(vectors, actual_total, cfg["weights"], parameter_groups(model)),
            "before_state_sha256": before,
        }
        del loss, grads, vectors, actual_total, expected_total
    finally:
        for p, requires_grad in zip(params, flags):
            p.requires_grad_(requires_grad)
    after = state_digest(model.state_dict())
    if after != before:
        raise RuntimeError("Gradient audit changed model parameters")
    result["after_state_sha256"] = after
    result["optimizer_updates"] = 0
    return result


def run_gradients(out, models, cache, pairs, training_cfg, objective, cfg, rank, device):
    replicate = rank
    if replicate >= cfg["gradient_replicates"]:
        raise ValueError("Unexpected gradient replicate")
    step = cfg["gradient_draw_step_start"] + replicate
    batches, drawn = draw_global(cache, pairs, training_cfg, step, device)
    write_npz(Path(out)/f"gradient_indices.rank{rank}.npz", drawn)
    report = {"replicate": replicate, "draw_step": step, "split": "train",
              "world_reconstruction": 4, "batch_counts": {k: len(v) for k, v in drawn.items()}, "models": {}}
    for name, model in models.items():
        print(f"[gradient rank{rank}] {name} existing objective, no optimizer updates", flush=True)
        report["models"][name] = diagnose_model(model, batches, training_cfg, objective)
    write_json(Path(out)/f"gradients.rank{rank}.json", report)

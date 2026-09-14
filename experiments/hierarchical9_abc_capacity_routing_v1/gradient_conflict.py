"""Measure P0 union/S0..S7 gradient conflict on the shared backbone.

Two matrices are reported on the same fixed held-out cohort:
- full per-head original objective (SDF + q-gradient + global Eikonal + tension)
- SDF-only value objective
No optimizer step is performed.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch.nn import functional as F

import abc_protocol as proto
from train_arm import prepare_batch_single

old = proto.old
HEADS = ["union"] + [f"s{i}" for i in range(8)]


def _cosine_report(vectors: dict[str, torch.Tensor]) -> dict:
    matrix = []
    norms = {k: float(torch.linalg.vector_norm(v).item()) for k, v in vectors.items()}
    negative = 0
    pairs = 0
    min_pair = None
    for a in HEADS:
        row = []
        for b in HEADS:
            va, vb = vectors[a], vectors[b]
            denom = torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb)
            value = float((torch.dot(va, vb) / denom.clamp_min(1e-20)).item())
            row.append(value)
            if HEADS.index(b) > HEADS.index(a):
                pairs += 1
                if value < 0:
                    negative += 1
                if min_pair is None or value < min_pair["cosine"]:
                    min_pair = {"a": a, "b": b, "cosine": value}
        matrix.append(row)
    return dict(heads=HEADS, matrix=matrix, norms=norms,
                negative_pairs=negative, total_pairs=pairs, min_pair=min_pair)


def _head_loss(model, batch, counts9, weights, obj, head: int, *, sdf_only: bool):
    inputs, sensor_target, sensor_grad, sensor_mask = batch
    target, target_grad, mask, _winner = obj.supervised_targets(sensor_target, sensor_grad, sensor_mask)
    q = inputs[:, 3:].detach().float().clone().requires_grad_(True)
    x = inputs[:, :3].detach().float()
    pred = model(torch.cat((x, q), 1)).float()
    valid = mask[:, head]
    if not valid.any() or counts9[head] <= 0:
        raise RuntimeError(f"No valid rows for head {head}")
    y = pred[:, head]
    error = y[valid] - target[valid, head]
    sdf = error.square().sum(dtype=torch.float32)
    if sdf_only:
        return sdf / counts9[head]
    grad_q = torch.autograd.grad(y.sum(), q, create_graph=True, retain_graph=True)[0].float()
    cosine = F.cosine_similarity(grad_q[valid], target_grad[valid, head], dim=-1, eps=1e-6)
    grad = (1.0-cosine).sum(dtype=torch.float32)
    norm = torch.linalg.vector_norm(grad_q[valid], dim=-1)
    eik = (norm-1.0).abs().sum(dtype=torch.float32)
    hvec = torch.autograd.grad(grad_q.sum(), q, create_graph=True, retain_graph=True)[0].float()
    tension = hvec[valid].square().sum(dtype=torch.float32)
    return (weights.sdf*sdf + weights.grad*grad + weights.eikonal*eik + weights.tension*tension) / counts9[head]


def compute(reference_root: Path, device: torch.device) -> dict:
    cache, p0, p0_sha = proto.load_reference_root(reference_root)
    baseline = old.module("train", old.SCRATCH)
    obj = old.module("objective", old.SCRATCH)
    weights = obj.LossWeights()
    artifact = Path(p0["args"]["artifact_root"])
    api = baseline.load_repo_api(old.REPO)
    dataset = api["VisibilityQ0Dataset"](str(artifact/old.DATA_REL), val_count=1000, seed=0)
    cache.verify_dataset(dataset, artifact/old.DATA_REL)
    oracle = api["PinocchioFOVOracle"](
        urdf_path=str(old.URDF), joint_names=api["DEFAULT_JOINT_NAMES"],
        sensor_frames=api["DEFAULT_SENSOR_FRAMES"], horizontal_fov_deg=50., vertical_fov_deg=66.,
        z_min=.2, z_max=.7, delta=.01,
    )
    args = proto.make_args(p0, Path("/tmp/abc_gradient_conflict"), "A", "pilot")
    args = copy.copy(args)
    args.val_global_batch_x = 96
    args.val_batch_q = 50
    args.val_microbatch_x = 96
    batches, _counts8, counts9, _ = prepare_batch_single(
        api, dataset, oracle, args, device, "val", baseline, obj
    )
    if len(batches) != 1:
        raise RuntimeError("Conflict cohort expected exactly one microbatch")
    batch = batches[0]
    model = proto.p0_model(p0, device).eval().requires_grad_(True)
    named = [(n,p) for n,p in model.named_parameters() if n.startswith("shared.")]
    params = [p for _n,p in named]
    last_ids = [i for i,(n,_p) in enumerate(named) if n.startswith("shared.4.")]
    if not last_ids:
        raise RuntimeError("Could not locate final shared 512->256 Linear")

    result = dict(parent_sha256=p0_sha, cohort_pairs=int(len(batch[0])), matrices={})
    for label, sdf_only in (("full_original_head_objective", False), ("sdf_only", True)):
        full_vectors, last_vectors = {}, {}
        for h, name in enumerate(HEADS):
            model.zero_grad(set_to_none=True)
            loss = _head_loss(model, batch, counts9, weights, obj, h, sdf_only=sdf_only)
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            full_vectors[name] = torch.cat([g.detach().reshape(-1).cpu().double() for g in grads])
            last_vectors[name] = torch.cat([grads[i].detach().reshape(-1).cpu().double() for i in last_ids])
        result["matrices"][label] = dict(
            full_shared=_cosine_report(full_vectors),
            last_shared_512_to_256=_cosine_report(last_vectors),
        )
    return result

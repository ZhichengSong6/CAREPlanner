#!/usr/bin/env python3
"""Synthetic checks, no CAREPlanner dataset/Pinocchio/GPU required.

python test_synthetic.py --device cpu
python -m torch.distributed.run --standalone --nproc_per_node=4 test_synthetic.py --ddp --device cpu
python -m torch.distributed.run --standalone --nproc_per_node=4 test_synthetic.py --ddp --device cuda
The optional --amp fp16/bf16 flag tests autocast loss arithmetic on CUDA.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from model import HierarchicalVisibilityCDF, value_and_q_grad
from objective import (LossWeights, counts_from_mask, loss_for_microbatch,
                       supervised_targets, summarize)


def synthetic(n: int = 28, device: str = "cpu"):
    generator = torch.Generator().manual_seed(217)
    inputs = torch.randn(n, 10, generator=generator).to(device)
    target = torch.randn(n, 8, generator=generator).to(device)
    grad = torch.randn(n, 8, 7, generator=generator).to(device)
    grad = torch.nn.functional.normalize(grad, dim=-1)
    mask = (torch.rand(n, 8, generator=generator) > 0.35).to(device)
    mask[0] = False  # all-sensor-invalid row
    mask[:7, 6:] = False  # rank 0 has no S6/S7 labels
    return inputs, target, grad, mask


def flat_grad(model):
    return torch.cat([p.grad.detach().flatten() if p.grad is not None else
                      torch.zeros_like(p).flatten() for p in model.parameters()])


def reference_loss(model, inputs, targets, gradients, sensor_mask, weights):
    """Independent single-batch formula; no global-count implementation reused."""
    target, target_grad, masks, _ = supervised_targets(targets, gradients, sensor_mask)
    q = inputs[:, 3:].detach().clone().requires_grad_(True)
    y = model(torch.cat((inputs[:, :3], q), dim=-1))
    terms = []
    for h in range(9):
        m = masks[:, h]
        g = torch.autograd.grad(y[:, h].sum(), q, create_graph=True, retain_graph=True)[0]
        hv = torch.autograd.grad(g.sum(), q, create_graph=True, retain_graph=True)[0]
        term = (weights.sdf * (y[m, h] - target[m, h]).square().mean()
                + weights.grad * (1 - torch.nn.functional.cosine_similarity(g[m], target_grad[m, h], dim=-1, eps=1e-6)).mean()
                + weights.eikonal * (g[m].norm(dim=-1) - 1).abs().mean()
                + weights.tension * hv[m].square().sum(dim=-1).mean())
        terms.append(term)
    valid = masks[:, 0]
    union_of_predictions = y[valid, 1:].masked_fill(~sensor_mask[valid], -torch.inf).max(dim=1).values
    cons = (y[valid, 0] - union_of_predictions).square().mean()
    return weights.union_objective * terms[0] + weights.sensor_objective * torch.stack(terms[1:]).mean() + weights.consistency * cons


def basic_tests(device: str, amp: str):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    full = HierarchicalVisibilityCDF().to(device)
    assert full.parameter_count() == 1_133_705, full.parameter_count()
    assert all(p.requires_grad for p in full.parameters())
    points = torch.randn(4, 10, device=device)
    pred = full(points)
    assert pred.shape == (4, 9)
    for s in (None, 0, 7):
        value, g = value_and_q_grad(full, points[:, :3], points[:, 3:], s)
        expected = pred[:, 0 if s is None else s + 1].detach()
        torch.testing.assert_close(value, expected, atol=2e-7, rtol=2e-6)
        assert g.shape == (4, 7) and torch.isfinite(g).all()
    # Exact same state keys/shapes are expected by the existing hierarchical loader.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        torch.save(full.state_dict(), path)
        restored = HierarchicalVisibilityCDF().to(device)
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        torch.testing.assert_close(restored(points), pred)
    # Full-size architecture: value + first/second input-derivative loss backward.
    full_batch = synthetic(n=16, device=device)
    full_counts = counts_from_mask(full_batch[-1])
    full_loss, full_stats = loss_for_microbatch(full, *full_batch, full_counts, LossWeights())
    full_loss.backward()
    assert torch.isfinite(flat_grad(full)).all() and torch.isfinite(full_stats).all()
    for module in [full.shared, full.union_head, *full.sensor_heads]:
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    del full, restored, pred, full_loss, full_stats, full_batch

    # Float64 finite differences, avoiding ambiguity from ReLU kinks.
    finite_model = HierarchicalVisibilityCDF((24, 16, 12), (12, 8)).double().to(device)
    p = (torch.randn(1, 10, device=device, dtype=torch.float64) + 0.217).requires_grad_(True)
    g = torch.autograd.grad(finite_model(p)[0, 8], p)[0][0, 3:]
    eps = 1e-6
    differences = []
    with torch.no_grad():
        for j in range(7):
            plus, minus = p.clone(), p.clone()
            plus[0, 3+j] += eps
            minus[0, 3+j] -= eps
            differences.append((finite_model(plus)[0, 8]-finite_model(minus)[0, 8]) / (2*eps))
    torch.testing.assert_close(g, torch.stack(differences), atol=1e-8, rtol=1e-4)

    batch = synthetic(device=device)
    counts = counts_from_mask(batch[-1])
    model = HierarchicalVisibilityCDF((24, 16, 12), (12, 8)).to(device)
    weights = LossWeights()
    loss, stats = loss_for_microbatch(model, *batch, counts, weights)
    assert torch.isfinite(loss) and torch.isfinite(stats).all()
    loss.backward()
    for name, module in [("trunk", model.shared), ("union", model.union_head)] + [
        (f"s{i}", head) for i, head in enumerate(model.sensor_heads)
    ]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()), name
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0, name
    reference = copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)
    ref_loss = reference_loss(reference, *batch, weights)
    ref_loss.backward()
    torch.testing.assert_close(loss.detach(), ref_loss.detach(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(flat_grad(model), flat_grad(reference), atol=3e-6, rtol=2e-4)

    # Unequal microbatch sizes must not change the globally normalized gradient.
    micro = copy.deepcopy(model)
    micro.zero_grad(set_to_none=True)
    stats_sum = torch.zeros_like(stats)
    for a, b in ((0, 3), (3, 17), (17, 28)):
        l, st = loss_for_microbatch(micro, *(v[a:b] for v in batch), counts, weights)
        l.backward()
        stats_sum += st
    torch.testing.assert_close(flat_grad(micro), flat_grad(model), atol=3e-6, rtol=2e-4)
    torch.testing.assert_close(stats_sum, stats, atol=2e-5, rtol=2e-5)
    assert abs(summarize(stats, weights)["loss"] - loss.item()) < 1e-5

    # Unavailable sensors never win the union GT; fully invalid rows stay finite.
    t = torch.tensor([[1., 999., -1., 0., 0., 0., 0., 0.], [float('inf')]*8], device=device)
    m = torch.zeros(2, 8, dtype=torch.bool, device=device)
    m[0, 0] = True
    ys, gs, ms, winner = supervised_targets(t, torch.zeros(2, 8, 7, device=device), m)
    assert ys[0, 0] == 1 and winner[0] == 0 and not ms[1].any()
    assert torch.isfinite(ys).all() and torch.isfinite(gs).all()

    # Regression for AMP denominator overflow: use half outputs with 400k counts.
    class HalfOutput(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return self.base(x).half()
    small = tuple(v[:4].clone() for v in batch)
    small[-1].fill_(True)
    normal_counts = torch.full((9,), 4.0, device=device)
    large_counts = normal_counts * 100000
    half = HalfOutput(copy.deepcopy(model))
    normal_loss, _ = loss_for_microbatch(half, *small, normal_counts, weights)
    large_loss, _ = loss_for_microbatch(half, *small, large_counts, weights)
    assert torch.isfinite(large_loss) and large_loss.item() > 0 and large_loss.dtype == torch.float32
    torch.testing.assert_close(large_loss * 100000, normal_loss, rtol=2e-5, atol=2e-6)
    assert not torch.isfinite(large_counts.half()).any()

    if device.startswith("cuda") and amp != "off":
        dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        with torch.autocast("cuda", dtype=dtype):
            amp_loss, amp_stats = loss_for_microbatch(model, *batch, counts, weights)
        assert amp_loss.dtype == torch.float32
        assert torch.isfinite(amp_loss) and torch.isfinite(amp_stats).all()
        model.zero_grad(set_to_none=True)
        amp_loss.backward()
        assert torch.isfinite(flat_grad(model)).all()
    return {"status": "PASS", "device": device, "amp": amp, "parameters": 1133705,
            "checks": ["architecture", "7D input gradients", "finite differences", "serialization",
                       "all 10 parameter groups receive gradients", "independent loss formula",
                       "microbatch invariance", "unavailable masks", "400k FP16-count regression"]}


def ddp_test(device_kind: str):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if device_kind == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    torch.set_num_threads(1)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        if world != 4:
            raise ValueError("DDP equivalence test expects exactly 4 ranks")
        torch.manual_seed(0)
        base = HierarchicalVisibilityCDF((24, 16, 12), (12, 8)).to(device)
        weights = LossWeights()
        whole = synthetic(device=str(device))
        counts = counts_from_mask(whole[-1])
        reference = copy.deepcopy(base)
        loss, _ = loss_for_microbatch(reference, *whole, counts, weights)
        loss.backward()
        expected = flat_grad(reference)
        ddp = DDP(base, device_ids=[local_rank] if device.type == "cuda" else None,
                  broadcast_buffers=False, find_unused_parameters=False)
        # Test repeated iterations, absent local heads, and no_sync accumulation.
        max_error = 0.0
        for _ in range(2):
            ddp.zero_grad(set_to_none=True)
            local = tuple(v[rank*7:(rank+1)*7] for v in whole)
            local_counts = counts_from_mask(local[-1])
            dist.all_reduce(local_counts)
            torch.testing.assert_close(local_counts, counts)
            for a, b in ((0, 3), (3, 7)):
                ctx = ddp.no_sync() if b != 7 else contextlib.nullcontext()
                with ctx:
                    value, _ = loss_for_microbatch(ddp, *(v[a:b] for v in local), counts,
                                                    weights, world_size=world)
                    value.backward()
            actual = flat_grad(ddp.module)
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=5e-4)
            max_error = max(max_error, float((actual - expected).abs().max()))
        if rank == 0:
            print(json.dumps({"status": "PASS", "test": "4-rank DDP vs full-batch parameter gradient",
                              "backend": dist.get_backend(), "max_abs_error": max_error}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    if args.ddp:
        ddp_test(args.device)
    else:
        print(json.dumps(basic_tests(args.device, args.amp), indent=2))

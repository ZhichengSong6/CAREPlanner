#!/usr/bin/env python3
"""Hierarchical union + per-sensor Visibility CDF model.

Architecture used by the next CAREPlanner VisCDF experiment:

    (x,q) -- NeRF-style [u,sin(u),cos(u)]
       |
       v
    shared trunk: 30 -> 1024 -> 512 -> 256
       |
       +-- union head: 256 -> 128 -> 128 -> 1
       +-- S0 head:    256 -> 128 -> 128 -> 1
       +-- ...
       +-- S7 head:    256 -> 128 -> 128 -> 1

Output layout is fixed:
    column 0   : dedicated union CDF
    columns 1:9: S0 ... S7 sensor-specific CDFs

The shared trunk learns common robot/point geometry while each sensor receives
its own nonlinear decoder instead of only a final linear readout.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn


NUM_SENSORS = 8
OUTPUT_DIM = 1 + NUM_SENSORS


def parse_layers(value, default: Sequence[int]) -> Tuple[int, ...]:
    if value is None:
        return tuple(int(v) for v in default)
    if isinstance(value, str):
        out = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    elif isinstance(value, Iterable):
        out = tuple(int(v) for v in value)
    else:
        raise TypeError(f"unsupported layer specification: {type(value)}")
    if not out or any(v <= 0 for v in out):
        raise ValueError(f"invalid layer specification: {value!r}")
    return out


def _make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers = []
    prev = int(in_dim)
    for width in hidden:
        width = int(width)
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


class HierarchicalVisibilityCDF(nn.Module):
    """Shared geometric trunk + dedicated union/sensor nonlinear heads."""

    def __init__(
        self,
        in_dim: int = 10,
        shared_layers=(1024, 512, 256),
        branch_layers=(128, 128),
        nerf: bool = True,
        num_sensors: int = NUM_SENSORS,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.nerf = bool(nerf)
        self.num_sensors = int(num_sensors)
        if self.num_sensors != NUM_SENSORS:
            raise ValueError(
                f"CAREPlanner hierarchical model expects {NUM_SENSORS} sensors"
            )

        self.shared_layers = parse_layers(
            shared_layers, default=(1024, 512, 256)
        )
        self.branch_layers = parse_layers(
            branch_layers, default=(128, 128)
        )
        self.encoded_dim = 3 * self.in_dim if self.nerf else self.in_dim

        shared = []
        prev = self.encoded_dim
        for width in self.shared_layers:
            shared.append(nn.Linear(prev, width))
            shared.append(nn.ReLU())
            prev = width
        self.shared = nn.Sequential(*shared)

        feature_dim = self.shared_layers[-1]
        self.union_head = _make_mlp(
            feature_dim, self.branch_layers, 1
        )
        self.sensor_heads = nn.ModuleList([
            _make_mlp(feature_dim, self.branch_layers, 1)
            for _ in range(self.num_sensors)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.nerf:
            return torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.shared(self.encode(x))
        union = self.union_head(feature)
        sensors = torch.cat(
            [head(feature) for head in self.sensor_heads], dim=-1
        )
        return torch.cat([union, sensors], dim=-1)

    @staticmethod
    def split_output(pred: torch.Tensor):
        if pred.ndim != 2 or pred.shape[1] != OUTPUT_DIM:
            raise RuntimeError(
                f"expected [N,{OUTPUT_DIM}] hierarchical output, "
                f"got {tuple(pred.shape)}"
            )
        return pred[:, 0], pred[:, 1:]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_from_checkpoint_args(args: dict) -> HierarchicalVisibilityCDF:
    return HierarchicalVisibilityCDF(
        in_dim=10,
        shared_layers=args.get("shared_layers", "1024,512,256"),
        branch_layers=args.get("branch_layers", "128,128"),
        nerf=bool(args.get("nerf", True)),
        num_sensors=NUM_SENSORS,
    )

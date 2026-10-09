"""Scratch-50k architectures for the R0/R1/R2 study.

R0: historical V1 architecture (shared 30->1024->512->256, then 9 decoders).
R1/R2: private-tail architecture (shared 30->1024->512, then nine independent
512->256 tails and nine decoders). R1 and R2 have identical parameters; only
training gradient routing differs.

All models are randomly initialized. No V1/P0/E checkpoint is loaded here.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch
from torch import nn
from torch.func import functional_call

NUM_SENSORS = 8
BRANCH_LAYERS = (128, 128)


def _decoder(in_dim: int, hidden: Sequence[int] = BRANCH_LAYERS) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_dim
    for out in hidden:
        layers += [nn.Linear(width, out), nn.ReLU()]
        width = out
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class Shared256CDF(nn.Module):
    """R0 = V1 architecture, reproduced locally for provenance isolation."""
    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 10
        self.encoded_dim = 30
        self.num_sensors = NUM_SENSORS
        self.nerf = True
        self.shared_layers = (1024, 512, 256)
        self.branch_layers = BRANCH_LAYERS
        self.shared = nn.Sequential(
            nn.Linear(30, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.union_head = _decoder(256)
        self.sensor_heads = nn.ModuleList([_decoder(256) for _ in range(NUM_SENSORS)])

    @staticmethod
    def encode(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != 10:
            raise ValueError(f"Expected [N,10], got {tuple(inputs.shape)}")
        return torch.cat((inputs, torch.sin(inputs), torch.cos(inputs)), dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.shared(self.encode(inputs))
        return torch.cat([self.union_head(h)] + [head(h) for head in self.sensor_heads], dim=1)

    def forward_union(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.union_head(self.shared(self.encode(inputs))).squeeze(-1)

    def forward_sensor(self, inputs: torch.Tensor, sensor_id: int, **_: object) -> torch.Tensor:
        if not 0 <= sensor_id < NUM_SENSORS:
            raise ValueError(sensor_id)
        return self.sensor_heads[sensor_id](self.shared(self.encode(inputs))).squeeze(-1)

    def forward_sensors(self, inputs: torch.Tensor, **_: object) -> torch.Tensor:
        h = self.shared(self.encode(inputs))
        return torch.cat([head(h) for head in self.sensor_heads], dim=1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture(self) -> dict:
        return {
            "name": "shared256_h9_v1_architecture",
            "shared": [30, 1024, 512, 256],
            "union_decoder": [256, 128, 128, 1],
            "sensor_decoders": 8,
            "sensor_decoder": [256, 128, 128, 1],
            "all_parameters_trainable": all(p.requires_grad for p in self.parameters()),
            "total_parameters": self.parameter_count(),
        }


class PrivateTailCDF(nn.Module):
    """R1/R2 architecture: 512D shared feature followed by nine private tails."""
    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 10
        self.encoded_dim = 30
        self.num_sensors = NUM_SENSORS
        self.nerf = True
        self.shared_layers = (1024, 512, 256)
        self.branch_layers = BRANCH_LAYERS
        self.early = nn.Sequential(
            nn.Linear(30, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
        )
        self.union_tail = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.sensor_tails = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 256), nn.ReLU()) for _ in range(NUM_SENSORS)
        ])
        self.union_head = _decoder(256)
        self.sensor_heads = nn.ModuleList([_decoder(256) for _ in range(NUM_SENSORS)])
        self._freeze_sensor_early_params = False

    @staticmethod
    def encode(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != 10:
            raise ValueError(f"Expected [N,10], got {tuple(inputs.shape)}")
        return torch.cat((inputs, torch.sin(inputs), torch.cos(inputs)), dim=-1)

    def _early_frozen_parameters(self, encoded: torch.Tensor) -> torch.Tensor:
        # Keep d(output)/d(input-q) exact while blocking parameter gradients into early.
        detached = {name: p.detach() for name, p in self.early.named_parameters()}
        return functional_call(self.early, detached, (encoded,))

    @contextmanager
    def sensor_early_parameter_frozen(self):
        old = self._freeze_sensor_early_params
        self._freeze_sensor_early_params = True
        try:
            yield self
        finally:
            self._freeze_sensor_early_params = old

    def _sensor_feature(self, encoded: torch.Tensor, freeze: bool) -> torch.Tensor:
        return self._early_frozen_parameters(encoded) if freeze else self.early(encoded)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(inputs)
        h_union = self.early(encoded)
        freeze = self._freeze_sensor_early_params
        h_sensor = self._sensor_feature(encoded, freeze)
        union = self.union_head(self.union_tail(h_union))
        sensors = [self.sensor_heads[s](self.sensor_tails[s](h_sensor)) for s in range(NUM_SENSORS)]
        return torch.cat([union] + sensors, dim=1)

    def forward_union(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.early(self.encode(inputs))
        return self.union_head(self.union_tail(h)).squeeze(-1)

    def forward_sensor(
        self, inputs: torch.Tensor, sensor_id: int, *, freeze_sensor_early: bool | None = None
    ) -> torch.Tensor:
        if not 0 <= sensor_id < NUM_SENSORS:
            raise ValueError(sensor_id)
        freeze = self._freeze_sensor_early_params if freeze_sensor_early is None else bool(freeze_sensor_early)
        h = self._sensor_feature(self.encode(inputs), freeze)
        return self.sensor_heads[sensor_id](self.sensor_tails[sensor_id](h)).squeeze(-1)

    def forward_sensors(
        self, inputs: torch.Tensor, *, freeze_sensor_early: bool | None = None
    ) -> torch.Tensor:
        freeze = self._freeze_sensor_early_params if freeze_sensor_early is None else bool(freeze_sensor_early)
        h = self._sensor_feature(self.encode(inputs), freeze)
        return torch.cat([self.sensor_heads[s](self.sensor_tails[s](h)) for s in range(NUM_SENSORS)], dim=1)

    def early_parameters(self) -> list[nn.Parameter]:
        return list(self.early.parameters())

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture(self) -> dict:
        return {
            "name": "private_tail_h9",
            "shared_early": [30, 1024, 512],
            "union_tail": [512, 256],
            "sensor_private_tails": 8,
            "sensor_tail": [512, 256],
            "union_decoder": [256, 128, 128, 1],
            "sensor_decoders": 8,
            "sensor_decoder": [256, 128, 128, 1],
            "all_parameters_trainable": all(p.requires_grad for p in self.parameters()),
            "total_parameters": self.parameter_count(),
        }


def build_model(arm: str) -> nn.Module:
    if arm == "R0":
        return Shared256CDF()
    if arm in ("R1", "R2"):
        return PrivateTailCDF()
    raise ValueError(f"Unknown arm {arm}")

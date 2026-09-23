"""Standalone R1 inference definition. Depends only on PyTorch.

Source of the architecture and state_dict names:
  ZhichengSong6/CAREPlanner, commit
  e9ada9d502fd622418f5fc1c28a8a52beb863364
  experiments/hierarchical9_scratch50k_r012_v1/model.py: PrivateTailCDF

This is an inference-only extraction, not a training/routing implementation.
It preserves all 9 branches. Runtime exposes S0..S7 using a shape-only view.
No ROS, Gazebo, checkpoint download, repository mutation, or execution at import.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import nn

SOURCE_COMMIT = "e9ada9d502fd622418f5fc1c28a8a52beb863364"
R1_SHA256 = "4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002"
R1_PARAMETERS = 2_184_329
NUM_SENSORS = 8


def _decoder() -> nn.Sequential:
    # Keep layer indices 0, 2, 4: they are part of checkpoint key names.
    return nn.Sequential(
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )


class PrivateTailCDF(nn.Module):
    """Full R1: [N,10] -> [N,9], ordered [union, S0, ..., S7]."""

    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 10
        self.encoded_dim = 30
        self.num_sensors = NUM_SENSORS
        self.nerf = True
        self.branch_layers = (128, 128)
        self.early = nn.Sequential(
            nn.Linear(30, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
        )
        self.union_tail = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.sensor_tails = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 256), nn.ReLU())
            for _ in range(NUM_SENSORS)
        ])
        self.union_head = _decoder()
        self.sensor_heads = nn.ModuleList([_decoder() for _ in range(NUM_SENSORS)])

    @staticmethod
    def encode(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 10:
            raise ValueError("Expected [N,10], got {}".format(tuple(inputs.shape)))
        # Block order matters. No pi factor, extra frequencies, or normalization.
        return torch.cat((inputs, torch.sin(inputs), torch.cos(inputs)), dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.early(self.encode(inputs))
        union = self.union_head(self.union_tail(h))
        sensors = [self.sensor_heads[s](self.sensor_tails[s](h))
                   for s in range(NUM_SENSORS)]
        return torch.cat([union] + sensors, dim=1)

    def forward_union(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.early(self.encode(inputs))
        return self.union_head(self.union_tail(h)).squeeze(-1)

    def forward_sensor(self, inputs: torch.Tensor, sensor_id: int) -> torch.Tensor:
        if not 0 <= sensor_id < NUM_SENSORS:
            raise ValueError("sensor_id must be in [0,7]")
        h = self.early(self.encode(inputs))
        return self.sensor_heads[sensor_id](self.sensor_tails[sensor_id](h)).squeeze(-1)

    def forward_sensors(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.early(self.encode(inputs))
        return torch.cat([
            self.sensor_heads[s](self.sensor_tails[s](h))
            for s in range(NUM_SENSORS)
        ], dim=1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HierarchicalSensorView(nn.Module):
    """Expose [N,8] sensor outputs without deleting union parameters."""

    def __init__(self, full_model: nn.Module) -> None:
        super().__init__()
        self.full_model = full_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pred = self.full_model(inputs)
        if pred.ndim != 2 or pred.shape[1] != 9:
            raise RuntimeError("Expected full output [N,9]")
        return pred[:, 1:9]


def _validate_checkpoint(checkpoint: dict) -> None:
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dictionary")
    expected = {
        "format": "care_h9_scratch50k_r012_v1",
        "arm": "R1",
        "completed": True,
        "step": 50000,
        "initialization": "random_from_scratch",
        "frozen_parameters": 0,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError("Invalid R1 metadata {}: {!r} != {!r}".format(
                key, checkpoint.get(key), value))
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Expected R1 architecture metadata")
    expected_architecture = {
        "name": "private_tail_h9",
        "shared_early": [30, 1024, 512],
        "union_tail": [512, 256],
        "sensor_tail": [512, 256],
        "union_decoder": [256, 128, 128, 1],
        "sensor_decoder": [256, 128, 128, 1],
    }
    for key, value in expected_architecture.items():
        if architecture.get(key) != value:
            raise ValueError("Unexpected architecture {}: {!r}".format(
                key, architecture.get(key)))
    for key, value in (("sensor_private_tails", 8), ("sensor_decoders", 8),
                       ("total_parameters", R1_PARAMETERS)):
        if key in architecture and architecture[key] != value:
            raise ValueError("Unexpected architecture {}: {!r}".format(
                key, architecture[key]))
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("Missing model_state")


def load_r1_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[HierarchicalSensorView, dict]:
    """Verify the fixed R1 artifact, strict-load, and return (sensor_view, ckpt).

    A full trusted training checkpoint includes optimizer/RNG metadata, hence
    weights_only=False. SHA256 is verified BEFORE deserialization. This loader
    deliberately does not accept arbitrary downloaded checkpoints.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent device fallback")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        actual = digest.hexdigest()
        if actual != R1_SHA256:
            raise ValueError("R1 SHA256 mismatch: {} != {}".format(actual, R1_SHA256))
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint)
    full = PrivateTailCDF()
    full.load_state_dict(checkpoint["model_state"], strict=True)
    if full.parameter_count() != R1_PARAMETERS:
        raise RuntimeError("Unexpected R1 parameter count")
    full = full.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    # Freezing parameters does not disable input-q autograd.
    return HierarchicalSensorView(full).eval(), checkpoint

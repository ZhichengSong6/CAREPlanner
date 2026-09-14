"""Function-preserving A/B/C routing variants built from the same P0 model.

A: freeze shared backbone + union; train existing sensor decoders only.
B: A + per-sensor zero-initialized residual adapters before the existing decoders.
C: freeze the early shared trunk through 512D; copy the original shared 512->256
   tail into eight private trainable tails, each followed by its sensor decoder.

All three variants are exactly function-equivalent to P0 before training.
"""
from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn

NUM_SENSORS = 8


def _freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    return module


class ResidualAdapter(nn.Module):
    """Small trainable residual block initialized to exact identity."""
    def __init__(self, width: int, bottleneck: int | None = None) -> None:
        super().__init__()
        bottleneck = bottleneck or max(4, width // 4)
        self.down = nn.Linear(width, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class ABCVisibilityModel(nn.Module):
    def __init__(self, p0: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in ("A", "B", "C"):
            raise ValueError(arm)
        self.arm = arm
        self.in_dim = 10
        self.num_sensors = NUM_SENSORS
        self.nerf = True
        self.shared_layers = tuple(getattr(p0, "shared_layers", (1024, 512, 256)))
        self.branch_layers = tuple(getattr(p0, "branch_layers", (128, 128)))
        if len(self.shared_layers) != 3:
            raise ValueError("ABC C split assumes exactly three shared hidden layers")

        # Union always stays exactly on the P0 path and is frozen.
        self.union_head = _freeze(copy.deepcopy(p0.union_head))

        children = list(p0.shared.children())
        if len(children) != 6:
            raise ValueError("Expected Linear/ReLU x3 shared trunk")

        if arm in ("A", "B"):
            self.shared = _freeze(copy.deepcopy(p0.shared))
            self.sensor_heads = copy.deepcopy(p0.sensor_heads)
            if arm == "B":
                width = self.shared_layers[-1]
                self.adapters = nn.ModuleList([
                    ResidualAdapter(width, max(4, width // 4)) for _ in range(NUM_SENSORS)
                ])
            else:
                self.adapters = None
        else:
            # Early trunk 30->1024->512 is immutable. The original final
            # 512->256 shared block is copied once for union and eight times for sensors.
            self.early = _freeze(nn.Sequential(*copy.deepcopy(children[:4])))
            self.union_tail = _freeze(nn.Sequential(*copy.deepcopy(children[4:])))
            self.private_tails = nn.ModuleList([
                nn.Sequential(*copy.deepcopy(children[4:])) for _ in range(NUM_SENSORS)
            ])
            self.sensor_heads = copy.deepcopy(p0.sensor_heads)

        # Defensive invariant: only intended sensor-specific modules are trainable.
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("ABC model has no trainable sensor-specific parameters")
        if any(n.startswith("union_head") for n in trainable):
            raise RuntimeError("Union head must remain frozen")
        if arm in ("A", "B") and any(n.startswith("shared") for n in trainable):
            raise RuntimeError("A/B shared backbone must remain frozen")
        if arm == "C" and any(n.startswith(("early", "union_tail")) for n in trainable):
            raise RuntimeError("C shared/union trunk must remain frozen")

    @staticmethod
    def encode(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != 10:
            raise ValueError(f"Expected [N,10], got {tuple(inputs.shape)}")
        return torch.cat((inputs, torch.sin(inputs), torch.cos(inputs)), dim=-1)

    def _sensor_features(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        e = self.encode(inputs)
        if self.arm in ("A", "B"):
            h = self.shared(e)
            if self.arm == "A":
                return [h] * NUM_SENSORS
            return [self.adapters[s](h) for s in range(NUM_SENSORS)]
        h512 = self.early(e)
        return [self.private_tails[s](h512) for s in range(NUM_SENSORS)]

    def forward_union(self, inputs: torch.Tensor) -> torch.Tensor:
        e = self.encode(inputs)
        if self.arm in ("A", "B"):
            h = self.shared(e)
        else:
            h = self.union_tail(self.early(e))
        return self.union_head(h).squeeze(-1)

    def forward_sensors(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = self._sensor_features(inputs)
        return torch.cat([
            self.sensor_heads[s](feats[s]) for s in range(NUM_SENSORS)
        ], dim=-1)

    def forward_sensor(self, inputs: torch.Tensor, sensor_id: int) -> torch.Tensor:
        if not 0 <= sensor_id < NUM_SENSORS:
            raise ValueError(sensor_id)
        e = self.encode(inputs)
        if self.arm in ("A", "B"):
            h = self.shared(e)
            if self.arm == "B":
                h = self.adapters[sensor_id](h)
        else:
            h = self.private_tails[sensor_id](self.early(e))
        return self.sensor_heads[sensor_id](h).squeeze(-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.forward_union(inputs)[:, None], self.forward_sensors(inputs)), dim=1)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture(self) -> dict:
        if self.arm == "A":
            routing = "frozen_P0_shared_plus_train_existing_sensor_decoders"
        elif self.arm == "B":
            routing = "frozen_P0_shared_plus_identity_residual_adapter_plus_sensor_decoder"
        else:
            routing = "frozen_early_30_1024_512_plus_private_copied_512_256_tail_plus_sensor_decoder"
        return dict(
            arm=self.arm,
            routing=routing,
            shared_layers=list(self.shared_layers),
            branch_layers=list(self.branch_layers),
            union_path_frozen=True,
            total_parameters=self.total_parameter_count(),
            trainable_parameters=self.trainable_parameter_count(),
        )


def build_arm(p0: nn.Module, arm: str) -> ABCVisibilityModel:
    model = ABCVisibilityModel(p0, arm)
    return model


def function_equivalence(p0: nn.Module, candidate: ABCVisibilityModel, inputs: torch.Tensor) -> dict:
    """Check exact-at-initialization values and raw-q gradients for all nine outputs."""
    p0.eval(); candidate.eval()
    with torch.no_grad():
        ref = p0(inputs.float())
        got = candidate(inputs.float())
        value_error = float((ref - got).abs().max().item())
    q0 = inputs[:, 3:].detach().float().clone().requires_grad_(True)
    x0 = inputs[:, :3].detach().float()
    q1 = q0.detach().clone().requires_grad_(True)
    ref_out = p0(torch.cat((x0, q0), 1)).float()
    got_out = candidate(torch.cat((x0, q1), 1)).float()
    grad_error = 0.0
    for h in range(9):
        gr = torch.autograd.grad(ref_out[:, h].sum(), q0, retain_graph=True)[0]
        gg = torch.autograd.grad(got_out[:, h].sum(), q1, retain_graph=True)[0]
        grad_error = max(grad_error, float((gr - gg).abs().max().item()))
    return {"max_abs_value_error": value_error, "max_abs_q_gradient_error": grad_error}

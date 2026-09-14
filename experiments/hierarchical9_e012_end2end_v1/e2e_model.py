"""End-to-end hierarchical9 with a shared early trunk and nine private tails.

All parameters are trainable.  The constructor copies a completed P0 model so
step-0 values and raw-q gradients are exactly P0-equivalent, but the saved E0/E1/E2
checkpoint is a complete independent network: shared trunk, union path and sensor
paths all live in this module and are updated during training.

For E1/E2 only, sensor losses can use a *parameter-frozen* shared-early call while
preserving input-q derivatives.  This is intentionally NOT tensor.detach(): the
sensor q-gradient/Eikonal/tension objectives still see the true early Jacobian.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterable

import torch
from torch import nn
from torch.func import functional_call

NUM_SENSORS = 8


class EndToEndPrivateTailCDF(nn.Module):
    def __init__(self, p0: nn.Module) -> None:
        super().__init__()
        self.in_dim = 10
        self.encoded_dim = 30
        self.num_sensors = NUM_SENSORS
        self.nerf = True
        self.shared_layers = tuple(getattr(p0, "shared_layers", (1024, 512, 256)))
        self.branch_layers = tuple(getattr(p0, "branch_layers", (128, 128)))
        if len(self.shared_layers) != 3:
            raise ValueError("Private-tail split assumes three P0 shared hidden layers")
        children = list(p0.shared.children())
        if len(children) != 6:
            raise ValueError("Expected P0 shared trunk = Linear/ReLU x3")

        # Shared representation: 30 -> 1024 -> 512.
        self.early = nn.Sequential(*copy.deepcopy(children[:4]))

        # P0's final 512 -> 256 + ReLU is copied into one union tail and eight
        # sensor-private tails.  This makes the whole model function-equivalent
        # to P0 before the first optimizer step.
        self.union_tail = nn.Sequential(*copy.deepcopy(children[4:]))
        self.sensor_tails = nn.ModuleList([
            nn.Sequential(*copy.deepcopy(children[4:])) for _ in range(NUM_SENSORS)
        ])
        self.union_head = copy.deepcopy(p0.union_head)
        self.sensor_heads = copy.deepcopy(p0.sensor_heads)

        # A loaded P0 may have requires_grad=False for evaluation.  E0/E1/E2 are
        # end-to-end training arms, so explicitly re-enable every copied parameter.
        self.requires_grad_(True)
        self._freeze_sensor_early_params = False

    @staticmethod
    def encode(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != 10:
            raise ValueError(f"Expected [N,10], got {tuple(inputs.shape)}")
        return torch.cat((inputs, torch.sin(inputs), torch.cos(inputs)), dim=-1)

    def _early_frozen_parameters(self, encoded: torch.Tensor) -> torch.Tensor:
        """Use detached early weights while retaining derivatives w.r.t. encoded.

        `encoded.detach()` would destroy q gradients, so E1/E2 instead run the
        exact same early module through torch.func.functional_call with detached
        parameter tensors.  Values and q derivatives are unchanged; only parameter
        gradients into `early` are blocked on this sensor path.
        """
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

    def _features(self, inputs: torch.Tensor, *, freeze_sensor_early: bool | None = None):
        encoded = self.encode(inputs)
        freeze = self._freeze_sensor_early_params if freeze_sensor_early is None else bool(freeze_sensor_early)
        h_union = self.early(encoded)
        h_sensor = self._early_frozen_parameters(encoded) if freeze else h_union
        return h_union, h_sensor

    def forward_union(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.early(self.encode(inputs))
        return self.union_head(self.union_tail(h)).squeeze(-1)

    def forward_sensor(
        self, inputs: torch.Tensor, sensor_id: int, *, freeze_sensor_early: bool | None = None
    ) -> torch.Tensor:
        if not 0 <= sensor_id < NUM_SENSORS:
            raise ValueError(sensor_id)
        encoded = self.encode(inputs)
        freeze = self._freeze_sensor_early_params if freeze_sensor_early is None else bool(freeze_sensor_early)
        h = self._early_frozen_parameters(encoded) if freeze else self.early(encoded)
        return self.sensor_heads[sensor_id](self.sensor_tails[sensor_id](h)).squeeze(-1)

    def forward_sensors(
        self, inputs: torch.Tensor, *, freeze_sensor_early: bool | None = None
    ) -> torch.Tensor:
        encoded = self.encode(inputs)
        freeze = self._freeze_sensor_early_params if freeze_sensor_early is None else bool(freeze_sensor_early)
        h = self._early_frozen_parameters(encoded) if freeze else self.early(encoded)
        return torch.cat([
            self.sensor_heads[s](self.sensor_tails[s](h)) for s in range(NUM_SENSORS)
        ], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h_union, h_sensor = self._features(inputs)
        union = self.union_head(self.union_tail(h_union))
        sensors = [
            self.sensor_heads[s](self.sensor_tails[s](h_sensor)) for s in range(NUM_SENSORS)
        ]
        return torch.cat([union] + sensors, dim=1)

    def early_parameters(self) -> list[nn.Parameter]:
        return list(self.early.parameters())

    def optimizer_groups(self, *, lr_early: float, lr_union: float, lr_sensor: float):
        return [
            {"name": "early", "params": list(self.early.parameters()), "lr": lr_early},
            {"name": "union", "params": list(self.union_tail.parameters()) + list(self.union_head.parameters()), "lr": lr_union},
            {"name": "sensor", "params": list(self.sensor_tails.parameters()) + list(self.sensor_heads.parameters()), "lr": lr_sensor},
        ]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict:
        return {
            "name": "end_to_end_private_tail_h9",
            "shared_early": [30, 1024, 512],
            "union_tail": [512, 256],
            "sensor_private_tails": 8,
            "sensor_tail": [512, 256],
            "branch_layers": list(self.branch_layers),
            "all_parameters_trainable": all(p.requires_grad for p in self.parameters()),
            "total_parameters": self.parameter_count(),
            "trainable_parameters": self.trainable_parameter_count(),
        }


def build_from_p0(p0: nn.Module) -> EndToEndPrivateTailCDF:
    return EndToEndPrivateTailCDF(p0)


def function_equivalence(p0: nn.Module, candidate: EndToEndPrivateTailCDF, inputs: torch.Tensor) -> dict:
    """Check P0 equivalence for all nine values and all nine raw-q gradients."""
    p0.eval(); candidate.eval()
    with torch.no_grad():
        ref = p0(inputs.float())
        got = candidate(inputs.float())
        value_error = float((ref - got).abs().max().item())
    x = inputs[:, :3].detach().float()
    q0 = inputs[:, 3:].detach().float().clone().requires_grad_(True)
    q1 = q0.detach().clone().requires_grad_(True)
    ref = p0(torch.cat((x, q0), 1)).float()
    got = candidate(torch.cat((x, q1), 1)).float()
    grad_error = 0.0
    for h in range(9):
        gr = torch.autograd.grad(ref[:, h].sum(), q0, retain_graph=True)[0]
        gg = torch.autograd.grad(got[:, h].sum(), q1, retain_graph=True)[0]
        grad_error = max(grad_error, float((gr - gg).abs().max().item()))
    return {"max_abs_value_error": value_error, "max_abs_q_gradient_error": grad_error}

"""Compatibility adapter for legacy evaluation helpers.

The original hierarchical9 planning sentinel's HeadView assumes every model has
one `.shared` tensor feeding all sensor heads.  Arm C intentionally violates that
assumption by using a private 512->256 tail per sensor.  This adapter preserves
exact values and input-q gradients while presenting only the narrow interface
expected by that legacy HeadView.  It is evaluation-only and has no trainable
or checkpoint semantics of its own.
"""
from __future__ import annotations

import torch
from torch import nn


class _SensorCall(nn.Module):
    """Call one branch without registering the wrapped model a second time."""
    def __init__(self, model: nn.Module, sensor_id: int) -> None:
        super().__init__()
        object.__setattr__(self, "_wrapped_model", model)
        self.sensor_id = int(sensor_id)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        model = object.__getattribute__(self, "_wrapped_model")
        return model.forward_sensor(inputs, self.sensor_id)[:, None]


class LegacyHeadViewCompatible(nn.Module):
    """Expose a generic branch model to the old `evaluate.HeadView` contract.

    For the legacy sensor path, `encode -> shared -> each sensor_head` is turned
    into identity -> identity -> `forward_sensor(s)`.  Therefore the legacy
    helper obtains exactly the model's real sensor values/gradients without
    pretending that Arm C has one common 256D feature.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.wrapped = model
        self.shared = nn.Identity()
        self.sensor_heads = nn.ModuleList([_SensorCall(model, s) for s in range(8)])

    @staticmethod
    def encode(inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def forward_union(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.wrapped.forward_union(inputs)

    def forward_sensor(self, inputs: torch.Tensor, sensor_id: int) -> torch.Tensor:
        return self.wrapped.forward_sensor(inputs, sensor_id)

    def forward_sensors(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.wrapped.forward_sensors(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.wrapped(inputs)


def legacy_compatible(model: nn.Module) -> LegacyHeadViewCompatible:
    return LegacyHeadViewCompatible(model)

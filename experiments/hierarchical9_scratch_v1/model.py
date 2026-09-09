"""CAREPlanner hierarchical9: unchanged depth, nine independent nonlinear heads.

State-dict keys match hierarchical_visibility_cdf_model.HierarchicalVisibilityCDF
at repository revision 841a4a992386388aa0cdb00c885f3c607c8e997c.
This file performs NO pretrained loading and freezes NO parameters.
"""
from __future__ import annotations

from typing import Sequence
import torch
from torch import nn

NUM_SENSORS = 8
OUTPUT_DIM = 9
SHARED_LAYERS = (1024, 512, 256)
BRANCH_LAYERS = (128, 128)


def _decoder(input_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


class HierarchicalVisibilityCDF(nn.Module):
    """Input [N,10]=[x,q]; output [N,9]=[union,S0,...,S7].

    The optional layer arguments support small synthetic unit tests. The formal
    training entrypoint always constructs the fixed default architecture.
    """
    def __init__(
        self,
        shared_layers: Sequence[int] = SHARED_LAYERS,
        branch_layers: Sequence[int] = BRANCH_LAYERS,
    ) -> None:
        super().__init__()
        self.in_dim = 10
        self.encoded_dim = 30
        self.nerf = True
        self.num_sensors = NUM_SENSORS
        self.shared_layers = tuple(shared_layers)
        self.branch_layers = tuple(branch_layers)
        if not self.shared_layers or not self.branch_layers:
            raise ValueError("Both trunk and decoder must contain hidden layers")
        if any(w <= 0 for w in self.shared_layers + self.branch_layers):
            raise ValueError("Hidden dimensions must be positive")
        layers: list[nn.Module] = []
        width = self.encoded_dim
        for out_width in self.shared_layers:
            layers.extend((nn.Linear(width, out_width), nn.ReLU()))
            width = out_width
        self.shared = nn.Sequential(*layers)
        self.union_head = _decoder(width, self.branch_layers)
        self.sensor_heads = nn.ModuleList([
            _decoder(width, self.branch_layers) for _ in range(NUM_SENSORS)
        ])

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != 10:
            raise ValueError(f"Expected [N,10], received {tuple(inputs.shape)}")
        return torch.cat((inputs, torch.sin(inputs), torch.cos(inputs)), dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.shared(self.encode(inputs))
        return torch.cat(
            [self.union_head(h)] + [head(h) for head in self.sensor_heads], dim=-1
        )

    def forward_union(self, inputs: torch.Tensor) -> torch.Tensor:
        """Skip all eight sensor decoders; no detach/no_grad on the input path."""
        return self.union_head(self.shared(self.encode(inputs))).squeeze(-1)

    def forward_sensor(self, inputs: torch.Tensor, sensor_id: int) -> torch.Tensor:
        """Skip union and the other seven decoders."""
        if not 0 <= sensor_id < NUM_SENSORS:
            raise ValueError(f"sensor_id must be in [0,7], got {sensor_id}")
        return self.sensor_heads[sensor_id](
            self.shared(self.encode(inputs))
        ).squeeze(-1)

    @staticmethod
    def split_output(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pred.ndim != 2 or pred.shape[-1] != OUTPUT_DIM:
            raise ValueError(f"Expected [N,9], received {tuple(pred.shape)}")
        return pred[:, 0], pred[:, 1:]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def value_and_q_grad(
    model: HierarchicalVisibilityCDF,
    x: torch.Tensor,
    q: torch.Tensor,
    sensor_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 evaluation helper, NOT automatic planner/runtime integration.

    x:[N,3], q:[N,7], both on the model's device. sensor_id=None selects UNION.
    Parameters may be frozen, but input autograd must remain enabled. Do not
    invoke inside torch.inference_mode(). No point-set min/obligation aggregation
    is done here: retain that aggregation in the existing runtime adapter.
    """
    if x.ndim != 2 or q.ndim != 2 or x.shape != (q.shape[0], 3) or q.shape[1] != 7:
        raise ValueError("Expected matching x:[N,3] and q:[N,7]")
    with torch.enable_grad(), torch.autocast(q.device.type, enabled=False):
        q_var = q.detach().float().clone().requires_grad_(True)
        inputs = torch.cat((x.detach().float(), q_var), dim=-1)
        value = (model.forward_union(inputs) if sensor_id is None else
                 model.forward_sensor(inputs, sensor_id))
        grad = torch.autograd.grad(value.sum(), q_var)[0]
    return value.detach(), grad.detach()

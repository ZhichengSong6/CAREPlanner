#!/usr/bin/env python3
"""Yiming-style neural configuration-space distance field.

The network architecture mirrors idiap/cdf/frankaemika/mlp.py but is kept local
so CAREPlanner does not need the original CDF repository at runtime.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn


def _mlp(channels, act_fn=nn.ReLU, islast=False):
    # Preserve the nested Sequential structure used by idiap/cdf so original
    # model_dict.pt state_dict key names match exactly.
    if not islast:
        layers = [
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), act_fn())
            for i in range(1, len(channels))
        ]
    else:
        layers = [
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), act_fn())
            for i in range(1, len(channels) - 1)
        ]
        layers.append(nn.Sequential(nn.Linear(channels[-2], channels[-1])))
    return nn.Sequential(*layers)


class MLPRegression(nn.Module):
    """Runtime-compatible copy of idiap/cdf frankaemika.mlp.MLPRegression."""

    def __init__(
        self,
        input_dims: int = 10,
        output_dims: int = 1,
        mlp_layers: Iterable[int] = (1024, 512, 256, 128, 128),
        skips=(),
        act_fn=nn.ReLU,
        nerf: bool = True,
    ):
        super().__init__()
        input_dims = int(input_dims)
        self.nerf = bool(nerf)
        if self.nerf:
            input_dims = 3 * input_dims

        # The original implementation mutates mlp_layers in-place. Work on a
        # fresh list but reproduce the same resulting module hierarchy.
        widths = [int(v) for v in mlp_layers]
        skips = [int(v) for v in skips]
        mlp_arr = []
        if skips:
            mlp_arr.append(widths[0:skips[0]])
            mlp_arr[0][-1] -= input_dims
            for s in range(1, len(skips)):
                mlp_arr.append(widths[skips[s - 1]:skips[s]])
                mlp_arr[-1][-1] -= input_dims
            mlp_arr.append(widths[skips[-1]:])
        else:
            mlp_arr.append(widths)

        mlp_arr[-1].append(int(output_dims))
        mlp_arr[0].insert(0, input_dims)

        self.layers = nn.ModuleList()
        for arr in mlp_arr[:-1]:
            self.layers.append(_mlp(arr, act_fn=act_fn, islast=False))
        self.layers.append(_mlp(mlp_arr[-1], act_fn=act_fn, islast=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nerf:
            x_nerf = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        else:
            x_nerf = x
        y = self.layers[0](x_nerf)
        for layer in self.layers[1:]:
            y = layer(torch.cat((y, x_nerf), dim=1))
        return y


def _looks_like_state_dict(obj) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    )


def _strip_common_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict(state)
    for prefix in ("module.", "model."):
        if out and all(k.startswith(prefix) for k in out):
            out = {k[len(prefix):]: v for k, v in out.items()}
    return out


def extract_state_dict(payload, checkpoint_key: str = "latest") -> Tuple[Dict[str, torch.Tensor], str]:
    """Return (state_dict, selected_key_description)."""
    if _looks_like_state_dict(payload):
        return _strip_common_prefixes(payload), "raw_state_dict"

    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint payload: expected a dictionary/state_dict")

    for key in ("model_state_dict", "state_dict", "model"):
        candidate = payload.get(key)
        if _looks_like_state_dict(candidate):
            return _strip_common_prefixes(candidate), key

    numeric = []
    for key, value in payload.items():
        if _looks_like_state_dict(value):
            try:
                numeric.append((int(key), key, value))
            except (TypeError, ValueError):
                pass

    if numeric:
        numeric.sort(key=lambda item: item[0])
        if str(checkpoint_key).lower() == "latest":
            _, key, state = numeric[-1]
            return _strip_common_prefixes(state), f"iteration:{key}"

        wanted = int(checkpoint_key)
        for ikey, key, state in numeric:
            if ikey == wanted:
                return _strip_common_prefixes(state), f"iteration:{key}"
        available = [v[0] for v in numeric]
        raise KeyError(
            f"checkpoint iteration {wanted} not found; available range "
            f"{min(available)}..{max(available)}"
        )

    raise ValueError(
        "Could not find a model state_dict in checkpoint. "
        "Run inspect_collision_cdf_checkpoint.py to inspect its structure."
    )


class CollisionCDF:
    """Scene-level CDF evaluator d(q)=min_p f(p,q) with autograd gradient."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        checkpoint_key: str = "latest",
        input_dims: int = 10,
        output_dims: int = 1,
        hidden_layers=(1024, 512, 256, 128, 128),
        nerf: bool = True,
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for collision CDF but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.model = MLPRegression(
            input_dims=input_dims,
            output_dims=output_dims,
            mlp_layers=hidden_layers,
            nerf=nerf,
        ).to(self.device)

        payload = torch.load(checkpoint_path, map_location=self.device)
        state, selected = extract_state_dict(payload, checkpoint_key)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.selected_checkpoint = selected

    def scene_distance_and_gradient(
        self, points_xyz: torch.Tensor, q: torch.Tensor
    ):
        """Return min distance [Nq], gradient [Nq,7], argmin point [Nq]."""
        points_xyz = points_xyz.to(self.device, dtype=torch.float32)
        q = q.to(self.device, dtype=torch.float32).detach().clone()
        q.requires_grad_(True)

        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError("points_xyz must have shape [Np,3]")
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError("q must have shape [Nq,7]")
        if points_xyz.shape[0] == 0 or q.shape[0] == 0:
            raise ValueError("points_xyz and q must both be non-empty")

        np_, nq = points_xyz.shape[0], q.shape[0]
        x_cat = points_xyz[:, None, :].expand(np_, nq, 3).reshape(-1, 3)
        q_cat = q[None, :, :].expand(np_, nq, 7).reshape(-1, 7)
        inputs = torch.cat((x_cat, q_cat), dim=-1)
        pred = self.model(inputs).reshape(np_, nq)

        min_distance, argmin = torch.min(pred, dim=0)
        grad = torch.autograd.grad(
            min_distance.sum(), q, retain_graph=False, create_graph=False
        )[0]
        return min_distance.detach(), grad.detach(), argmin.detach()

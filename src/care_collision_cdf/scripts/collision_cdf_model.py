#!/usr/bin/env python3
"""Yiming-style neural configuration-space distance field.

The network architecture mirrors idiap/cdf/frankaemika/mlp.py but is kept local
so CAREPlanner does not need the original CDF repository at runtime.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple
import re

import torch
from torch import nn


def resolve_activation(name: str):
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU
    if key == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported collision CDF activation: {name}")


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
        activation: str = "gelu",
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


def infer_mlp_architecture(
    state: Dict[str, torch.Tensor],
    raw_input_dims: int = 10,
):
    """Infer the no-skip Yiming MLP architecture from a state_dict.

    Expected original-CDF key layout:
      layers.0.0.0.weight
      layers.0.1.0.weight
      ...

    Returns a dict with raw input dims, output dims, hidden layers and nerf flag.
    """
    pattern = re.compile(r"^layers\.0\.(\d+)\.0\.weight$")
    weights = []
    for key, value in state.items():
        match = pattern.match(key)
        if match and torch.is_tensor(value) and value.ndim == 2:
            weights.append((int(match.group(1)), key, value))
    weights.sort(key=lambda item: item[0])

    if not weights:
        raise ValueError(
            "Could not infer Yiming MLP architecture: no "
            "layers.0.<i>.0.weight tensors found"
        )

    expected = list(range(len(weights)))
    actual = [item[0] for item in weights]
    if actual != expected:
        raise ValueError(
            f"Unsupported/non-contiguous Yiming MLP layer indices: {actual}"
        )

    dims = []
    previous_out = None
    for idx, key, weight in weights:
        out_features, in_features = map(int, weight.shape)
        if previous_out is None:
            dims.append(in_features)
        elif in_features != previous_out:
            raise ValueError(
                f"Unsupported skip/branched architecture at {key}: "
                f"in_features={in_features}, previous_out={previous_out}"
            )
        dims.append(out_features)
        previous_out = out_features

    encoded_input_dims = int(dims[0])
    if encoded_input_dims == int(raw_input_dims) * 3:
        nerf = True
    elif encoded_input_dims == int(raw_input_dims):
        nerf = False
    else:
        raise ValueError(
            f"Checkpoint first layer expects {encoded_input_dims} features; "
            f"expected {raw_input_dims} (plain) or {raw_input_dims*3} (NeRF encoding)"
        )

    return {
        "input_dims": int(raw_input_dims),
        "encoded_input_dims": encoded_input_dims,
        "output_dims": int(dims[-1]),
        "hidden_layers": [int(v) for v in dims[1:-1]],
        "nerf": bool(nerf),
        "linear_dims": [int(v) for v in dims],
    }


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

        # Load weights first, then recover the exact no-skip Yiming MLP widths
        # from the checkpoint. This avoids coupling runtime to a hand-written
        # hidden_layers list and supports larger retrained CDFs.
        payload = torch.load(checkpoint_path, map_location=self.device)
        state, selected = extract_state_dict(payload, checkpoint_key)
        architecture = infer_mlp_architecture(state, raw_input_dims=input_dims)
        activation_name = str(activation).strip().lower()
        activation_cls = resolve_activation(activation_name)

        if int(output_dims) != int(architecture["output_dims"]):
            raise ValueError(
                f"Configured output_dims={output_dims} but checkpoint has "
                f"{architecture['output_dims']}"
            )

        self.model = MLPRegression(
            input_dims=architecture["input_dims"],
            output_dims=architecture["output_dims"],
            mlp_layers=architecture["hidden_layers"],
            act_fn=activation_cls,
            nerf=architecture["nerf"],
        ).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.selected_checkpoint = selected
        self.architecture = dict(architecture)
        self.architecture["activation"] = activation_name
        self.signed = True

    def pair_distance_and_gradient(
        self, points_xyz: torch.Tensor, q: torch.Tensor
    ):
        """Return one-to-one CDF distance [B] and dq gradient [B,7].

        Row b evaluates f(points_xyz[b], q[b]).  Because each network row is
        independent, autograd of the summed outputs w.r.t. the batched q tensor
        returns the per-pair gradient without constructing a dense Jacobian.
        """
        points_xyz = points_xyz.to(self.device, dtype=torch.float32)
        q = q.to(self.device, dtype=torch.float32).detach().clone()
        q.requires_grad_(True)

        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError("points_xyz must have shape [B,3]")
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError("q must have shape [B,7]")
        if points_xyz.shape[0] == 0:
            raise ValueError("pair batch must be non-empty")
        if points_xyz.shape[0] != q.shape[0]:
            raise ValueError(
                f"pair batch mismatch: points={points_xyz.shape[0]} q={q.shape[0]}"
            )

        inputs = torch.cat((points_xyz, q), dim=-1)
        pred = self.model(inputs).reshape(-1)
        grad = torch.autograd.grad(
            pred.sum(), q, retain_graph=False, create_graph=False
        )[0]
        return pred.detach(), grad.detach()

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

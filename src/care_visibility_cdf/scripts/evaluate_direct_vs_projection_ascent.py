#!/usr/bin/env python3
"""
Unified standalone evaluation for a learned CAREPlanner Visibility CDF.

This script evaluates, on exactly the same sampled (x, q_init) pairs:

1) Learned zero-level projection
   - learned boundary: |f(x, q_proj)| < epsilon_f
   - oracle boundary:  |g(x, q_proj)| < epsilon_g
   - true FOV inside rates: g(x, q_proj) >= threshold

2) Projection followed by local learned-gradient ascent
   - q <- q + alpha * grad_q f / ||grad_q f||
   - no target tau is used by the optimizer
   - several oracle margin thresholds are used only for evaluation

3) Conditional metrics
   - success given that projection reached the learned boundary
   - success given that projection reached the oracle boundary
   - whether learned f increases while oracle g decreases

The script reports three cohorts from the same samples:
   all
   initial_outside:      g(x, q_init) < 0
   initial_far_outside:  g(x, q_init) <= -far_margin

This version has no imports from CAREPlanner training/oracle helper scripts.

Expected repository placement:
    src/care_visibility_cdf/scripts/evaluate_projection_then_ascent.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)


DEFAULT_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "wrist_joint1",
    "wrist_joint2",
    "wrist_joint3",
]

DEFAULT_SENSOR_FRAMES = [
    "link2_sensor1_tof_link",
    "link2_sensor2_tof_link",
    "link3_sensor1_tof_link",
    "link3_sensor2_tof_link",
    "link4_sensor1_tof_link",
    "link4_sensor2_tof_link",
    "EE_sensor1_tof_link",
    "EE_sensor2_tof_link",
]


def _parse_number_list(value, expected_length=None, default=None):
    if value is None:
        values = list(default or [])
    elif isinstance(value, str):
        values = [float(v) for v in value.replace(",", " ").split() if v]
    else:
        values = [float(v) for v in value]
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} values, got {len(values)} from {value!r}."
        )
    return values


def _rpy_rotation_matrix(rpy):
    """URDF fixed-axis RPY rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _make_transform(xyz, rpy):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _rpy_rotation_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return transform


class YimingMLP(nn.Module):
    """Exact architecture used by train_signed_visibility_cdf_pairwise_replace.py."""

    def __init__(
        self,
        in_dim=10,
        out_dim=1,
        activation="relu",
        model_arch="yiming",
        mlp_layers=(1024, 512, 256, 128, 128),
        skips=(),
        nerf=True,
        **_ignored,
    ):
        super().__init__()
        if activation != "relu":
            raise ValueError("Yiming checkpoint requires activation='relu'.")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.model_arch = str(model_arch)
        self.nerf = bool(nerf)
        self.skips = tuple(int(v) for v in skips)
        self.mlp_layers = [int(v) for v in mlp_layers]

        encoded_dim = 3 * self.in_dim if self.nerf else self.in_dim
        self.encoded_dim = encoded_dim

        arrays = []
        layers = list(self.mlp_layers)
        skips_list = list(self.skips)
        if skips_list:
            arrays.append(layers[0 : skips_list[0]])
            arrays[0][-1] -= encoded_dim
            for index in range(1, len(skips_list)):
                arrays.append(layers[skips_list[index - 1] : skips_list[index]])
                arrays[-1][-1] -= encoded_dim
            arrays.append(layers[skips_list[-1] :])
        else:
            arrays.append(layers)

        arrays[-1].append(self.out_dim)
        arrays[0].insert(0, encoded_dim)

        self.layers = nn.ModuleList()
        for channels in arrays[:-1]:
            self.layers.append(self._make_mlp(channels, is_last=False))
        self.layers.append(self._make_mlp(arrays[-1], is_last=True))

    @staticmethod
    def _make_mlp(channels, is_last):
        blocks = []
        if not is_last:
            for index in range(1, len(channels)):
                blocks.append(
                    nn.Sequential(nn.Linear(channels[index - 1], channels[index]), nn.ReLU())
                )
        else:
            for index in range(1, len(channels) - 1):
                blocks.append(
                    nn.Sequential(nn.Linear(channels[index - 1], channels[index]), nn.ReLU())
                )
            blocks.append(nn.Sequential(nn.Linear(channels[-2], channels[-1])))
        return nn.Sequential(*blocks)

    def encode(self, inputs):
        if self.nerf:
            return torch.cat((inputs, torch.sin(inputs), torch.cos(inputs)), dim=-1)
        return inputs

    def forward(self, inputs):
        encoded = self.encode(inputs)
        output = self.layers[0](encoded)
        for layer in self.layers[1:]:
            output = layer(torch.cat((output, encoded), dim=1))
        return output


class StandaloneURDFModel:
    """Minimal URDF joint graph parser needed for sensor forward kinematics."""

    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.joints = []
        self.child_to_joint = {}

        for element in root.findall("joint"):
            name = element.attrib.get("name", "")
            joint_type = element.attrib.get("type", "fixed")
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                raise RuntimeError(f"URDF joint {name!r} lacks parent or child.")

            parent = parent_element.attrib["link"]
            child = child_element.attrib["link"]

            origin_element = element.find("origin")
            if origin_element is None:
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
            else:
                xyz = _parse_number_list(
                    origin_element.attrib.get("xyz"), 3, [0.0, 0.0, 0.0]
                )
                rpy = _parse_number_list(
                    origin_element.attrib.get("rpy"), 3, [0.0, 0.0, 0.0]
                )

            axis_element = element.find("axis")
            axis = _parse_number_list(
                None if axis_element is None else axis_element.attrib.get("xyz"),
                3,
                [1.0, 0.0, 0.0],
            )
            axis_array = np.asarray(axis, dtype=np.float32)
            norm = float(np.linalg.norm(axis_array))
            if norm > 1e-12:
                axis_array /= norm

            spec = {
                "name": name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin_np": _make_transform(xyz, rpy),
                "axis_np": axis_array,
            }
            if child in self.child_to_joint:
                raise RuntimeError(f"URDF link {child!r} has multiple parent joints.")
            self.joints.append(spec)
            self.child_to_joint[child] = spec

    def chain_to(self, base_frame, target_frame):
        chain_reversed = []
        current = target_frame
        visited = set()
        while current != base_frame:
            if current in visited:
                raise RuntimeError(
                    f"Cycle detected while finding URDF chain {base_frame!r} -> {target_frame!r}."
                )
            visited.add(current)
            joint = self.child_to_joint.get(current)
            if joint is None:
                raise RuntimeError(
                    f"Cannot find URDF parent joint for link {current!r} while building "
                    f"chain {base_frame!r} -> {target_frame!r}."
                )
            chain_reversed.append(joint)
            current = joint["parent"]
        return list(reversed(chain_reversed))


class PinocchioFOVOracle:
    """
    Standalone FOV-only oracle reproducing prepare_chain_specs + visibility_g_batch.

    The class name is retained so the rest of the evaluator does not depend on any
    training, data-generation, validation, or self-occlusion script.
    """

    def __init__(
        self,
        urdf_path,
        joint_names,
        sensor_frames,
        horizontal_fov_deg,
        vertical_fov_deg,
        z_min,
        z_max,
        delta,
        base_frame="base_link",
    ):
        self.urdf_path = str(urdf_path)
        self.base_frame = str(base_frame)
        self.joint_names = list(joint_names)
        self.sensor_frames = list(sensor_frames)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.delta = float(delta)

        self.robot = StandaloneURDFModel(self.urdf_path)
        joint_to_index = {name: index for index, name in enumerate(self.joint_names)}
        self.chain_templates = []
        for frame in self.sensor_frames:
            chain = []
            for joint in self.robot.chain_to(self.base_frame, frame):
                chain.append(
                    {
                        "name": joint["name"],
                        "type": joint["type"],
                        "q_index": joint_to_index.get(joint["name"], -1),
                        "origin_np": joint["origin_np"],
                        "axis_np": joint["axis_np"],
                    }
                )
            self.chain_templates.append(chain)
        self._chain_cache = {}

        print("")
        print("=== FOV Oracle ===")
        print("backend:               standalone URDF XML FK + analytic FOV margin")
        print(f"urdf:                  {self.urdf_path}")
        print(f"base_frame:            {self.base_frame}")
        print(f"joint_names:           {self.joint_names}")
        print(f"sensor_frames:         {self.sensor_frames}")
        print(f"horizontal_fov_deg:    {self.horizontal_fov_deg}")
        print(f"vertical_fov_deg:      {self.vertical_fov_deg}")
        print(f"z_min/z_max:           {self.z_min}, {self.z_max}")
        print(f"delta:                 {self.delta}")
        print("==================")
        print("")

    def _chains(self, device, dtype):
        key = (str(device), str(dtype))
        if key not in self._chain_cache:
            converted = []
            for chain in self.chain_templates:
                converted_chain = []
                for joint in chain:
                    converted_chain.append(
                        {
                            "name": joint["name"],
                            "type": joint["type"],
                            "q_index": joint["q_index"],
                            "origin": torch.as_tensor(
                                joint["origin_np"], device=device, dtype=dtype
                            ),
                            "axis": torch.as_tensor(
                                joint["axis_np"], device=device, dtype=dtype
                            ),
                        }
                    )
                converted.append(converted_chain)
            self._chain_cache[key] = converted
        return self._chain_cache[key]

    @staticmethod
    def _batch_eye(batch_size, device, dtype):
        return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(
            batch_size, 1, 1
        )

    @classmethod
    def _revolute_transform(cls, axis, values):
        batch_size = values.shape[0]
        device, dtype = values.device, values.dtype
        x, y, z = axis[0], axis[1], axis[2]
        cosine = torch.cos(values)
        sine = torch.sin(values)
        one_minus_cosine = 1.0 - cosine

        rotation = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
        rotation[:, 0, 0] = cosine + x * x * one_minus_cosine
        rotation[:, 0, 1] = x * y * one_minus_cosine - z * sine
        rotation[:, 0, 2] = x * z * one_minus_cosine + y * sine
        rotation[:, 1, 0] = y * x * one_minus_cosine + z * sine
        rotation[:, 1, 1] = cosine + y * y * one_minus_cosine
        rotation[:, 1, 2] = y * z * one_minus_cosine - x * sine
        rotation[:, 2, 0] = z * x * one_minus_cosine - y * sine
        rotation[:, 2, 1] = z * y * one_minus_cosine + x * sine
        rotation[:, 2, 2] = cosine + z * z * one_minus_cosine

        transform = cls._batch_eye(batch_size, device, dtype)
        transform[:, :3, :3] = rotation
        return transform

    @classmethod
    def _prismatic_transform(cls, axis, values):
        transform = cls._batch_eye(values.shape[0], values.device, values.dtype)
        transform[:, :3, 3] = values.unsqueeze(-1) * axis.unsqueeze(0)
        return transform

    @classmethod
    def _fk_sensor_batch(cls, chain, q_batch):
        batch_size = q_batch.shape[0]
        transform = cls._batch_eye(batch_size, q_batch.device, q_batch.dtype)
        for joint in chain:
            origin = joint["origin"].unsqueeze(0).expand(batch_size, -1, -1)
            transform = torch.bmm(transform, origin)

            q_index = joint["q_index"]
            joint_type = joint["type"]
            if q_index < 0 or joint_type == "fixed":
                continue
            values = q_batch[:, q_index]
            if joint_type in ("revolute", "continuous"):
                motion = cls._revolute_transform(joint["axis"], values)
            elif joint_type == "prismatic":
                motion = cls._prismatic_transform(joint["axis"], values)
            else:
                raise RuntimeError(
                    f"Unsupported movable URDF joint type {joint_type!r} "
                    f"for joint {joint['name']!r}."
                )
            transform = torch.bmm(transform, motion)
        return transform

    @torch.no_grad()
    def signed_fov_margins(self, p_world, q_query):
        if p_world.ndim == 1:
            p_world = p_world.unsqueeze(0)
        if p_world.ndim != 2 or p_world.shape[-1] != 3:
            raise ValueError(f"Expected p_world shape [Bx,3], got {tuple(p_world.shape)}.")
        if q_query.ndim != 2 or q_query.shape[-1] != len(self.joint_names):
            raise ValueError(
                f"Expected q_query shape [Bq,{len(self.joint_names)}], "
                f"got {tuple(q_query.shape)}."
            )

        p_world = p_world.to(device=q_query.device, dtype=q_query.dtype)
        ax = math.tan(math.radians(self.horizontal_fov_deg) * 0.5)
        ay = math.tan(math.radians(self.vertical_fov_deg) * 0.5)
        nx = math.sqrt(1.0 + ax * ax)
        ny = math.sqrt(1.0 + ay * ay)

        sensor_margins = []
        for chain in self._chains(q_query.device, q_query.dtype):
            transform = self._fk_sensor_batch(chain, q_query)
            rotation = transform[:, :3, :3]
            translation = transform[:, :3, 3]

            difference_world = p_world[:, None, :] - translation[None, :, :]
            point_sensor = torch.einsum("qji,bqj->bqi", rotation, difference_world)
            x = point_sensor[:, :, 0]
            y = point_sensor[:, :, 1]
            z = point_sensor[:, :, 2]

            planes = torch.stack(
                [
                    (x + z * ax) / nx,
                    (-x + z * ax) / nx,
                    (y + z * ay) / ny,
                    (-y + z * ay) / ny,
                    z - self.z_min,
                    self.z_max - z,
                ],
                dim=-1,
            )
            margin = torch.min(planes, dim=-1).values
            sensor_margins.append(margin)

        raw_margin = torch.stack(sensor_margins, dim=-1)
        signed_margin = raw_margin - self.delta
        sign = torch.where(
            signed_margin >= 0.0,
            torch.ones_like(signed_margin),
            -torch.ones_like(signed_margin),
        )
        return raw_margin, sign


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_npz_data(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)

    x = d["x"].astype(np.float32)

    if "valid_fov" in d.files:
        valid_fov = d["valid_fov"].astype(np.bool_)
        valid_point = np.any(valid_fov, axis=tuple(range(1, valid_fov.ndim)))
        valid_x = x[valid_point]
        if len(valid_x) == 0:
            print("[WARN] No valid_x found from valid_fov; using all x points.")
            valid_x = x
    else:
        print("[WARN] valid_fov not found; dataset x-source will use all x points.")
        valid_x = x

    if "q_min" in d.files and "q_max" in d.files:
        q_min = d["q_min"].astype(np.float32)
        q_max = d["q_max"].astype(np.float32)
    else:
        q_min = np.full((7,), -math.pi, dtype=np.float32)
        q_max = np.full((7,), math.pi, dtype=np.float32)
        print("[WARN] q_min/q_max not found. Using [-pi, pi].")

    return {
        "x": x,
        "valid_x": valid_x,
        "q_min": q_min,
        "q_max": q_max,
        "x_min": x.min(axis=0),
        "x_max": x.max(axis=0),
    }


def torch_load_checkpoint(path: str, device: torch.device):
    # Explicit weights_only=False avoids behavior changes in newer PyTorch,
    # while the fallback keeps compatibility with older versions.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_checkpoint_args(raw_args) -> Dict:
    if raw_args is None:
        return {}
    if isinstance(raw_args, Mapping):
        return dict(raw_args)
    if hasattr(raw_args, "__dict__"):
        return vars(raw_args)
    raise TypeError(f"Unsupported checkpoint args type: {type(raw_args)}")


def build_model_from_checkpoint(ckpt, device: torch.device):
    """Reconstruct the exact Exp1/Exp4A Yiming MLP without importing training code."""
    ckpt_args = normalize_checkpoint_args(ckpt.get("args", {}))
    model_arch = ckpt_args.get("model_arch", "yiming")
    is_yiming = (
        model_arch == "yiming"
        or "mlp_layers" in ckpt_args
        or bool(ckpt_args.get("nerf", False))
    )
    if not is_yiming:
        raise RuntimeError(
            "This standalone evaluator supports the signed Yiming checkpoints used by "
            "Exp1/Exp4A. The supplied checkpoint does not look like that architecture."
        )

    raw_layers = ckpt_args.get("mlp_layers", "1024,512,256,128,128")
    if isinstance(raw_layers, str):
        mlp_layers = tuple(int(value) for value in raw_layers.split(",") if value.strip())
    elif isinstance(raw_layers, (list, tuple)):
        mlp_layers = tuple(int(value) for value in raw_layers)
    else:
        raise RuntimeError(f"Unsupported mlp_layers type: {type(raw_layers)}")

    raw_skips = ckpt_args.get("skips", "")
    if isinstance(raw_skips, str):
        skips = tuple(int(value) for value in raw_skips.split(",") if value.strip())
    elif isinstance(raw_skips, (list, tuple)):
        skips = tuple(int(value) for value in raw_skips)
    else:
        skips = ()

    model = YimingMLP(
        in_dim=10,
        out_dim=1,
        activation=ckpt_args.get("activation", "relu"),
        model_arch=model_arch,
        mlp_layers=mlp_layers,
        skips=skips,
        nerf=bool(ckpt_args.get("nerf", True)),
    ).to(device)

    if "model_state" not in ckpt:
        raise KeyError("Checkpoint does not contain 'model_state'.")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    first_weight = ckpt["model_state"].get("layers.0.0.0.weight")
    if first_weight is not None and tuple(first_weight.shape) != (mlp_layers[0], 30):
        raise RuntimeError(
            "Checkpoint/model input mismatch: expected first weight shape "
            f"({mlp_layers[0]}, 30), got {tuple(first_weight.shape)}."
        )

    print(f"[model] reconstructed standalone YimingMLP: {mlp_layers}, nerf={model.nerf}")
    return model, ckpt_args


# -----------------------------------------------------------------------------
# Sampling and fields
# -----------------------------------------------------------------------------


def sample_q(
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    batch_q: int,
    device: torch.device,
) -> torch.Tensor:
    u = torch.rand((batch_q, 7), device=device)
    return q_min[None, :] + u * (q_max - q_min)[None, :]


def sample_x(data: Mapping[str, np.ndarray], device: torch.device, x_source: str):
    if x_source == "unit_box":
        # Matches the previous Yiming-style evaluator:
        # x,y in [-0.5, 0.5], z in [0, 1].
        x = torch.rand((1, 3), device=device, dtype=torch.float32)
        offset = torch.tensor(
            [[0.5, 0.5, 0.0]], device=device, dtype=torch.float32
        )
        return x - offset

    if x_source == "dataset":
        valid_x = data["valid_x"]
        idx = np.random.randint(0, len(valid_x))
        return torch.as_tensor(
            valid_x[idx : idx + 1], device=device, dtype=torch.float32
        )

    if x_source == "random_box":
        x_min = data["x_min"]
        x_max = data["x_max"]
        u = np.random.rand(1, 3).astype(np.float32)
        x = x_min[None, :] + u * (x_max - x_min)[None, :]
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    raise ValueError(f"Unknown x_source: {x_source}")


def model_value_and_grad_q(
    x: torch.Tensor,
    q: torch.Tensor,
    model: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return f(x,q), grad_q f(x,q), and the differentiable q tensor."""
    q_grad = q.detach().clone().requires_grad_(True)

    bx = x.shape[0]
    bq = q_grad.shape[0]

    x_cat = x[:, None, :].expand(bx, bq, 3).reshape(bx * bq, 3)
    q_cat = q_grad[None, :, :].expand(bx, bq, 7).reshape(bx * bq, 7)
    inputs = torch.cat([x_cat, q_cat], dim=-1)

    pred = model(inputs).reshape(bx, bq)
    f = pred.min(dim=0).values

    grad = torch.autograd.grad(
        outputs=f,
        inputs=q_grad,
        grad_outputs=torch.ones_like(f),
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
    return f, grad, q_grad


@torch.no_grad()
def model_value(
    x: torch.Tensor,
    q: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    bx = x.shape[0]
    bq = q.shape[0]

    x_cat = x[:, None, :].expand(bx, bq, 3).reshape(bx * bq, 3)
    q_cat = q[None, :, :].expand(bx, bq, 7).reshape(bx * bq, 7)
    inputs = torch.cat([x_cat, q_cat], dim=-1)

    pred = model(inputs).reshape(bx, bq)
    return pred.min(dim=0).values


@torch.no_grad()
def oracle_visibility_g(
    x: torch.Tensor,
    q: torch.Tensor,
    fov_oracle: PinocchioFOVOracle,
) -> torch.Tensor:
    """g(x,q) = max_sensor(raw_sensor_margin - delta)."""
    raw_margin, _ = fov_oracle.signed_fov_margins(x, q)
    g_sensor = raw_margin - fov_oracle.delta
    return torch.max(g_sensor, dim=-1).values.squeeze(0)


# -----------------------------------------------------------------------------
# Updates
# -----------------------------------------------------------------------------


def clamp_configuration(
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return q
    return torch.maximum(torch.minimum(q, q_max[None, :]), q_min[None, :])


def learned_projection_step(
    q: torch.Tensor,
    f: torch.Tensor,
    grad: torch.Tensor,
    damping: float,
    max_step_norm: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Damped normalized-Newton projection to f=0:

        raw_step = f * grad / (||grad||^2 + eps)
        q_new    = q - damping * clip(raw_step)
    """
    grad_norm_sq = torch.sum(grad * grad, dim=-1, keepdim=True)
    raw_step = f.unsqueeze(-1) * grad / torch.clamp(grad_norm_sq, min=eps)
    raw_step_norm = torch.linalg.norm(raw_step, dim=-1, keepdim=True)

    if max_step_norm > 0.0:
        scale = torch.clamp(
            max_step_norm / torch.clamp(raw_step_norm, min=eps), max=1.0
        )
        clipped_step = raw_step * scale
        clipped = raw_step_norm.squeeze(-1) > max_step_norm
    else:
        clipped_step = raw_step
        clipped = torch.zeros_like(raw_step_norm.squeeze(-1), dtype=torch.bool)

    applied_step = damping * clipped_step
    q_new = q - applied_step

    diag = {
        "grad_norm": torch.sqrt(torch.clamp(grad_norm_sq.squeeze(-1), min=eps)),
        "raw_step_norm": raw_step_norm.squeeze(-1),
        "applied_step_norm": torch.linalg.norm(applied_step, dim=-1),
        "clipped": clipped,
    }
    return q_new, diag


def learned_ascent_step(
    q: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    max_step_norm: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Local normalized-gradient ascent with no target tau:

        direction = grad / ||grad||
        q_new = q + step_size * direction
    """
    grad_norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
    direction = grad / torch.clamp(grad_norm, min=eps)
    raw_step = step_size * direction
    raw_step_norm = torch.linalg.norm(raw_step, dim=-1, keepdim=True)

    if max_step_norm > 0.0:
        scale = torch.clamp(
            max_step_norm / torch.clamp(raw_step_norm, min=eps), max=1.0
        )
        applied_step = raw_step * scale
        clipped = raw_step_norm.squeeze(-1) > max_step_norm
    else:
        applied_step = raw_step
        clipped = torch.zeros_like(raw_step_norm.squeeze(-1), dtype=torch.bool)

    q_new = q + applied_step
    diag = {
        "grad_norm": grad_norm.squeeze(-1),
        "raw_step_norm": raw_step_norm.squeeze(-1),
        "applied_step_norm": torch.linalg.norm(applied_step, dim=-1),
        "clipped": clipped,
    }
    return q_new, diag


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def parse_float_list(text: str) -> List[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def parse_int_list(text: str) -> List[int]:
    values = sorted(set(int(v.strip()) for v in text.split(",") if v.strip()))
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("Expected positive integer steps.")
    return values


def threshold_key(value: float) -> str:
    return f"{value:.6g}"


def safe_rate(event: np.ndarray, mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        return float("nan")
    return float(np.mean(event[mask]))


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(values[mask]))


def safe_median(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.median(values[mask]))


def distribution_stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    selected = values[mask]
    if selected.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    qs = np.quantile(selected, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "min": float(np.min(selected)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "p25": float(qs[3]),
        "p50": float(qs[4]),
        "p75": float(qs[5]),
        "p90": float(qs[6]),
        "p95": float(qs[7]),
        "p99": float(qs[8]),
        "max": float(np.max(selected)),
    }


def make_cohorts(init_g: np.ndarray, far_margin: float) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones_like(init_g, dtype=bool),
        "initial_outside": init_g < 0.0,
        "initial_far_outside": init_g <= -far_margin,
    }


def projection_metrics_for_cohort(
    cohort: np.ndarray,
    proj_f: np.ndarray,
    proj_g: np.ndarray,
    epsilon_f: float,
    epsilon_g: float,
    margin_thresholds: Sequence[float],
) -> Dict:
    learned_boundary = np.abs(proj_f) < epsilon_f
    oracle_boundary = np.abs(proj_g) < epsilon_g

    result = {
        "count": int(cohort.sum()),
        "learned_boundary_rate": safe_rate(learned_boundary, cohort),
        "oracle_boundary_rate": safe_rate(oracle_boundary, cohort),
        "both_boundaries_rate": safe_rate(
            learned_boundary & oracle_boundary, cohort
        ),
        "oracle_boundary_given_learned_boundary": safe_rate(
            oracle_boundary, cohort & learned_boundary
        ),
        "learned_boundary_count": int((cohort & learned_boundary).sum()),
        "oracle_boundary_count": int((cohort & oracle_boundary).sum()),
        "proj_f_stats": distribution_stats(proj_f, cohort),
        "proj_g_stats": distribution_stats(proj_g, cohort),
        "oracle_g_given_learned_boundary_stats": distribution_stats(
            proj_g, cohort & learned_boundary
        ),
        "abs_oracle_g_given_learned_boundary_stats": distribution_stats(
            np.abs(proj_g), cohort & learned_boundary
        ),
    }

    inside = {}
    inside_given_f_boundary = {}
    inside_given_g_boundary = {}
    for threshold in margin_thresholds:
        key = threshold_key(threshold)
        event = proj_g >= threshold
        inside[key] = safe_rate(event, cohort)
        inside_given_f_boundary[key] = safe_rate(
            event, cohort & learned_boundary
        )
        inside_given_g_boundary[key] = safe_rate(
            event, cohort & oracle_boundary
        )

    result["oracle_inside_rate"] = inside
    result["oracle_inside_given_learned_boundary"] = inside_given_f_boundary
    result["oracle_inside_given_oracle_boundary"] = inside_given_g_boundary
    return result


def ascent_metrics_for_cohort(
    cohort: np.ndarray,
    proj_f: np.ndarray,
    proj_g: np.ndarray,
    current_f: np.ndarray,
    current_g: np.ndarray,
    epsilon_f: float,
    epsilon_g: float,
    margin_thresholds: Sequence[float],
) -> Dict:
    learned_boundary_at_projection = np.abs(proj_f) < epsilon_f
    oracle_boundary_at_projection = np.abs(proj_g) < epsilon_g

    delta_f = current_f - proj_f
    delta_g = current_g - proj_g

    f_increased = delta_f > 0.0
    g_increased = delta_g > 0.0
    exploitation = f_increased & (delta_g < 0.0)

    result = {
        "count": int(cohort.sum()),
        "current_f_stats": distribution_stats(current_f, cohort),
        "current_g_stats": distribution_stats(current_g, cohort),
        "delta_f_stats": distribution_stats(delta_f, cohort),
        "delta_g_stats": distribution_stats(delta_g, cohort),
        "delta_f_positive_rate": safe_rate(f_increased, cohort),
        "delta_g_positive_rate": safe_rate(g_increased, cohort),
        "learned_up_oracle_down_rate": safe_rate(exploitation, cohort),
        "delta_g_positive_given_learned_boundary": safe_rate(
            g_increased, cohort & learned_boundary_at_projection
        ),
        "delta_g_positive_given_oracle_boundary": safe_rate(
            g_increased, cohort & oracle_boundary_at_projection
        ),
        "delta_g_mean_given_learned_boundary": safe_mean(
            delta_g, cohort & learned_boundary_at_projection
        ),
        "delta_g_mean_given_oracle_boundary": safe_mean(
            delta_g, cohort & oracle_boundary_at_projection
        ),
        "delta_g_median_given_learned_boundary": safe_median(
            delta_g, cohort & learned_boundary_at_projection
        ),
        "delta_g_median_given_oracle_boundary": safe_median(
            delta_g, cohort & oracle_boundary_at_projection
        ),
        "crossed_true_boundary_rate_given_projection_outside": safe_rate(
            current_g >= 0.0, cohort & (proj_g < 0.0)
        ),
        "projection_outside_count": int((cohort & (proj_g < 0.0)).sum()),
    }

    inside = {}
    inside_given_f_boundary = {}
    inside_given_g_boundary = {}
    crossed_threshold = {}

    for threshold in margin_thresholds:
        key = threshold_key(threshold)
        event = current_g >= threshold
        inside[key] = safe_rate(event, cohort)
        inside_given_f_boundary[key] = safe_rate(
            event, cohort & learned_boundary_at_projection
        )
        inside_given_g_boundary[key] = safe_rate(
            event, cohort & oracle_boundary_at_projection
        )
        crossed_threshold[key] = safe_rate(
            event, cohort & (proj_g < threshold)
        )

    result["oracle_inside_rate"] = inside
    result["oracle_inside_given_projection_learned_boundary"] = (
        inside_given_f_boundary
    )
    result["oracle_inside_given_projection_oracle_boundary"] = (
        inside_given_g_boundary
    )
    result["crossed_threshold_rate_given_projection_below_threshold"] = (
        crossed_threshold
    )
    return result


def format_rate(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------




def clamp_with_event(
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clamp q and return one boolean per sample indicating any joint was clipped."""
    if not enabled:
        return q, torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
    q_clamped = torch.maximum(torch.minimum(q, q_max[None, :]), q_min[None, :])
    clipped = torch.any(torch.abs(q_clamped - q) > 1e-10, dim=-1)
    return q_clamped, clipped


def final_limit_contact(
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    near_min = q <= (q_min[None, :] + tolerance)
    near_max = q >= (q_max[None, :] - tolerance)
    return torch.any(near_min | near_max, dim=-1)


def boundary_sweep_metrics(
    cohort: np.ndarray,
    proj_f: np.ndarray,
    proj_g: np.ndarray,
    tolerances: Sequence[float],
    default_epsilon_f: float,
) -> Dict:
    default_f_boundary = np.abs(proj_f) < default_epsilon_f
    result = {"count": int(cohort.sum()), "by_tolerance": {}}
    for tolerance in tolerances:
        key = threshold_key(tolerance)
        learned_boundary = np.abs(proj_f) < tolerance
        oracle_boundary = np.abs(proj_g) < tolerance
        result["by_tolerance"][key] = {
            "learned_boundary_rate": safe_rate(learned_boundary, cohort),
            "oracle_boundary_rate": safe_rate(oracle_boundary, cohort),
            "both_boundaries_rate": safe_rate(
                learned_boundary & oracle_boundary, cohort
            ),
            "oracle_boundary_given_same_tolerance_learned_boundary": safe_rate(
                oracle_boundary, cohort & learned_boundary
            ),
            "oracle_boundary_given_default_learned_boundary": safe_rate(
                oracle_boundary, cohort & default_f_boundary
            ),
        }
    result["proj_f_stats"] = distribution_stats(proj_f, cohort)
    result["proj_g_stats"] = distribution_stats(proj_g, cohort)
    result["abs_proj_g_stats"] = distribution_stats(np.abs(proj_g), cohort)
    return result


def method_metrics_for_cohort(
    cohort: np.ndarray,
    init_g: np.ndarray,
    final_f: np.ndarray,
    final_g: np.ndarray,
    endpoint_distance: np.ndarray,
    path_length: np.ndarray,
    max_actual_step: np.ndarray,
    clamp_any: np.ndarray,
    final_limit_contact_mask: np.ndarray,
    margin_thresholds: Sequence[float],
    reference_g: np.ndarray | None = None,
) -> Dict:
    delta_g_init = final_g - init_g
    denom = np.maximum(path_length, 1e-8)
    gain_per_path = delta_g_init / denom

    result = {
        "count": int(cohort.sum()),
        "final_f_stats": distribution_stats(final_f, cohort),
        "final_g_stats": distribution_stats(final_g, cohort),
        "delta_g_from_init_stats": distribution_stats(delta_g_init, cohort),
        "delta_g_from_init_positive_rate": safe_rate(delta_g_init > 0.0, cohort),
        "endpoint_distance_stats": distribution_stats(endpoint_distance, cohort),
        "path_length_stats": distribution_stats(path_length, cohort),
        "max_actual_step_stats": distribution_stats(max_actual_step, cohort),
        "oracle_gain_per_path_stats": distribution_stats(gain_per_path, cohort),
        "clamp_any_rate": safe_rate(clamp_any, cohort),
        "final_joint_limit_contact_rate": safe_rate(
            final_limit_contact_mask, cohort
        ),
        "oracle_inside_rate": {},
        "path_length_given_success": {},
        "endpoint_distance_given_success": {},
    }
    if reference_g is not None:
        delta_reference = final_g - reference_g
        result["delta_g_from_reference_stats"] = distribution_stats(
            delta_reference, cohort
        )
        result["delta_g_from_reference_positive_rate"] = safe_rate(
            delta_reference > 0.0, cohort
        )

    for threshold in margin_thresholds:
        key = threshold_key(threshold)
        success = final_g >= threshold
        result["oracle_inside_rate"][key] = safe_rate(success, cohort)
        result["path_length_given_success"][key] = safe_mean(
            path_length, cohort & success
        )
        result["endpoint_distance_given_success"][key] = safe_mean(
            endpoint_distance, cohort & success
        )
    return result


def comparison_metrics_for_cohort(
    cohort: np.ndarray,
    direct_g: np.ndarray,
    hybrid_g: np.ndarray,
    direct_path: np.ndarray,
    hybrid_path: np.ndarray,
    margin_thresholds: Sequence[float],
) -> Dict:
    difference = hybrid_g - direct_g
    result = {
        "count": int(cohort.sum()),
        "hybrid_minus_direct_g_stats": distribution_stats(difference, cohort),
        "hybrid_greater_rate": safe_rate(difference > 1e-8, cohort),
        "direct_greater_rate": safe_rate(difference < -1e-8, cohort),
        "approximately_equal_rate": safe_rate(np.abs(difference) <= 1e-8, cohort),
        "hybrid_minus_direct_path_stats": distribution_stats(
            hybrid_path - direct_path, cohort
        ),
        "by_threshold": {},
    }
    for threshold in margin_thresholds:
        key = threshold_key(threshold)
        direct_success = direct_g >= threshold
        hybrid_success = hybrid_g >= threshold
        result["by_threshold"][key] = {
            "direct_success_rate": safe_rate(direct_success, cohort),
            "hybrid_success_rate": safe_rate(hybrid_success, cohort),
            "hybrid_only_success_rate": safe_rate(
                hybrid_success & ~direct_success, cohort
            ),
            "direct_only_success_rate": safe_rate(
                direct_success & ~hybrid_success, cohort
            ),
            "both_success_rate": safe_rate(
                hybrid_success & direct_success, cohort
            ),
            "neither_success_rate": safe_rate(
                ~hybrid_success & ~direct_success, cohort
            ),
        }
    return result


def _append(chunks: List[np.ndarray], tensor: torch.Tensor, dtype=np.float32):
    chunks.append(tensor.detach().cpu().numpy().astype(dtype))


def evaluate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_npz_data(args.data)
    ckpt = torch_load_checkpoint(args.checkpoint, device)
    model, ckpt_args = build_model_from_checkpoint(ckpt, device)

    q_min = torch.as_tensor(data["q_min"], device=device, dtype=torch.float32)
    q_max = torch.as_tensor(data["q_max"], device=device, dtype=torch.float32)

    horizontal_fov_deg = float(
        ckpt_args.get("horizontal_fov_deg", args.horizontal_fov_deg)
    )
    vertical_fov_deg = float(
        ckpt_args.get("vertical_fov_deg", args.vertical_fov_deg)
    )
    z_min = float(ckpt_args.get("z_min", args.z_min))
    z_max = float(ckpt_args.get("z_max", args.z_max))
    delta = float(ckpt_args.get("delta", args.delta))

    fov_oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
        z_min=z_min,
        z_max=z_max,
        delta=delta,
    )

    ascent_steps = args.ascent_steps
    max_ascent_steps = max(ascent_steps)
    margin_thresholds = sorted(set(args.margin_thresholds))
    boundary_tolerances = sorted(set(args.boundary_tolerances))

    print("\n=== Direct Ascent vs Projection -> Ascent Evaluation ===")
    print(f"checkpoint:              {args.checkpoint}")
    print(f"data:                    {args.data}")
    print(f"urdf:                    {args.urdf}")
    print(f"device:                  {device}")
    print(f"seed:                    {args.seed}")
    print(f"x_source:                {args.x_source}")
    print(f"num_trials / batch_q:    {args.num_trials} / {args.batch_q}")
    print(f"projection iters:        {args.projection_iters}")
    print(f"projection damping:      {args.projection_damping}")
    print(f"projection max step:     {args.projection_max_step_norm}")
    print(f"ascent step size:        {args.ascent_step_size}")
    print(f"ascent snapshots:        {ascent_steps}")
    print(f"boundary tolerances:     {boundary_tolerances}")
    print(f"oracle thresholds:       {margin_thresholds}")
    print(f"joint limit tolerance:   {args.joint_limit_tolerance}")
    print(f"clamp_q:                 {args.clamp_q}")
    print("===========================================================\n")

    base_names = ["init_f", "init_g", "proj_f", "proj_g"]
    chunks = {name: [] for name in base_names}
    projection_movement = {
        "endpoint": [], "path": [], "max_step": [], "clamp_any": [],
        "limit_contact": [],
    }
    methods = {"direct": {}, "hybrid": {}}
    for method in methods:
        for step in ascent_steps:
            methods[method][step] = {
                "f": [], "g": [], "endpoint": [], "path": [],
                "max_step": [], "clamp_any": [], "limit_contact": [],
            }

    diag_sums = {
        "projection": {"grad": 0.0, "raw": 0.0, "applied": 0.0,
                       "algorithm_clip": 0.0, "joint_clip": 0.0, "count": 0},
        "direct": {"grad": 0.0, "raw": 0.0, "applied": 0.0,
                   "algorithm_clip": 0.0, "joint_clip": 0.0, "count": 0},
        "hybrid": {"grad": 0.0, "raw": 0.0, "applied": 0.0,
                   "algorithm_clip": 0.0, "joint_clip": 0.0, "count": 0},
    }

    def accumulate_diag(name, diag, joint_clip):
        n = diag["grad_norm"].numel()
        d = diag_sums[name]
        d["grad"] += float(diag["grad_norm"].sum().item())
        d["raw"] += float(diag["raw_step_norm"].sum().item())
        d["applied"] += float(diag["applied_step_norm"].sum().item())
        d["algorithm_clip"] += float(diag["clipped"].sum().item())
        d["joint_clip"] += float(joint_clip.sum().item())
        d["count"] += int(n)

    t_start = time.time()

    for trial in range(args.num_trials):
        x = sample_x(data, device=device, x_source=args.x_source)
        q_init = sample_q(q_min, q_max, args.batch_q, device=device)
        init_f = model_value(x, q_init, model)
        init_g = oracle_visibility_g(x, q_init, fov_oracle)

        # Stage 1: learned zero-level projection.
        q_proj = q_init.detach().clone()
        proj_path = torch.zeros(args.batch_q, device=device)
        proj_max_step = torch.zeros(args.batch_q, device=device)
        proj_clamp_any = torch.zeros(args.batch_q, device=device, dtype=torch.bool)
        for _ in range(args.projection_iters):
            q_prev = q_proj
            f, grad, q_grad = model_value_and_grad_q(x, q_prev, model)
            q_raw, diag = learned_projection_step(
                q=q_grad,
                f=f,
                grad=grad,
                damping=args.projection_damping,
                max_step_norm=args.projection_max_step_norm,
            )
            q_proj, joint_clip = clamp_with_event(
                q_raw.detach(), q_min, q_max, args.clamp_q
            )
            actual_step = torch.linalg.norm(q_proj - q_prev, dim=-1)
            proj_path += actual_step
            proj_max_step = torch.maximum(proj_max_step, actual_step)
            proj_clamp_any |= joint_clip
            accumulate_diag("projection", diag, joint_clip)

        proj_f = model_value(x, q_proj, model)
        proj_g = oracle_visibility_g(x, q_proj, fov_oracle)
        proj_endpoint = torch.linalg.norm(q_proj - q_init, dim=-1)
        proj_limit_contact = final_limit_contact(
            q_proj, q_min, q_max, args.joint_limit_tolerance
        )

        _append(chunks["init_f"], init_f)
        _append(chunks["init_g"], init_g)
        _append(chunks["proj_f"], proj_f)
        _append(chunks["proj_g"], proj_g)
        _append(projection_movement["endpoint"], proj_endpoint)
        _append(projection_movement["path"], proj_path)
        _append(projection_movement["max_step"], proj_max_step)
        _append(projection_movement["clamp_any"], proj_clamp_any, np.bool_)
        _append(projection_movement["limit_contact"], proj_limit_contact, np.bool_)

        # Run direct and two-stage learned ascent from exactly the same samples.
        q_direct = q_init.detach().clone()
        q_hybrid = q_proj.detach().clone()
        direct_path = torch.zeros(args.batch_q, device=device)
        direct_max_step = torch.zeros(args.batch_q, device=device)
        direct_clamp_any = torch.zeros(args.batch_q, device=device, dtype=torch.bool)
        hybrid_ascent_path = torch.zeros(args.batch_q, device=device)
        hybrid_max_step = proj_max_step.clone()
        hybrid_clamp_any = proj_clamp_any.clone()

        for step in range(1, max_ascent_steps + 1):
            # Direct learned ascent from q_init.
            q_prev = q_direct
            _, grad, q_grad = model_value_and_grad_q(x, q_prev, model)
            q_raw, diag = learned_ascent_step(
                q=q_grad,
                grad=grad,
                step_size=args.ascent_step_size,
                max_step_norm=args.ascent_max_step_norm,
            )
            q_direct, joint_clip = clamp_with_event(
                q_raw.detach(), q_min, q_max, args.clamp_q
            )
            actual_step = torch.linalg.norm(q_direct - q_prev, dim=-1)
            direct_path += actual_step
            direct_max_step = torch.maximum(direct_max_step, actual_step)
            direct_clamp_any |= joint_clip
            accumulate_diag("direct", diag, joint_clip)

            # Projection followed by learned local ascent.
            q_prev = q_hybrid
            _, grad, q_grad = model_value_and_grad_q(x, q_prev, model)
            q_raw, diag = learned_ascent_step(
                q=q_grad,
                grad=grad,
                step_size=args.ascent_step_size,
                max_step_norm=args.ascent_max_step_norm,
            )
            q_hybrid, joint_clip = clamp_with_event(
                q_raw.detach(), q_min, q_max, args.clamp_q
            )
            actual_step = torch.linalg.norm(q_hybrid - q_prev, dim=-1)
            hybrid_ascent_path += actual_step
            hybrid_max_step = torch.maximum(hybrid_max_step, actual_step)
            hybrid_clamp_any |= joint_clip
            accumulate_diag("hybrid", diag, joint_clip)

            if step in ascent_steps:
                direct_f = model_value(x, q_direct, model)
                direct_g = oracle_visibility_g(x, q_direct, fov_oracle)
                hybrid_f = model_value(x, q_hybrid, model)
                hybrid_g = oracle_visibility_g(x, q_hybrid, fov_oracle)

                direct_endpoint = torch.linalg.norm(q_direct - q_init, dim=-1)
                hybrid_endpoint = torch.linalg.norm(q_hybrid - q_init, dim=-1)
                hybrid_total_path = proj_path + hybrid_ascent_path
                direct_limit_contact = final_limit_contact(
                    q_direct, q_min, q_max, args.joint_limit_tolerance
                )
                hybrid_limit_contact = final_limit_contact(
                    q_hybrid, q_min, q_max, args.joint_limit_tolerance
                )

                vals = {
                    "direct": {
                        "f": direct_f, "g": direct_g,
                        "endpoint": direct_endpoint, "path": direct_path,
                        "max_step": direct_max_step, "clamp_any": direct_clamp_any,
                        "limit_contact": direct_limit_contact,
                    },
                    "hybrid": {
                        "f": hybrid_f, "g": hybrid_g,
                        "endpoint": hybrid_endpoint, "path": hybrid_total_path,
                        "max_step": hybrid_max_step, "clamp_any": hybrid_clamp_any,
                        "limit_contact": hybrid_limit_contact,
                    },
                }
                for method, method_values in vals.items():
                    for key, tensor in method_values.items():
                        dtype = np.bool_ if key in {"clamp_any", "limit_contact"} else np.float32
                        _append(methods[method][step][key], tensor, dtype)

        if args.print_every > 0 and (
            trial == 0 or (trial + 1) % args.print_every == 0
        ):
            elapsed = time.time() - t_start
            print(
                f"trial {trial + 1:04d}/{args.num_trials} "
                f"init_inside={(init_g >= 0).float().mean().item():.4f} "
                f"proj_g<0.01={(torch.abs(proj_g) < 0.01).float().mean().item():.4f} "
                f"proj_g<0.03={(torch.abs(proj_g) < 0.03).float().mean().item():.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    projection_arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in projection_movement.items()
    }
    method_arrays = {"direct": {}, "hybrid": {}}
    for method in method_arrays:
        for step in ascent_steps:
            method_arrays[method][step] = {
                key: np.concatenate(value, axis=0)
                for key, value in methods[method][step].items()
            }

    init_g = arrays["init_g"]
    proj_f = arrays["proj_f"]
    proj_g = arrays["proj_g"]
    cohorts = make_cohorts(init_g, args.far_margin)

    boundary_result = {
        name: boundary_sweep_metrics(
            mask, proj_f, proj_g, boundary_tolerances, args.epsilon_f
        )
        for name, mask in cohorts.items()
    }

    projection_movement_result = {
        name: {
            "count": int(mask.sum()),
            "endpoint_distance_stats": distribution_stats(
                projection_arrays["endpoint"], mask
            ),
            "path_length_stats": distribution_stats(
                projection_arrays["path"], mask
            ),
            "max_actual_step_stats": distribution_stats(
                projection_arrays["max_step"], mask
            ),
            "clamp_any_rate": safe_rate(
                projection_arrays["clamp_any"], mask
            ),
            "final_joint_limit_contact_rate": safe_rate(
                projection_arrays["limit_contact"], mask
            ),
        }
        for name, mask in cohorts.items()
    }

    result_methods = {"direct": {}, "hybrid": {}}
    comparison = {}
    for step in ascent_steps:
        comparison[str(step)] = {}
        for method in result_methods:
            result_methods[method][str(step)] = {}
            values = method_arrays[method][step]
            for name, mask in cohorts.items():
                reference = proj_g if method == "hybrid" else None
                result_methods[method][str(step)][name] = method_metrics_for_cohort(
                    cohort=mask,
                    init_g=init_g,
                    final_f=values["f"],
                    final_g=values["g"],
                    endpoint_distance=values["endpoint"],
                    path_length=values["path"],
                    max_actual_step=values["max_step"],
                    clamp_any=values["clamp_any"],
                    final_limit_contact_mask=values["limit_contact"],
                    margin_thresholds=margin_thresholds,
                    reference_g=reference,
                )
        for name, mask in cohorts.items():
            direct = method_arrays["direct"][step]
            hybrid = method_arrays["hybrid"][step]
            comparison[str(step)][name] = comparison_metrics_for_cohort(
                cohort=mask,
                direct_g=direct["g"],
                hybrid_g=hybrid["g"],
                direct_path=direct["path"],
                hybrid_path=hybrid["path"],
                margin_thresholds=margin_thresholds,
            )

    def finalize_diag(d):
        count = max(int(d["count"]), 1)
        return {
            "grad_norm_mean": d["grad"] / count,
            "raw_step_norm_mean": d["raw"] / count,
            "applied_step_norm_mean_before_joint_clamp": d["applied"] / count,
            "algorithm_step_clip_rate": d["algorithm_clip"] / count,
            "joint_limit_clamp_rate_per_update": d["joint_clip"] / count,
            "sample_updates": int(d["count"]),
        }

    elapsed_sec = time.time() - t_start
    result = {
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_step": ckpt.get("step", None),
            "checkpoint_best_val": ckpt.get("best_val", None),
            "data": args.data,
            "urdf": args.urdf,
            "device": str(device),
            "seed": args.seed,
            "x_source": args.x_source,
            "num_trials": args.num_trials,
            "batch_q": args.batch_q,
            "total_samples": int(init_g.size),
            "projection_iters": args.projection_iters,
            "projection_damping": args.projection_damping,
            "projection_max_step_norm": args.projection_max_step_norm,
            "epsilon_f": args.epsilon_f,
            "boundary_tolerances": boundary_tolerances,
            "ascent_step_size": args.ascent_step_size,
            "ascent_max_step_norm": args.ascent_max_step_norm,
            "ascent_steps": ascent_steps,
            "margin_thresholds": margin_thresholds,
            "far_margin": args.far_margin,
            "clamp_q": args.clamp_q,
            "joint_limit_tolerance": args.joint_limit_tolerance,
            "horizontal_fov_deg": horizontal_fov_deg,
            "vertical_fov_deg": vertical_fov_deg,
            "z_min": z_min,
            "z_max": z_max,
            "delta": delta,
        },
        "initial": {
            name: {
                "count": int(mask.sum()),
                "f_stats": distribution_stats(arrays["init_f"], mask),
                "g_stats": distribution_stats(init_g, mask),
                "oracle_inside_rate": safe_rate(init_g >= 0.0, mask),
            }
            for name, mask in cohorts.items()
        },
        "projection_boundary_sweep": boundary_result,
        "projection_movement": projection_movement_result,
        "methods": result_methods,
        "comparison": comparison,
        "diagnostics": {
            name: finalize_diag(value) for name, value in diag_sums.items()
        },
        "elapsed_sec": elapsed_sec,
    }

    print("\n=== Strict projection boundary sweep: initial_outside ===")
    print("tol      learned      oracle       oracle|learned(default eps_f)")
    for tolerance in boundary_tolerances:
        key = threshold_key(tolerance)
        r = boundary_result["initial_outside"]["by_tolerance"][key]
        print(
            f"{tolerance:<8g} "
            f"{format_rate(r['learned_boundary_rate']):>10s} "
            f"{format_rate(r['oracle_boundary_rate']):>12s} "
            f"{format_rate(r['oracle_boundary_given_default_learned_boundary']):>29s}"
        )

    print("\n=== Direct vs hybrid: initial_outside ===")
    header = "step method   path_mean endpoint_mean"
    for threshold in margin_thresholds:
        header += f"   g>={threshold_key(threshold)}"
    print(header)
    for step in ascent_steps:
        for method in ["direct", "hybrid"]:
            r = result_methods[method][str(step)]["initial_outside"]
            row = (
                f"{step:4d} {method:7s} "
                f"{r['path_length_stats']['mean']:9.4f} "
                f"{r['endpoint_distance_stats']['mean']:13.4f}"
            )
            for threshold in margin_thresholds:
                row += f"   {format_rate(r['oracle_inside_rate'][threshold_key(threshold)]):>8s}"
            print(row)

    print("\n=== Hybrid advantage: initial_outside ===")
    print("step  mean(g_hybrid-g_direct)  hybrid_only(g>=0) direct_only(g>=0)")
    zero_key = threshold_key(0.0)
    for step in ascent_steps:
        r = comparison[str(step)]["initial_outside"]
        t = r["by_threshold"][zero_key]
        print(
            f"{step:4d} {r['hybrid_minus_direct_g_stats']['mean']:24.5f} "
            f"{format_rate(t['hybrid_only_success_rate']):>18s} "
            f"{format_rate(t['direct_only_success_rate']):>17s}"
        )

    print("\n=== Diagnostics ===")
    print(json.dumps(result["diagnostics"], indent=2))
    print(f"elapsed_sec: {elapsed_sec:.2f}")

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=True)
        print(f"saved JSON: {args.output}")

    if args.save_samples:
        sample_dir = os.path.dirname(os.path.abspath(args.save_samples))
        os.makedirs(sample_dir, exist_ok=True)
        save = {
            "init_f": arrays["init_f"],
            "init_g": init_g,
            "proj_f": proj_f,
            "proj_g": proj_g,
            "projection_endpoint": projection_arrays["endpoint"],
            "projection_path": projection_arrays["path"],
        }
        for method in ["direct", "hybrid"]:
            for step in ascent_steps:
                values = method_arrays[method][step]
                for key in ["f", "g", "endpoint", "path", "max_step", "clamp_any", "limit_contact"]:
                    save[f"{method}_step_{step}_{key}"] = values[key]
        np.savez_compressed(args.save_samples, **save)
        print(f"saved samples: {args.save_samples}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct learned ascent against learned zero-level projection "
            "followed by local ascent on identical samples, with strict boundary "
            "tolerances and joint-space movement costs."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "exp1_yiming_k500_fov_signed/final.pt"
        ),
    )
    parser.add_argument(
        "--data",
        default=(
            "src/care_visibility_cdf/data/"
            "visibility_yiming_style_grid30_q20000_k500_fovonly.npz"
        ),
    )
    parser.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--batch-q", type=int, default=1000)
    parser.add_argument(
        "--x-source", choices=["unit_box", "random_box", "dataset"],
        default="unit_box"
    )
    parser.add_argument("--projection-iters", type=int, default=10)
    parser.add_argument("--projection-damping", type=float, default=0.5)
    parser.add_argument("--projection-max-step-norm", type=float, default=0.25)
    parser.add_argument(
        "--epsilon-f", type=float, default=0.03,
        help="Default learned-boundary tolerance for conditional metrics."
    )
    parser.add_argument(
        "--boundary-tolerances", type=parse_float_list,
        default=parse_float_list("0.005,0.01,0.02,0.03"),
        help="Absolute f/g tolerances for the strict projection-boundary sweep."
    )
    parser.add_argument("--ascent-step-size", type=float, default=0.05)
    parser.add_argument("--ascent-max-step-norm", type=float, default=0.25)
    parser.add_argument(
        "--ascent-steps", type=parse_int_list,
        default=parse_int_list("1,2,3,5,10")
    )
    parser.add_argument(
        "--margin-thresholds", type=parse_float_list,
        default=parse_float_list("0,0.005,0.01,0.03")
    )
    parser.add_argument("--far-margin", type=float, default=0.03)
    parser.add_argument(
        "--clamp-q", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--joint-limit-tolerance", type=float, default=1e-5,
        help="A final state is considered at a joint limit within this tolerance."
    )
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=66.0)
    parser.add_argument("--z-min", type=float, default=0.20)
    parser.add_argument("--z-max", type=float, default=0.70)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--output",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "exp1_yiming_k500_fov_signed/"
            "direct_vs_projection_ascent.json"
        ),
    )
    parser.add_argument(
        "--save-samples", default="",
        help="Optional compressed .npz with per-sample values and movement costs."
    )
    args = parser.parse_args()
    if args.num_trials <= 0 or args.batch_q <= 0:
        parser.error("num-trials and batch-q must be positive.")
    if args.projection_iters < 0:
        parser.error("projection-iters must be non-negative.")
    if args.epsilon_f <= 0:
        parser.error("epsilon-f must be positive.")
    if any(v <= 0 for v in args.boundary_tolerances):
        parser.error("all boundary-tolerances must be positive.")
    if args.ascent_step_size <= 0:
        parser.error("ascent-step-size must be positive.")
    if args.ascent_max_step_norm < 0:
        parser.error("ascent-max-step-norm must be non-negative.")
    if args.joint_limit_tolerance < 0:
        parser.error("joint-limit-tolerance must be non-negative.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from urdf_parser_py.urdf import URDF


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    find_chain_joints,
    get_joint_limits,
    make_transform,
)
from check_visibility_self_occlusion import load_collision_primitives  # noqa: E402
from generate_visibility_raw_samples import sample_q_batch  # noqa: E402


DEFAULT_P_MIN = [-0.95, -0.95, 0.0]
DEFAULT_P_MAX = [0.95, 0.95, 1.15]
DEFAULT_Q_MIN = [-3.14, -2.3, -3.14, -3.14, -3.14, -3.14, -1.2]
DEFAULT_Q_MAX = [3.14, 2.3, 3.14, 3.14, 3.14, 3.14, 1.2]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        if activation == "relu":
            act = nn.ReLU
        elif activation == "softplus":
            act = lambda: nn.Softplus(beta=10.0)
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = [nn.Linear(input_dim, hidden_dim), act()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def names_from_npz(data, key, default):
    if key in data:
        return [str(x) for x in data[key]]
    return list(default)


def scalar_from_npz(data, key, default):
    if key in data:
        return float(np.asarray(data[key]).item())
    return default


def torch_tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(array, device=device, dtype=dtype)


def origin_np(origin):
    if origin is None:
        return np.eye(4, dtype=np.float32)
    xyz = origin.xyz if origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = origin.rpy if origin.rpy is not None else [0.0, 0.0, 0.0]
    return make_transform(xyz, rpy).astype(np.float32)


def joint_origin_np(joint):
    return origin_np(joint.origin)


def prepare_chain_specs(robot, base_frame, target_frames, joint_names, device):
    joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    all_specs = []
    for frame in target_frames:
        chain = find_chain_joints(robot, base_frame, frame)
        specs = []
        for joint in chain:
            axis = joint.axis if joint.axis is not None else [1.0, 0.0, 0.0]
            axis = np.asarray(axis, dtype=np.float32)
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axis = axis / norm
            specs.append({
                "name": joint.name,
                "type": joint.type,
                "q_index": joint_name_to_index.get(joint.name, -1),
                "origin": torch_tensor(joint_origin_np(joint), device),
                "axis": torch_tensor(axis, device),
            })
        all_specs.append(specs)
    return all_specs


def batch_eye(batch_size, device, dtype=torch.float32):
    return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)


def batch_revolute_transform(axis, q):
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(q)
    s = torch.sin(q)
    one_minus_c = 1.0 - c

    rot = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
    rot[:, 0, 0] = c + x * x * one_minus_c
    rot[:, 0, 1] = x * y * one_minus_c - z * s
    rot[:, 0, 2] = x * z * one_minus_c + y * s
    rot[:, 1, 0] = y * x * one_minus_c + z * s
    rot[:, 1, 1] = c + y * y * one_minus_c
    rot[:, 1, 2] = y * z * one_minus_c - x * s
    rot[:, 2, 0] = z * x * one_minus_c - y * s
    rot[:, 2, 1] = z * y * one_minus_c + x * s
    rot[:, 2, 2] = c + z * z * one_minus_c

    transform = batch_eye(batch_size, device, dtype)
    transform[:, :3, :3] = rot
    return transform


def batch_prismatic_transform(axis, q):
    batch_size = q.shape[0]
    transform = batch_eye(batch_size, q.device, q.dtype)
    transform[:, :3, 3] = q.unsqueeze(-1) * axis.unsqueeze(0)
    return transform


def fk_batch(chain_specs, q_batch):
    batch_size = q_batch.shape[0]
    transform = batch_eye(batch_size, q_batch.device, q_batch.dtype)
    for spec in chain_specs:
        origin = spec["origin"].to(dtype=q_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        transform = torch.bmm(transform, origin)
        if spec["q_index"] < 0 or spec["type"] == "fixed":
            continue
        q = q_batch[:, spec["q_index"]]
        if spec["type"] in ["revolute", "continuous"]:
            transform = torch.bmm(transform, batch_revolute_transform(spec["axis"], q))
        elif spec["type"] == "prismatic":
            transform = torch.bmm(transform, batch_prismatic_transform(spec["axis"], q))
    return transform


def visibility_margins_batch(point, q_batch, sensor_chain_specs, h_fov, v_fov, z_min, z_max, delta):
    batch_size = q_batch.shape[0]
    point = point.to(device=q_batch.device, dtype=q_batch.dtype).reshape(1, 3).expand(batch_size, 3)

    ax = math.tan(math.radians(h_fov) * 0.5)
    ay = math.tan(math.radians(v_fov) * 0.5)
    nx = math.sqrt(1.0 + ax * ax)
    ny = math.sqrt(1.0 + ay * ay)

    margins = []
    for specs in sensor_chain_specs:
        tf = fk_batch(specs, q_batch)
        rot = tf[:, :3, :3]
        trans = tf[:, :3, 3]
        p_sensor = torch.bmm(rot.transpose(1, 2), (point - trans).unsqueeze(-1)).squeeze(-1)
        x = p_sensor[:, 0]
        y = p_sensor[:, 1]
        z = p_sensor[:, 2]
        planes = torch.stack([
            (x + z * ax) / nx,
            (-x + z * ax) / nx,
            (y + z * ay) / ny,
            (-y + z * ay) / ny,
            z - z_min,
            z_max - z,
        ], dim=1)
        margin, _ = torch.min(planes, dim=1)
        margins.append(margin)
    sensor_margins = torch.stack(margins, dim=1)
    best_margin, best_sensor = torch.max(sensor_margins, dim=1)
    g = best_margin - delta
    visible_fov = sensor_margins >= delta
    return sensor_margins, best_sensor, best_margin, g, visible_fov


def transform_points_inv(tf, points):
    rot = tf[:, :3, :3]
    trans = tf[:, :3, 3]
    return torch.bmm(rot.transpose(1, 2), (points - trans).unsqueeze(-1)).squeeze(-1)


def segment_box_hit(local_start, local_end, half_size, ignore_start_inside):
    d = local_end - local_start
    eps = 1e-12
    inside_start = torch.all((local_start >= -half_size) & (local_start <= half_size), dim=1)
    t_min = torch.zeros(local_start.shape[0], device=local_start.device, dtype=local_start.dtype)
    t_max = torch.ones_like(t_min)
    valid = torch.ones_like(t_min, dtype=torch.bool)

    for axis in range(3):
        da = d[:, axis]
        p0 = local_start[:, axis]
        hs = half_size[axis]
        parallel = torch.abs(da) < eps
        valid = valid & (~parallel | ((p0 >= -hs) & (p0 <= hs)))
        inv = torch.where(parallel, torch.ones_like(da), 1.0 / da)
        t1 = (-hs - p0) * inv
        t2 = (hs - p0) * inv
        ta = torch.minimum(t1, t2)
        tb = torch.maximum(t1, t2)
        t_min = torch.where(parallel, t_min, torch.maximum(t_min, ta))
        t_max = torch.where(parallel, t_max, torch.minimum(t_max, tb))

    hit = valid & (t_min <= t_max) & (t_min >= 0.0) & (t_min <= 1.0)
    if ignore_start_inside:
        hit = hit & (~inside_start)
    return hit


def segment_sphere_hit(local_start, local_end, radius, ignore_start_inside):
    d = local_end - local_start
    a = torch.sum(d * d, dim=1)
    b = 2.0 * torch.sum(local_start * d, dim=1)
    c = torch.sum(local_start * local_start, dim=1) - radius * radius
    disc = b * b - 4.0 * a * c
    valid = (a > 1e-12) & (disc >= 0.0)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    denom = 2.0 * torch.where(a > 1e-12, a, torch.ones_like(a))
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    hit = valid & (((t1 >= 0.0) & (t1 <= 1.0)) | ((t2 >= 0.0) & (t2 <= 1.0)))
    if ignore_start_inside:
        hit = hit & (torch.sum(local_start * local_start, dim=1) > radius * radius)
    return hit


def segment_cylinder_hit(local_start, local_end, radius, half_length, ignore_start_inside):
    d = local_end - local_start
    x0, y0, z0 = local_start[:, 0], local_start[:, 1], local_start[:, 2]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    r2 = radius * radius

    inside_start = (x0 * x0 + y0 * y0 <= r2) & (torch.abs(z0) <= half_length)
    hit = torch.zeros_like(x0, dtype=torch.bool)

    a = dx * dx + dy * dy
    b = 2.0 * (x0 * dx + y0 * dy)
    c = x0 * x0 + y0 * y0 - r2
    disc = b * b - 4.0 * a * c
    valid_side = (a > 1e-12) & (disc >= 0.0)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    denom = 2.0 * torch.where(a > 1e-12, a, torch.ones_like(a))
    for t in [(-b - sqrt_disc) / denom, (-b + sqrt_disc) / denom]:
        z = z0 + t * dz
        hit = hit | (valid_side & (t >= 0.0) & (t <= 1.0) & (z >= -half_length) & (z <= half_length))

    valid_cap = torch.abs(dz) > 1e-12
    safe_dz = torch.where(valid_cap, dz, torch.ones_like(dz))
    for z_cap in [-half_length, half_length]:
        t = (z_cap - z0) / safe_dz
        x = x0 + t * dx
        y = y0 + t * dy
        hit = hit | (valid_cap & (t >= 0.0) & (t <= 1.0) & (x * x + y * y <= r2))

    if ignore_start_inside:
        hit = hit & (~inside_start)
    return hit


def prepare_torch_occlusion_context(args, sensor_frames, joint_names, device):
    if not args.occlusion_urdf:
        return None
    robot = URDF.from_xml_file(args.occlusion_urdf)
    primitives = load_collision_primitives(robot)
    if not primitives:
        raise RuntimeError(f"No supported collision primitives found in {args.occlusion_urdf}.")

    sensor_specs = prepare_chain_specs(robot, args.base_frame, sensor_frames, joint_names, device)
    link_specs = [
        prepare_chain_specs(robot, args.base_frame, [primitive["link"]], joint_names, device)[0]
        for primitive in primitives
    ]
    collision_origins = [torch_tensor(primitive["origin"], device) for primitive in primitives]
    return {
        "primitives": primitives,
        "sensor_specs": sensor_specs,
        "link_specs": link_specs,
        "collision_origins": collision_origins,
    }


@torch.no_grad()
def torch_sensor_occlusion(point, q_batch, context, args):
    batch_size = q_batch.shape[0]
    num_sensors = len(context["sensor_specs"])
    device = q_batch.device
    point = point.to(device=device, dtype=q_batch.dtype).reshape(1, 3).expand(batch_size, 3)
    sensor_occluded = torch.zeros((batch_size, num_sensors), device=device, dtype=torch.bool)

    for sensor_idx, specs in enumerate(context["sensor_specs"]):
        sensor_tf = fk_batch(specs, q_batch)
        origin = sensor_tf[:, :3, 3]
        vec = point - origin
        length = torch.linalg.norm(vec, dim=1)
        valid_ray = length >= args.min_ray_length
        direction = vec / torch.clamp(length, min=1e-12).unsqueeze(1)
        start = origin + args.ray_start_offset * direction
        end = point - args.point_end_offset * direction
        segment_len = torch.linalg.norm(end - start, dim=1)
        valid_ray = valid_ray & (segment_len >= args.min_ray_length)

        hit_any = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        for primitive, link_specs, col_origin in zip(
            context["primitives"],
            context["link_specs"],
            context["collision_origins"],
        ):
            if primitive["link"] in args.ignore_links:
                continue
            link_tf = fk_batch(link_specs, q_batch)
            col_tf = torch.bmm(link_tf, col_origin.to(dtype=q_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1))
            local_start = transform_points_inv(col_tf, start)
            local_end = transform_points_inv(col_tf, end)

            geom = primitive["geometry"]
            if primitive["type"] == "box":
                half_size = torch_tensor(0.5 * np.asarray(geom.size, dtype=np.float32), device, dtype=q_batch.dtype)
                hit = segment_box_hit(local_start, local_end, half_size, args.ignore_start_inside)
            elif primitive["type"] == "cylinder":
                hit = segment_cylinder_hit(
                    local_start,
                    local_end,
                    float(geom.radius),
                    0.5 * float(geom.length),
                    args.ignore_start_inside,
                )
            elif primitive["type"] == "sphere":
                hit = segment_sphere_hit(local_start, local_end, float(geom.radius), args.ignore_start_inside)
            else:
                continue
            hit_any = hit_any | (valid_ray & hit)

        sensor_occluded[:, sensor_idx] = hit_any

    return sensor_occluded


def normalize_pq(p, q, p_min, p_max, q_min, q_max):
    p_norm = 2.0 * (p - p_min.reshape(1, 3)) / (p_max - p_min).reshape(1, 3) - 1.0
    q_norm = 2.0 * (q - q_min.reshape(1, 7)) / (q_max - q_min).reshape(1, 7) - 1.0
    return np.concatenate([p_norm, q_norm], axis=1).astype(np.float32)


def grad_wrt_q_physical(pred, x, q_scale):
    grad_x = torch.autograd.grad(
        pred.sum(),
        x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad_x[:, 3:] * q_scale.view(1, 7)


def tension_loss_from_grad_q(grad_q, x, q_scale):
    cols = []
    for j in range(grad_q.shape[1]):
        grad2_x = torch.autograd.grad(
            grad_q[:, j].sum(),
            x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        cols.append(grad2_x[:, 3:] * q_scale.view(1, 7))
    hessian = torch.stack(cols, dim=2)
    return (hessian ** 2).mean()


def compute_losses(model, batch, q_scale, weights):
    x = batch["x"]
    if weights["eikonal"] > 0.0 or weights["tension"] > 0.0 or weights["grad"] > 0.0:
        x = x.detach().clone().requires_grad_(True)

    y = batch["y"]
    pred = model(x)
    mse = torch.mean((pred - y) ** 2)

    zero = pred.new_tensor(0.0)
    eikonal = zero
    tension = zero
    grad_loss = zero
    grad_cos = zero

    if weights["eikonal"] > 0.0 or weights["tension"] > 0.0 or weights["grad"] > 0.0:
        grad_q = grad_wrt_q_physical(pred, x, q_scale)
        grad_norm = torch.linalg.norm(grad_q, dim=1)
        eikonal = torch.mean((grad_norm - 1.0) ** 2)
        if weights["grad"] > 0.0:
            grad_gt = batch["grad_q_gt"]
            grad_cos_each = torch.sum(grad_q * grad_gt, dim=1) / (
                torch.linalg.norm(grad_q, dim=1) * torch.linalg.norm(grad_gt, dim=1) + 1e-8
            )
            grad_cos = grad_cos_each.mean()
            grad_loss = torch.mean(1.0 - grad_cos_each)
        if weights["tension"] > 0.0:
            tension = tension_loss_from_grad_q(grad_q, x, q_scale)

    total = (
        weights["cdf"] * mse
        + weights["eikonal"] * eikonal
        + weights["tension"] * tension
        + weights["grad"] * grad_loss
    )
    return {
        "total": total,
        "mse": mse.detach(),
        "eikonal": eikonal.detach(),
        "tension": tension.detach(),
        "grad": grad_loss.detach(),
        "grad_cos": grad_cos.detach(),
    }


class OnlineVisibilityCDFSampler:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.rng = np.random.default_rng(args.seed)

        q0 = np.load(args.q0, allow_pickle=True)
        required = ["grid_points", "q0_templates", "q0_g", "num_q0"]
        missing = [key for key in required if key not in q0]
        if missing:
            raise RuntimeError(f"q0 file missing required keys: {missing}")

        self.q0 = q0
        self.grid_points = q0["grid_points"].astype(np.float32)
        self.q0_templates = q0["q0_templates"].astype(np.float32)
        self.q0_g = q0["q0_g"]
        self.num_q0 = q0["num_q0"].astype(np.int64)
        self.joint_names = args.joint_names or names_from_npz(q0, "joint_names", DEFAULT_JOINT_NAMES)
        self.sensor_frames = args.sensor_frames or names_from_npz(q0, "sensor_frames", DEFAULT_SENSOR_FRAMES)

        args.horizontal_fov_deg = scalar_from_npz(q0, "horizontal_fov_deg", 50.0) if args.horizontal_fov_deg is None else args.horizontal_fov_deg
        args.vertical_fov_deg = scalar_from_npz(q0, "vertical_fov_deg", 66.0) if args.vertical_fov_deg is None else args.vertical_fov_deg
        args.z_min = scalar_from_npz(q0, "z_min", 0.20) if args.z_min is None else args.z_min
        args.z_max = scalar_from_npz(q0, "z_max", 0.70) if args.z_max is None else args.z_max
        args.delta = scalar_from_npz(q0, "delta", 0.01) if args.delta is None else args.delta

        valid_mask = q0["valid_mask_with_occlusion"].astype(np.bool_) if "valid_mask_with_occlusion" in q0 else q0["valid_mask"].astype(np.bool_)
        self.valid_rows = np.where(valid_mask & (self.num_q0 > 0))[0].astype(np.int64)
        if len(self.valid_rows) == 0:
            raise RuntimeError("No valid Q0 rows found.")

        if args.val_points > 0:
            perm = np.random.default_rng(args.seed + 17).permutation(self.valid_rows)
            self.val_rows = perm[:min(args.val_points, len(perm))]
            self.train_rows = perm[min(args.val_points, len(perm)):]
            if len(self.train_rows) == 0:
                self.train_rows = self.valid_rows
        else:
            self.train_rows = self.valid_rows
            self.val_rows = self.valid_rows

        self.p_min = np.asarray(args.p_min, dtype=np.float32)
        self.p_max = np.asarray(args.p_max, dtype=np.float32)
        self.q_min = np.asarray(args.q_min, dtype=np.float32)
        self.q_max = np.asarray(args.q_max, dtype=np.float32)

        robot = URDF.from_xml_file(args.urdf)
        self.joint_limits = get_joint_limits(robot, self.joint_names)
        self.sensor_chain_specs = prepare_chain_specs(robot, args.base_frame, self.sensor_frames, self.joint_names, device)
        self.occlusion_context = prepare_torch_occlusion_context(args, self.sensor_frames, self.joint_names, device)

    def select_valid_q0(self, row):
        k = int(self.num_q0[row])
        if k <= 0:
            return None
        finite = np.isfinite(self.q0_g[row, :k]) & np.isfinite(self.q0_templates[row, :k]).all(axis=1)
        if "q0_valid_with_occlusion" in self.q0:
            finite = finite & self.q0["q0_valid_with_occlusion"][row, :k].astype(np.bool_)
        idx = np.where(finite)[0]
        if len(idx) == 0:
            return None
        return self.q0_templates[row, idx]

    @torch.no_grad()
    def visibility_label_batch(self, point_np, q_batch_np):
        q_batch = torch_tensor(q_batch_np, self.device)
        point = torch_tensor(point_np, self.device)
        sensor_margins, _best_sensor, _best_margin, _g, visible_fov = visibility_margins_batch(
            point,
            q_batch,
            self.sensor_chain_specs,
            self.args.horizontal_fov_deg,
            self.args.vertical_fov_deg,
            self.args.z_min,
            self.args.z_max,
            self.args.delta,
        )
        if self.occlusion_context is not None:
            sensor_occluded = torch_sensor_occlusion(point, q_batch, self.occlusion_context, self.args)
            sensor_visible = visible_fov & (~sensor_occluded)
        else:
            sensor_visible = visible_fov
        visible = torch.any(sensor_visible, dim=1)
        return visible.detach().cpu().numpy().astype(np.bool_)

    def sample_batch(self, batch_x, batch_q, split="train"):
        rows_pool = self.train_rows if split == "train" else self.val_rows
        rows = self.rng.choice(rows_pool, size=batch_x, replace=(len(rows_pool) < batch_x))

        p_list = []
        q_list = []
        y_list = []
        visible_list = []
        grad_list = []

        for row in rows:
            q0_set = self.select_valid_q0(int(row))
            if q0_set is None:
                continue

            point = self.grid_points[int(row)]
            q_batch_np = sample_q_batch(self.rng, self.joint_limits, self.joint_names, batch_q).astype(np.float32)
            p_batch_np = np.repeat(point.reshape(1, 3), batch_q, axis=0).astype(np.float32)

            if self.args.metric == "l1":
                dist_all = np.sum(np.abs(q_batch_np[:, None, :] - q0_set[None, :, :]), axis=2)
            else:
                diff_all = q_batch_np[:, None, :] - q0_set[None, :, :]
                dist_all = np.linalg.norm(diff_all, axis=2)
            nearest_local = np.argmin(dist_all, axis=1)
            unsigned = dist_all[np.arange(batch_q), nearest_local].astype(np.float32)
            nearest_q0 = q0_set[nearest_local]

            visible = self.visibility_label_batch(point, q_batch_np)
            sign = np.where(visible, 1.0, -1.0).astype(np.float32)
            y = sign * unsigned
            if self.args.target_clip > 0.0:
                y = np.clip(y, -self.args.target_clip, self.args.target_clip)

            diff = q_batch_np - nearest_q0
            norm = np.linalg.norm(diff, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-6)
            grad_q_gt = sign.reshape(-1, 1) * diff / norm

            p_list.append(p_batch_np)
            q_list.append(q_batch_np)
            y_list.append(y.astype(np.float32))
            visible_list.append(visible)
            grad_list.append(grad_q_gt.astype(np.float32))

        if not p_list:
            raise RuntimeError("Failed to sample a non-empty online batch.")

        p = np.concatenate(p_list, axis=0)
        q = np.concatenate(q_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        visible = np.concatenate(visible_list, axis=0)
        grad = np.concatenate(grad_list, axis=0)
        x = normalize_pq(p, q, self.p_min, self.p_max, self.q_min, self.q_max)

        return {
            "x": torch.from_numpy(x).to(self.device),
            "y": torch.from_numpy(y).to(self.device),
            "visible": torch.from_numpy(visible).to(self.device),
            "grad_q_gt": torch.from_numpy(grad).to(self.device),
        }


@torch.no_grad()
def eval_online(model, sampler, args):
    model.eval()
    total = 0
    se = 0.0
    ae = 0.0
    sign_ok = 0
    false_visible = 0
    false_invisible = 0
    near = {0.25: [0, 0.0], 0.5: [0, 0.0], 1.0: [0, 0.0]}

    for _ in range(args.val_batches):
        batch = sampler.sample_batch(args.batch_x, args.batch_q, split="val")
        y = batch["y"]
        visible = batch["visible"]
        pred = model(batch["x"])
        err = pred - y
        total += len(y)
        se += torch.sum(err ** 2).item()
        ae += torch.sum(torch.abs(err)).item()
        pred_visible = pred >= 0.0
        sign_ok += torch.count_nonzero(pred_visible == visible).item()
        false_visible += torch.count_nonzero(pred_visible & (~visible)).item()
        false_invisible += torch.count_nonzero((~pred_visible) & visible).item()
        abs_y = torch.abs(y)
        abs_err = torch.abs(err)
        for band in near:
            mask = abs_y <= band
            count = torch.count_nonzero(mask).item()
            if count > 0:
                near[band][0] += count
                near[band][1] += torch.sum(abs_err[mask]).item()

    out = {
        "mse": se / max(total, 1),
        "rmse": math.sqrt(se / max(total, 1)),
        "mae": ae / max(total, 1),
        "sign_acc": sign_ok / max(total, 1),
        "false_visible_rate": false_visible / max(total, 1),
        "false_invisible_rate": false_invisible / max(total, 1),
    }
    for band, (count, err_sum) in near.items():
        out[f"near_{band:g}_count"] = count
        out[f"near_{band:g}_mae"] = err_sum / count if count > 0 else float("nan")
    return out


def save_checkpoint(args, model, optimizer, epoch, global_step, best_val, sampler, path):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_mse": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_dim": 10,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "activation": args.activation,
        },
        "normalization": {
            "p_min": sampler.p_min.tolist(),
            "p_max": sampler.p_max.tolist(),
            "q_min": sampler.q_min.tolist(),
            "q_max": sampler.q_max.tolist(),
            "target_clip": args.target_clip,
            "metric": args.metric,
        },
        "training_mode": "online_yiming_style_torch_occlusion",
        "q0_path": args.q0,
        "urdf": args.urdf,
        "occlusion_urdf": args.occlusion_urdf,
        "batch_x": args.batch_x,
        "batch_q": args.batch_q,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train visibility CDF with Yiming-style online q sampling and batched torch self-occlusion."
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--q0", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--joint-names", nargs="*", default=None)
    parser.add_argument("--sensor-frames", nargs="*", default=None)
    parser.add_argument("--occlusion-urdf", default="")
    parser.add_argument("--ray-start-offset", type=float, default=0.03)
    parser.add_argument("--point-end-offset", type=float, default=0.005)
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--ignore-start-inside", action="store_true", default=True)
    parser.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    parser.add_argument("--ignore-links", nargs="*", default=[])

    parser.add_argument("--metric", choices=["l2", "l1"], default="l2")
    parser.add_argument("--target-clip", type=float, default=5.0)
    parser.add_argument("--p-min", nargs=3, type=float, default=DEFAULT_P_MIN)
    parser.add_argument("--p-max", nargs=3, type=float, default=DEFAULT_P_MAX)
    parser.add_argument("--q-min", nargs=7, type=float, default=DEFAULT_Q_MIN)
    parser.add_argument("--q-max", nargs=7, type=float, default=DEFAULT_Q_MAX)
    parser.add_argument("--horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--vertical-fov-deg", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--activation", choices=["relu", "softplus", "silu"], default="relu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-x", type=int, default=10)
    parser.add_argument("--batch-q", type=int, default=100)
    parser.add_argument("--val-points", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-cdf-weight", type=float, default=1.0)
    parser.add_argument("--loss-eikonal-weight", type=float, default=0.0)
    parser.add_argument("--loss-tension-weight", type=float, default=0.0)
    parser.add_argument("--loss-grad-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    sampler = OnlineVisibilityCDFSampler(args, device)
    model = MLP(10, 1, args.hidden_dim, args.num_layers, args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    q_min = torch.tensor(args.q_min, dtype=torch.float32, device=device)
    q_max = torch.tensor(args.q_max, dtype=torch.float32, device=device)
    q_scale = 2.0 / (q_max - q_min)
    weights = {
        "cdf": args.loss_cdf_weight,
        "eikonal": args.loss_eikonal_weight,
        "tension": args.loss_tension_weight,
        "grad": args.loss_grad_weight,
    }

    config = vars(args).copy()
    config.update({
        "device_used": str(device),
        "valid_q0_rows": int(len(sampler.valid_rows)),
        "train_q0_rows": int(len(sampler.train_rows)),
        "val_q0_rows": int(len(sampler.val_rows)),
        "samples_per_step": int(args.batch_x * args.batch_q),
        "training_mode": "online_yiming_style_torch_occlusion",
        "loss_formula": "cdf*MSE, with optional eikonal/tension/grad terms disabled by default",
    })
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("")
    print("=== Online Visibility CDF Training Config ===")
    print(f"urdf:          {args.urdf}")
    print(f"q0:            {args.q0}")
    print(f"occlusion:     {args.occlusion_urdf if args.occlusion_urdf else '<none>'}")
    print("occlusion_backend: torch")
    print(f"output_dir:    {args.output_dir}")
    print(f"device:        {device}")
    print(f"valid q0 rows: {len(sampler.valid_rows)} train={len(sampler.train_rows)} val={len(sampler.val_rows)}")
    print(f"online batch:  batch_x={args.batch_x}, batch_q={args.batch_q}, samples/step={args.batch_x * args.batch_q}")
    print(f"steps/epoch:   {args.steps_per_epoch}")
    print(f"model:         MLP input=10 hidden={args.hidden_dim} layers={args.num_layers} activation={args.activation}")
    print(f"loss weights:  cdf={weights['cdf']} eikonal={weights['eikonal']} tension={weights['tension']} grad={weights['grad']}")
    print(f"target_clip:   {args.target_clip}")

    best_val = float("inf")
    global_step = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "mse": 0.0, "eikonal": 0.0, "tension": 0.0, "grad": 0.0, "grad_cos": 0.0}
        count = 0

        for _ in range(args.steps_per_epoch):
            batch = sampler.sample_batch(args.batch_x, args.batch_q, split="train")
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, batch, q_scale, weights)
            losses["total"].backward()
            optimizer.step()

            bs = len(batch["y"])
            count += bs
            global_step += 1
            for key in sums:
                sums[key] += losses[key].item() * bs

        train_log = {key: sums[key] / max(count, 1) for key in sums}
        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs

        if do_eval:
            val_log = eval_online(model, sampler, args)
            scheduler.step(val_log["mse"])
            history.append({"epoch": epoch, "global_step": global_step, "train": train_log, "val": val_log})
            print(
                f"[epoch {epoch:04d} step {global_step:07d}] "
                f"train_total={train_log['total']:.6f} train_mse={train_log['mse']:.6f} "
                f"grad_cos={train_log['grad_cos']:.4f} "
                f"val_mse={val_log['mse']:.6f} val_mae={val_log['mae']:.6f} "
                f"sign_acc={val_log['sign_acc']:.4f} near0.5_count={val_log['near_0.5_count']} "
                f"near0.5_mae={val_log['near_0.5_mae']:.6f} elapsed={time.time() - t0:.1f}s"
            )

            if val_log["mse"] < best_val:
                best_val = val_log["mse"]
                save_checkpoint(args, model, optimizer, epoch, global_step, best_val, sampler, os.path.join(args.output_dir, "best.pt"))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args, model, optimizer, epoch, global_step, best_val, sampler, os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"))

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("")
    print(f"best_val_mse: {best_val:.9f}")
    print(f"saved:        {os.path.join(args.output_dir, 'best.pt')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

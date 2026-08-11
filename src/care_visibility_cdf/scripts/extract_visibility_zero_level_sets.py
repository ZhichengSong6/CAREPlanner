#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from urdf_parser_py.urdf import URDF

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    find_chain_joints,
    fk_transform,
    get_joint_limits,
    make_transform,
)
from check_visibility_self_occlusion import (  # noqa: E402
    load_collision_primitives,
    raycast_self_occlusion,
)


def torch_tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(array, device=device, dtype=dtype)


def joint_origin_np(joint) -> np.ndarray:
    if joint.origin is None:
        return np.eye(4)
    xyz = joint.origin.xyz if joint.origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = joint.origin.rpy if joint.origin.rpy is not None else [0.0, 0.0, 0.0]
    return make_transform(xyz, rpy)


def prepare_chain_specs(robot: URDF, base_frame: str, sensor_frames: List[str],
                        joint_names: List[str], device):
    joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    all_specs = []

    for frame in sensor_frames:
        chain = find_chain_joints(robot, base_frame, frame)
        specs = []
        for joint in chain:
            q_index = joint_name_to_index.get(joint.name, -1)
            axis = joint.axis if joint.axis is not None else [1.0, 0.0, 0.0]
            axis = np.asarray(axis, dtype=np.float32)
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axis = axis / norm
            specs.append({
                "name": joint.name,
                "type": joint.type,
                "q_index": q_index,
                "origin": torch_tensor(joint_origin_np(joint), device),
                "axis": torch_tensor(axis, device),
            })
        all_specs.append(specs)

    return all_specs


def batch_eye(batch_size: int, device):
    return torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)


def batch_revolute_transform(axis: torch.Tensor, q: torch.Tensor, device):
    batch_size = q.shape[0]
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(q)
    s = torch.sin(q)
    one_minus_c = 1.0 - c

    rot = torch.zeros((batch_size, 3, 3), device=device, dtype=q.dtype)
    rot[:, 0, 0] = c + x * x * one_minus_c
    rot[:, 0, 1] = x * y * one_minus_c - z * s
    rot[:, 0, 2] = x * z * one_minus_c + y * s
    rot[:, 1, 0] = y * x * one_minus_c + z * s
    rot[:, 1, 1] = c + y * y * one_minus_c
    rot[:, 1, 2] = y * z * one_minus_c - x * s
    rot[:, 2, 0] = z * x * one_minus_c - y * s
    rot[:, 2, 1] = z * y * one_minus_c + x * s
    rot[:, 2, 2] = c + z * z * one_minus_c

    transform = batch_eye(batch_size, device)
    transform[:, :3, :3] = rot
    return transform


def batch_prismatic_transform(axis: torch.Tensor, q: torch.Tensor, device):
    batch_size = q.shape[0]
    transform = batch_eye(batch_size, device)
    transform[:, :3, 3] = q.unsqueeze(-1) * axis.unsqueeze(0)
    return transform


def fk_sensor_batch(chain_specs, q_batch: torch.Tensor):
    batch_size = q_batch.shape[0]
    device = q_batch.device
    transform = batch_eye(batch_size, device)

    for spec in chain_specs:
        origin = spec["origin"].unsqueeze(0).expand(batch_size, -1, -1)
        transform = torch.bmm(transform, origin)

        if spec["q_index"] < 0 or spec["type"] == "fixed":
            continue

        q = q_batch[:, spec["q_index"]]
        if spec["type"] in ["revolute", "continuous"]:
            motion = batch_revolute_transform(spec["axis"], q, device)
            transform = torch.bmm(transform, motion)
        elif spec["type"] == "prismatic":
            motion = batch_prismatic_transform(spec["axis"], q, device)
            transform = torch.bmm(transform, motion)

    return transform


def visibility_g_batch(point: torch.Tensor, q_batch: torch.Tensor, all_chain_specs,
                       horizontal_fov_deg: float, vertical_fov_deg: float,
                       z_min: float, z_max: float, delta: float):
    """
    Visibility FOV margin batch evaluator.

    Backward-compatible modes:

    1. Single point mode:
        point:   [3]
        q_batch: [Bq, 7]

        returns:
            g:              [Bq]
            sensor_margins: [Bq, S]
            active_sensor:  [Bq]
            active_planes:  [Bq, S]

    2. Batched point mode:
        point:   [Bx, 3]
        q_batch: [Bq, 7]

        returns:
            g:              [Bx, Bq]
            sensor_margins: [Bx, Bq, S]
            active_sensor:  [Bx, Bq]
            active_planes:  [Bx, Bq, S]

    This keeps old zero-level data generation unchanged while allowing
    signed CDF training to compute FOV signs for all sampled points at once.
    """

    device = q_batch.device
    dtype = q_batch.dtype
    Bq = q_batch.shape[0]

    point = point.to(device=device, dtype=dtype)

    single_point = False
    if point.ndim == 1:
        point = point.reshape(1, 3)
        single_point = True
    elif point.ndim == 2:
        if point.shape[1] != 3:
            raise RuntimeError(f"Expected point shape [Bx,3], got {tuple(point.shape)}")
    else:
        raise RuntimeError(f"Expected point shape [3] or [Bx,3], got {tuple(point.shape)}")

    Bx = point.shape[0]

    ax = math.tan(math.radians(horizontal_fov_deg) * 0.5)
    ay = math.tan(math.radians(vertical_fov_deg) * 0.5)
    nx = math.sqrt(1.0 + ax * ax)
    ny = math.sqrt(1.0 + ay * ay)

    sensor_margins = []
    active_planes = []

    for chain_specs in all_chain_specs:
        # transform: [Bq,4,4], sensor frame to world frame
        transform = fk_sensor_batch(chain_specs, q_batch)

        rot = transform[:, :3, :3]      # [Bq,3,3]
        trans = transform[:, :3, 3]     # [Bq,3]

        # point_world - sensor_origin_world:
        # [Bx,1,3] - [1,Bq,3] -> [Bx,Bq,3]
        diff_world = point[:, None, :] - trans[None, :, :]

        # Transform world vector into sensor frame:
        # point_sensor[b,q,i] = sum_j rot[q,j,i] * diff_world[b,q,j]
        # equivalent to R^T @ diff.
        point_sensor = torch.einsum("qji,bqj->bqi", rot, diff_world)

        x = point_sensor[:, :, 0]   # [Bx,Bq]
        y = point_sensor[:, :, 1]
        z = point_sensor[:, :, 2]

        planes = torch.stack(
            [
                (x + z * ax) / nx,
                (-x + z * ax) / nx,
                (y + z * ay) / ny,
                (-y + z * ay) / ny,
                z - z_min,
                z_max - z,
            ],
            dim=-1,
        )  # [Bx,Bq,6]

        margin, plane_idx = torch.min(planes, dim=-1)  # [Bx,Bq]

        sensor_margins.append(margin)
        active_planes.append(plane_idx)

    sensor_margins = torch.stack(sensor_margins, dim=-1)  # [Bx,Bq,S]
    active_planes = torch.stack(active_planes, dim=-1)    # [Bx,Bq,S]

    best_margin, active_sensor = torch.max(sensor_margins, dim=-1)  # [Bx,Bq]
    g = best_margin - delta

    if single_point:
        # Preserve old API for data generation.
        return (
            g.squeeze(0),                         # [Bq]
            sensor_margins.squeeze(0),            # [Bq,S]
            active_sensor.squeeze(0),             # [Bq]
            active_planes.squeeze(0),             # [Bq,S]
        )

    return g, sensor_margins, active_sensor, active_planes


def visibility_objective_batch(point: torch.Tensor, q_batch: torch.Tensor, all_chain_specs,
                               horizontal_fov_deg: float, vertical_fov_deg: float,
                               z_min: float, z_max: float, delta: float,
                               target_sensor: int = -1):
    g_global, sensor_margins, active_sensor, active_planes = visibility_g_batch(
        point,
        q_batch,
        all_chain_specs,
        horizontal_fov_deg,
        vertical_fov_deg,
        z_min,
        z_max,
        delta,
    )
    if target_sensor >= 0:
        g = sensor_margins[:, target_sensor] - delta
    else:
        g = g_global
    return g, sensor_margins, active_sensor, active_planes


def sample_q_batch(rng: np.random.Generator, joint_limits: Dict[str, Tuple[float, float]],
                   joint_names: List[str], batch_size: int):
    q = np.zeros((batch_size, len(joint_names)), dtype=np.float32)
    for joint_idx, name in enumerate(joint_names):
        lower, upper = joint_limits[name]
        q[:, joint_idx] = rng.uniform(lower, upper, size=batch_size)
    return q


def clamp_q_(q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor):
    with torch.no_grad():
        q.copy_(torch.max(torch.min(q, q_max.unsqueeze(0)), q_min.unsqueeze(0)))


def refine_q_to_zero_level(point_np: np.ndarray, q_init_np: np.ndarray, all_chain_specs,
                           q_min: torch.Tensor, q_max: torch.Tensor, args, device,
                           target_sensor: int = -1):
    point = torch_tensor(point_np, device)
    q = torch_tensor(q_init_np, device).clone().detach().requires_grad_(True)

    with torch.no_grad():
        g_initial, _, _, _ = visibility_objective_batch(
            point,
            q,
            all_chain_specs,
            args.horizontal_fov_deg,
            args.vertical_fov_deg,
            args.z_min,
            args.z_max,
            args.delta,
            target_sensor,
        )

    if args.zero_level_optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [q],
            lr=args.lbfgs_lr,
            max_iter=args.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            g, _, _, _ = visibility_objective_batch(
                point,
                q,
                all_chain_specs,
                args.horizontal_fov_deg,
                args.vertical_fov_deg,
                args.z_min,
                args.z_max,
                args.delta,
                target_sensor,
            )
            loss = torch.sum(g * g)
            loss.backward()
            return loss

        optimizer.step(closure)
        best_q = q.detach()
    else:
        optimizer = torch.optim.Adam([q], lr=args.lr)
        best_q = q.detach().clone()
        best_abs_g = torch.abs(g_initial).detach().clone()

        for _ in range(args.max_iter):
            optimizer.zero_grad()
            g, _, _, _ = visibility_objective_batch(
                point,
                q,
                all_chain_specs,
                args.horizontal_fov_deg,
                args.vertical_fov_deg,
                args.z_min,
                args.z_max,
                args.delta,
                target_sensor,
            )
            loss = torch.mean(g * g)
            loss.backward()
            optimizer.step()
            clamp_q_(q, q_min, q_max)

            with torch.no_grad():
                g_after, _, _, _ = visibility_objective_batch(
                    point,
                    q,
                    all_chain_specs,
                    args.horizontal_fov_deg,
                    args.vertical_fov_deg,
                    args.z_min,
                    args.z_max,
                    args.delta,
                    target_sensor,
                )
                abs_g = torch.abs(g_after)
                improve = abs_g < best_abs_g
                best_abs_g[improve] = abs_g[improve]
                best_q[improve] = q.detach()[improve]

    with torch.no_grad():
        g, sensor_margins, active_sensor, active_planes = visibility_objective_batch(
            point,
            best_q,
            all_chain_specs,
            args.horizontal_fov_deg,
            args.vertical_fov_deg,
            args.z_min,
            args.z_max,
            args.delta,
            target_sensor,
        )
        abs_g = torch.abs(g)
        boundary_mask = ((best_q > q_min.unsqueeze(0)) & (best_q < q_max.unsqueeze(0))).all(dim=1)
        keep = (abs_g <= args.epsilon) & boundary_mask
        keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
        return (
            best_q[keep].detach().cpu().numpy(),
            g[keep].detach().cpu().numpy(),
            sensor_margins[keep].detach().cpu().numpy(),
            active_sensor[keep].detach().cpu().numpy(),
            active_planes[keep].detach().cpu().numpy(),
            q.detach().new_tensor(q_init_np)[keep].detach().cpu().numpy(),
            g_initial[keep].detach().cpu().numpy(),
            keep_indices.detach().cpu().numpy(),
        )


def greedy_farthest_downsample(q: np.ndarray, k: int, seed: int):
    if len(q) <= k:
        return np.arange(len(q), dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, len(q)))]
    dist_sq = np.sum((q - q[selected[0]]) ** 2, axis=1)

    for _ in range(1, k):
        idx = int(np.argmax(dist_sq))
        selected.append(idx)
        new_dist_sq = np.sum((q - q[idx]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)

    return np.asarray(selected, dtype=np.int64)


def q_row_to_map(joint_names: List[str], q_row: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(joint_names, q_row)}


def prepare_occlusion_context(args, sensor_frames: List[str], joint_names: List[str], device):
    if not args.occlusion_urdf:
        return None

    occlusion_robot = URDF.from_xml_file(args.occlusion_urdf)
    primitives = load_collision_primitives(occlusion_robot)
    if not primitives:
        raise RuntimeError(
            f"No supported collision primitives found in {args.occlusion_urdf}. "
            "Expected box/cylinder/sphere collision geometry."
        )

    ignore_links = set(args.ignore_links)
    primitives = [primitive for primitive in primitives if primitive["link"] not in ignore_links]

    collision_chains = [
        find_chain_joints(occlusion_robot, args.base_frame, primitive["link"])
        for primitive in primitives
    ]
    sensor_chains = [
        find_chain_joints(occlusion_robot, args.base_frame, frame)
        for frame in sensor_frames
    ]

    use_torch = args.occlusion_backend == "torch" or (
        args.occlusion_backend == "auto" and device.type == "cuda"
    )

    context = {
        "robot": occlusion_robot,
        "primitives": primitives,
        "collision_chains": collision_chains,
        "sensor_chains": sensor_chains,
        "backend": "torch" if use_torch else "cpu",
    }
    if use_torch:
        context["collision_chain_specs"] = [
            prepare_chain_specs(occlusion_robot, args.base_frame, [primitive["link"]], joint_names, device)[0]
            for primitive in primitives
        ]
        context["sensor_chain_specs"] = prepare_chain_specs(
            occlusion_robot, args.base_frame, sensor_frames, joint_names, device
        )
        context["primitive_specs"] = [
            make_torch_primitive_spec(primitive, device)
            for primitive in primitives
        ]
    return context


def make_torch_primitive_spec(primitive, device):
    geom = primitive["geometry"]
    spec = {
        "type": primitive["type"],
        "origin": torch_tensor(primitive["origin"], device),
    }
    if primitive["type"] == "box":
        spec["half_size"] = torch_tensor(0.5 * np.asarray(geom.size, dtype=np.float32), device)
    elif primitive["type"] == "cylinder":
        spec["radius"] = float(geom.radius)
        spec["half_length"] = 0.5 * float(geom.length)
    elif primitive["type"] == "sphere":
        spec["radius"] = float(geom.radius)
    return spec


def transform_points_inverse_batch(transform: torch.Tensor, points: torch.Tensor):
    rot = transform[:, :3, :3]
    trans = transform[:, :3, 3]
    return torch.bmm(rot.transpose(1, 2), (points - trans).unsqueeze(-1)).squeeze(-1)


def segment_box_hit_batch(local_start: torch.Tensor, local_end: torch.Tensor,
                          half_size: torch.Tensor, ignore_start_inside: bool):
    direction = local_end - local_start
    eps = 1e-12
    parallel = torch.abs(direction) < eps
    outside_parallel = parallel & ((local_start < -half_size.unsqueeze(0)) | (local_start > half_size.unsqueeze(0)))

    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    t1 = (-half_size.unsqueeze(0) - local_start) / safe_direction
    t2 = ( half_size.unsqueeze(0) - local_start) / safe_direction
    t_near = torch.minimum(t1, t2)
    t_far = torch.maximum(t1, t2)
    t_near = torch.where(parallel, torch.full_like(t_near, -torch.inf), t_near)
    t_far = torch.where(parallel, torch.full_like(t_far, torch.inf), t_far)

    t_min = torch.maximum(torch.max(t_near, dim=1).values, torch.zeros(local_start.shape[0], device=local_start.device))
    t_max = torch.minimum(torch.min(t_far, dim=1).values, torch.ones(local_start.shape[0], device=local_start.device))
    hit = (t_min <= t_max) & (~torch.any(outside_parallel, dim=1))

    if ignore_start_inside:
        inside = torch.all((local_start >= -half_size.unsqueeze(0)) & (local_start <= half_size.unsqueeze(0)), dim=1)
        hit = hit & (~inside)

    return hit, t_min


def segment_sphere_hit_batch(local_start: torch.Tensor, local_end: torch.Tensor,
                             radius: float, ignore_start_inside: bool):
    direction = local_end - local_start
    a = torch.sum(direction * direction, dim=1)
    b = 2.0 * torch.sum(local_start * direction, dim=1)
    c = torch.sum(local_start * local_start, dim=1) - radius * radius
    disc = b * b - 4.0 * a * c
    valid = (a > 1e-12) & (disc >= 0.0)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    denom = 2.0 * torch.where(a > 1e-12, a, torch.ones_like(a))
    t_a = (-b - sqrt_disc) / denom
    t_b = (-b + sqrt_disc) / denom
    inf = torch.full_like(t_a, torch.inf)
    t_candidates = torch.stack([
        torch.where((0.0 <= t_a) & (t_a <= 1.0), t_a, inf),
        torch.where((0.0 <= t_b) & (t_b <= 1.0), t_b, inf),
    ], dim=1)
    t = torch.min(t_candidates, dim=1).values
    hit = valid & torch.isfinite(t)
    if ignore_start_inside:
        inside = torch.sum(local_start * local_start, dim=1) <= radius * radius
        hit = hit & (~inside)
    return hit, t


def segment_cylinder_hit_batch(local_start: torch.Tensor, local_end: torch.Tensor,
                               radius: float, half_length: float, ignore_start_inside: bool):
    direction = local_end - local_start
    px, py, pz = local_start[:, 0], local_start[:, 1], local_start[:, 2]
    dx, dy, dz = direction[:, 0], direction[:, 1], direction[:, 2]
    inf = torch.full_like(px, torch.inf)
    candidates = []

    a = dx * dx + dy * dy
    b = 2.0 * (px * dx + py * dy)
    c = px * px + py * py - radius * radius
    disc = b * b - 4.0 * a * c
    side_valid = (a > 1e-12) & (disc >= 0.0)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    denom = 2.0 * torch.where(a > 1e-12, a, torch.ones_like(a))
    for t_side in [(-b - sqrt_disc) / denom, (-b + sqrt_disc) / denom]:
        z = pz + t_side * dz
        ok = side_valid & (0.0 <= t_side) & (t_side <= 1.0) & (-half_length <= z) & (z <= half_length)
        candidates.append(torch.where(ok, t_side, inf))

    cap_valid = torch.abs(dz) > 1e-12
    for z_cap in [-half_length, half_length]:
        t_cap = (z_cap - pz) / torch.where(cap_valid, dz, torch.ones_like(dz))
        x = px + t_cap * dx
        y = py + t_cap * dy
        ok = cap_valid & (0.0 <= t_cap) & (t_cap <= 1.0) & (x * x + y * y <= radius * radius)
        candidates.append(torch.where(ok, t_cap, inf))

    t = torch.min(torch.stack(candidates, dim=1), dim=1).values
    hit = torch.isfinite(t)
    if ignore_start_inside:
        inside = (px * px + py * py <= radius * radius) & (torch.abs(pz) <= half_length)
        hit = hit & (~inside)
    return hit, t


def compute_q0_occlusion_torch(point: np.ndarray, q_rows: np.ndarray, sensor_indices: np.ndarray,
                               occlusion_context, args, device):
    count = len(q_rows)
    occluded = torch.zeros((count,), dtype=torch.bool, device=device)
    if occlusion_context is None or count == 0:
        return occluded.detach().cpu().numpy()

    valid_sensor = sensor_indices >= 0
    if not np.all(valid_sensor):
        occluded[torch_tensor(~valid_sensor, device, dtype=torch.bool)] = True
    if not np.any(valid_sensor):
        return occluded.detach().cpu().numpy()

    q = torch_tensor(q_rows, device)
    sensor_indices_t = torch.as_tensor(sensor_indices, device=device, dtype=torch.long)
    point_t = torch_tensor(point, device).reshape(1, 3).expand(count, 3)

    sensor_transforms = batch_eye(count, device).to(dtype=q.dtype)
    for sensor_idx, chain_specs in enumerate(occlusion_context["sensor_chain_specs"]):
        mask = sensor_indices_t == sensor_idx
        if torch.any(mask):
            sensor_transforms[mask] = fk_sensor_batch(chain_specs, q[mask])

    sensor_origin = sensor_transforms[:, :3, 3]
    vec = point_t - sensor_origin
    length = torch.linalg.norm(vec, dim=1)
    valid_ray = length >= args.min_ray_length
    direction = vec / torch.clamp(length.unsqueeze(1), min=args.min_ray_length)
    start = sensor_origin + args.ray_start_offset * direction
    end = point_t - args.point_end_offset * direction
    segment = end - start
    segment_len = torch.linalg.norm(segment, dim=1)
    valid_ray = valid_ray & (segment_len >= args.min_ray_length)

    for primitive_spec, chain_specs in zip(
        occlusion_context["primitive_specs"],
        occlusion_context["collision_chain_specs"],
    ):
        link_transform = fk_sensor_batch(chain_specs, q)
        origin = primitive_spec["origin"].unsqueeze(0).expand(count, -1, -1)
        collision_transform = torch.bmm(link_transform, origin)
        local_start = transform_points_inverse_batch(collision_transform, start)
        local_end = transform_points_inverse_batch(collision_transform, end)

        if primitive_spec["type"] == "box":
            hit, t = segment_box_hit_batch(
                local_start,
                local_end,
                primitive_spec["half_size"],
                args.ignore_start_inside,
            )
        elif primitive_spec["type"] == "cylinder":
            hit, t = segment_cylinder_hit_batch(
                local_start,
                local_end,
                primitive_spec["radius"],
                primitive_spec["half_length"],
                args.ignore_start_inside,
            )
        elif primitive_spec["type"] == "sphere":
            hit, t = segment_sphere_hit_batch(
                local_start,
                local_end,
                primitive_spec["radius"],
                args.ignore_start_inside,
            )
        else:
            continue

        distance = torch.clamp(t, 0.0, 1.0) * segment_len + args.ray_start_offset
        hit = hit & valid_ray & (distance >= args.min_hit_distance)
        occluded = occluded | hit

    return occluded.detach().cpu().numpy()


def compute_q0_occlusion(point: np.ndarray, q_rows: np.ndarray, sensor_indices: np.ndarray,
                         joint_names: List[str], occlusion_context, args):
    occluded = np.zeros((len(q_rows),), dtype=np.bool_)
    if occlusion_context is None or len(q_rows) == 0:
        return occluded

    for idx, (q_row, sensor_idx) in enumerate(zip(q_rows, sensor_indices)):
        if sensor_idx < 0:
            occluded[idx] = True
            continue
        q_map = q_row_to_map(joint_names, q_row)
        sensor_transform = fk_transform(
            occlusion_context["sensor_chains"][int(sensor_idx)],
            q_map,
        )
        is_occluded, _ = raycast_self_occlusion(
            occlusion_context["robot"],
            occlusion_context["collision_chains"],
            occlusion_context["primitives"],
            sensor_transform,
            point.astype(np.float64),
            q_map,
            args,
        )
        occluded[idx] = is_occluded
    return occluded


def append_q0_block(
    out_idx: int,
    q0: np.ndarray,
    g0: np.ndarray,
    sensor_margins0: np.ndarray,
    active_sensor0: np.ndarray,
    active_planes0: np.ndarray,
    q0_init: np.ndarray,
    q0_init_g0: np.ndarray,
    kept_candidate_indices: np.ndarray,
    init_indices: np.ndarray,
    target_count: int,
    seed: int,
    q0_templates: np.ndarray,
    q0_g: np.ndarray,
    q0_sensor_margins: np.ndarray,
    q0_active_sensor: np.ndarray,
    q0_active_planes: np.ndarray,
    q0_init_templates: np.ndarray,
    q0_init_g: np.ndarray,
    q0_delta_q: np.ndarray,
    q0_delta_norm_l1: np.ndarray,
    q0_delta_norm_l2: np.ndarray,
    q0_source_topk_rank: np.ndarray,
    q0_source_random_index: np.ndarray,
    q0_active_sensor_occluded: np.ndarray,
    q0_valid_with_occlusion: np.ndarray,
    occluded: np.ndarray,
):
    if len(q0) == 0:
        return 0

    keep_indices = greedy_farthest_downsample(q0, target_count, seed)
    q0 = q0[keep_indices]
    g0 = g0[keep_indices]
    sensor_margins0 = sensor_margins0[keep_indices]
    active_sensor0 = active_sensor0[keep_indices]
    active_planes0 = active_planes0[keep_indices]
    q0_init = q0_init[keep_indices]
    q0_init_g0 = q0_init_g0[keep_indices]
    kept_candidate_indices = kept_candidate_indices[keep_indices]
    occluded = occluded[keep_indices]
    delta_q = q0 - q0_init
    count = min(len(q0), target_count)

    q0_templates[out_idx, :count] = q0[:count]
    q0_g[out_idx, :count] = g0[:count]
    q0_sensor_margins[out_idx, :count] = sensor_margins0[:count]
    q0_active_sensor[out_idx, :count] = active_sensor0[:count]
    q0_active_planes[out_idx, :count] = active_planes0[:count]
    q0_init_templates[out_idx, :count] = q0_init[:count]
    q0_init_g[out_idx, :count] = q0_init_g0[:count]
    q0_delta_q[out_idx, :count] = delta_q[:count]
    q0_delta_norm_l1[out_idx, :count] = np.sum(np.abs(delta_q[:count]), axis=1)
    q0_delta_norm_l2[out_idx, :count] = np.linalg.norm(delta_q[:count], axis=1)
    q0_source_topk_rank[out_idx, :count] = kept_candidate_indices[:count].astype(np.int32)
    q0_source_random_index[out_idx, :count] = init_indices[kept_candidate_indices[:count]].astype(np.int32)
    q0_active_sensor_occluded[out_idx, :count] = occluded[:count]
    q0_valid_with_occlusion[out_idx, :count] = ~occluded[:count]
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract M4 visibility zero-level-set configurations Q0(p)."
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--raw", required=True, help="M3 raw visibility .npz file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--joint-names", nargs="*", default=None)
    parser.add_argument("--sensor-frames", nargs="*", default=None)

    parser.add_argument("--p-count", type=int, default=200,
                        help="Number of grid points to process. Use 0 for all.")
    parser.add_argument("--p-seed", type=int, default=0)
    parser.add_argument("--q-init-per-p", type=int, default=2048)
    parser.add_argument("--optimize-top-k", type=int, default=128)
    parser.add_argument("--optimize-all-initial", action="store_true",
                        help="Optimize all q_init samples. This matches the CDF L-BFGS data-generation style.")
    parser.add_argument("--target-q0-per-p", type=int, default=32)
    parser.add_argument("--extract-per-sensor", action="store_true",
                        help="Also extract per-sensor zero-level sets Q0_s(p).")
    parser.add_argument("--per-sensor-target-q0-per-p", type=int, default=8)
    parser.add_argument("--per-sensor-optimize-top-k", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--zero-level-optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--progress-every", type=int, default=10)

    parser.add_argument("--horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--vertical-fov-deg", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--occlusion-urdf", default="",
                        help="Optional simplified collision URDF for self-occlusion checking.")
    parser.add_argument("--occlusion-backend", choices=["auto", "cpu", "torch"], default="auto",
                        help="Self-occlusion backend. auto uses torch on CUDA and CPU otherwise.")
    parser.add_argument("--keep-occluded-q0", action="store_true",
                        help="Record occlusion fields but do not filter occluded q0 candidates.")
    parser.add_argument("--ray-start-offset", type=float, default=0.03)
    parser.add_argument("--point-end-offset", type=float, default=0.005)
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--ignore-start-inside", action="store_true", default=True)
    parser.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    parser.add_argument("--ignore-links", nargs="*", default=[])

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    raw = np.load(args.raw, allow_pickle=True)
    grid_points = raw["grid_points"].astype(np.float32)
    joint_names = args.joint_names or [str(name) for name in raw.get("joint_names", DEFAULT_JOINT_NAMES)]
    sensor_frames = args.sensor_frames or [str(name) for name in raw.get("sensor_frames", DEFAULT_SENSOR_FRAMES)]

    args.horizontal_fov_deg = float(raw["horizontal_fov_deg"]) if args.horizontal_fov_deg is None else args.horizontal_fov_deg
    args.vertical_fov_deg = float(raw["vertical_fov_deg"]) if args.vertical_fov_deg is None else args.vertical_fov_deg
    args.z_min = float(raw["z_min"]) if args.z_min is None else args.z_min
    args.z_max = float(raw["z_max"]) if args.z_max is None else args.z_max
    args.delta = float(raw["delta"]) if args.delta is None else args.delta

    rng = np.random.default_rng(args.seed)
    p_rng = np.random.default_rng(args.p_seed)
    robot = URDF.from_xml_file(args.urdf)
    joint_limits = get_joint_limits(robot, joint_names)
    q_min_np = np.asarray([joint_limits[name][0] for name in joint_names], dtype=np.float32)
    q_max_np = np.asarray([joint_limits[name][1] for name in joint_names], dtype=np.float32)
    q_min = torch_tensor(q_min_np, device)
    q_max = torch_tensor(q_max_np, device)
    all_chain_specs = prepare_chain_specs(robot, args.base_frame, sensor_frames, joint_names, device)
    occlusion_context = prepare_occlusion_context(args, sensor_frames, joint_names, device)

    if args.p_count > 0:
        p_indices = p_rng.choice(len(grid_points), size=min(args.p_count, len(grid_points)), replace=False)
    else:
        p_indices = np.arange(len(grid_points), dtype=np.int64)

    k = args.target_q0_per_p
    q0_templates = np.full((len(p_indices), k, len(joint_names)), np.nan, dtype=np.float32)
    q0_g = np.full((len(p_indices), k), np.nan, dtype=np.float32)
    q0_sensor_margins = np.full((len(p_indices), k, len(sensor_frames)), np.nan, dtype=np.float32)
    q0_active_sensor = np.full((len(p_indices), k), -1, dtype=np.int16)
    q0_active_planes = np.full((len(p_indices), k, len(sensor_frames)), -1, dtype=np.int16)
    q0_init_templates = np.full((len(p_indices), k, len(joint_names)), np.nan, dtype=np.float32)
    q0_init_g = np.full((len(p_indices), k), np.nan, dtype=np.float32)
    q0_delta_q = np.full((len(p_indices), k, len(joint_names)), np.nan, dtype=np.float32)
    q0_delta_norm_l1 = np.full((len(p_indices), k), np.nan, dtype=np.float32)
    q0_delta_norm_l2 = np.full((len(p_indices), k), np.nan, dtype=np.float32)
    q0_source_topk_rank = np.full((len(p_indices), k), -1, dtype=np.int32)
    q0_source_random_index = np.full((len(p_indices), k), -1, dtype=np.int32)
    q0_active_sensor_occluded = np.zeros((len(p_indices), k), dtype=np.bool_)
    q0_valid_with_occlusion = np.zeros((len(p_indices), k), dtype=np.bool_)
    num_q0 = np.zeros((len(p_indices),), dtype=np.int32)
    num_q0_visible = np.zeros((len(p_indices),), dtype=np.int32)
    num_q0_before_occlusion = np.zeros((len(p_indices),), dtype=np.int32)
    num_q0_occluded_candidates = np.zeros((len(p_indices),), dtype=np.int32)

    ps_k = args.per_sensor_target_q0_per_p
    num_sensors = len(sensor_frames)
    if args.extract_per_sensor:
        sensor_q0_templates = np.full((len(p_indices), num_sensors, ps_k, len(joint_names)), np.nan, dtype=np.float32)
        sensor_q0_g = np.full((len(p_indices), num_sensors, ps_k), np.nan, dtype=np.float32)
        sensor_q0_sensor_margins = np.full((len(p_indices), num_sensors, ps_k, num_sensors), np.nan, dtype=np.float32)
        sensor_q0_active_planes = np.full((len(p_indices), num_sensors, ps_k, num_sensors), -1, dtype=np.int16)
        sensor_q0_init_templates = np.full((len(p_indices), num_sensors, ps_k, len(joint_names)), np.nan, dtype=np.float32)
        sensor_q0_init_g = np.full((len(p_indices), num_sensors, ps_k), np.nan, dtype=np.float32)
        sensor_q0_delta_q = np.full((len(p_indices), num_sensors, ps_k, len(joint_names)), np.nan, dtype=np.float32)
        sensor_q0_delta_norm_l1 = np.full((len(p_indices), num_sensors, ps_k), np.nan, dtype=np.float32)
        sensor_q0_delta_norm_l2 = np.full((len(p_indices), num_sensors, ps_k), np.nan, dtype=np.float32)
        sensor_q0_source_topk_rank = np.full((len(p_indices), num_sensors, ps_k), -1, dtype=np.int32)
        sensor_q0_source_random_index = np.full((len(p_indices), num_sensors, ps_k), -1, dtype=np.int32)
        sensor_q0_occluded = np.zeros((len(p_indices), num_sensors, ps_k), dtype=np.bool_)
        sensor_q0_valid_with_occlusion = np.zeros((len(p_indices), num_sensors, ps_k), dtype=np.bool_)
        num_sensor_q0 = np.zeros((len(p_indices), num_sensors), dtype=np.int32)
        num_sensor_q0_visible = np.zeros((len(p_indices), num_sensors), dtype=np.int32)
        num_sensor_q0_before_occlusion = np.zeros((len(p_indices), num_sensors), dtype=np.int32)
        num_sensor_q0_occluded_candidates = np.zeros((len(p_indices), num_sensors), dtype=np.int32)

    print("")
    print("=== M4 Visibility Zero-Level Extraction Config ===")
    print(f"raw:             {args.raw}")
    print(f"output:          {args.output}")
    print(f"device:          {device}")
    print(f"p_count:         {len(p_indices)}")
    print(f"q_init_per_p:    {args.q_init_per_p}")
    print(f"zero_optimizer:  {args.zero_level_optimizer}")
    if args.zero_level_optimizer == "lbfgs":
        print(f"lbfgs_lr:        {args.lbfgs_lr}")
    print(f"optimize_all:    {args.optimize_all_initial}")
    print(f"optimize_top_k:  {'all' if args.optimize_all_initial else args.optimize_top_k}")
    print(f"target_q0_per_p: {args.target_q0_per_p}")
    print(f"epsilon:         {args.epsilon}")
    print(f"extract_per_sensor: {args.extract_per_sensor}")
    if args.extract_per_sensor:
        print(f"per_sensor_top_k:   {args.per_sensor_optimize_top_k}")
        print(f"per_sensor_k:       {args.per_sensor_target_q0_per_p}")
    print(f"occlusion_urdf:  {args.occlusion_urdf if args.occlusion_urdf else '<none>'}")
    print(f"occlusion_backend: {occlusion_context['backend'] if occlusion_context is not None else '<none>'}")
    print(f"keep_occluded:   {args.keep_occluded_q0}")
    print(f"max_iter:        {args.max_iter}")
    print(f"lr:              {args.lr}")
    print(f"fov/range/delta: h={args.horizontal_fov_deg}, v={args.vertical_fov_deg}, z=[{args.z_min}, {args.z_max}], delta={args.delta}")

    t0 = time.time()
    for out_idx, p_idx in enumerate(p_indices):
        point = grid_points[p_idx]
        q_init = sample_q_batch(rng, joint_limits, joint_names, args.q_init_per_p)

        with torch.no_grad():
            g_init, _, _, _ = visibility_g_batch(
                torch_tensor(point, device),
                torch_tensor(q_init, device),
                all_chain_specs,
                args.horizontal_fov_deg,
                args.vertical_fov_deg,
                args.z_min,
                args.z_max,
                args.delta,
            )
            abs_g_init = torch.abs(g_init).detach().cpu().numpy()
        if args.optimize_all_initial:
            init_indices = np.arange(len(q_init), dtype=np.int64)
        else:
            top_k = min(args.optimize_top_k, len(q_init))
            init_indices_unsorted = np.argpartition(abs_g_init, top_k - 1)[:top_k]
            init_indices = init_indices_unsorted[np.argsort(abs_g_init[init_indices_unsorted])]
        q_candidates = q_init[init_indices]

        (
            q0,
            g0,
            sensor_margins0,
            active_sensor0,
            active_planes0,
            q0_init,
            q0_init_g0,
            kept_candidate_indices,
        ) = refine_q_to_zero_level(
            point,
            q_candidates,
            all_chain_specs,
            q_min,
            q_max,
            args,
            device,
        )

        if len(q0) > 0:
            if occlusion_context is not None and occlusion_context["backend"] == "torch":
                occluded = compute_q0_occlusion_torch(
                    point, q0, active_sensor0, occlusion_context, args, device
                )
            else:
                occluded = compute_q0_occlusion(
                    point, q0, active_sensor0, joint_names, occlusion_context, args
                )
            num_q0_before_occlusion[out_idx] = len(q0)
            num_q0_occluded_candidates[out_idx] = int(np.count_nonzero(occluded))
            if occlusion_context is not None and not args.keep_occluded_q0:
                keep_unoccluded = ~occluded
                q0 = q0[keep_unoccluded]
                g0 = g0[keep_unoccluded]
                sensor_margins0 = sensor_margins0[keep_unoccluded]
                active_sensor0 = active_sensor0[keep_unoccluded]
                active_planes0 = active_planes0[keep_unoccluded]
                q0_init = q0_init[keep_unoccluded]
                q0_init_g0 = q0_init_g0[keep_unoccluded]
                kept_candidate_indices = kept_candidate_indices[keep_unoccluded]
                occluded = occluded[keep_unoccluded]

            count = append_q0_block(
                out_idx,
                q0,
                g0,
                sensor_margins0,
                active_sensor0,
                active_planes0,
                q0_init,
                q0_init_g0,
                kept_candidate_indices,
                init_indices,
                k,
                args.seed + out_idx,
                q0_templates,
                q0_g,
                q0_sensor_margins,
                q0_active_sensor,
                q0_active_planes,
                q0_init_templates,
                q0_init_g,
                q0_delta_q,
                q0_delta_norm_l1,
                q0_delta_norm_l2,
                q0_source_topk_rank,
                q0_source_random_index,
                q0_active_sensor_occluded,
                q0_valid_with_occlusion,
                occluded,
            )
            num_q0[out_idx] = count
            num_q0_visible[out_idx] = int(np.count_nonzero(q0_valid_with_occlusion[out_idx, :count]))

        if args.extract_per_sensor:
            with torch.no_grad():
                _, all_sensor_margins_init, _, _ = visibility_g_batch(
                    torch_tensor(point, device),
                    torch_tensor(q_init, device),
                    all_chain_specs,
                    args.horizontal_fov_deg,
                    args.vertical_fov_deg,
                    args.z_min,
                    args.z_max,
                    args.delta,
                )
                all_sensor_margins_init = all_sensor_margins_init.detach().cpu().numpy()

            for sensor_idx in range(num_sensors):
                g_sensor_init = all_sensor_margins_init[:, sensor_idx] - args.delta
                if args.optimize_all_initial:
                    sensor_init_indices = np.arange(len(q_init), dtype=np.int64)
                else:
                    top_k_sensor = min(args.per_sensor_optimize_top_k, len(q_init))
                    sensor_init_unsorted = np.argpartition(np.abs(g_sensor_init), top_k_sensor - 1)[:top_k_sensor]
                    sensor_init_indices = sensor_init_unsorted[np.argsort(np.abs(g_sensor_init[sensor_init_unsorted]))]
                sensor_candidates = q_init[sensor_init_indices]

                (
                    sq0,
                    sg0,
                    ssensor_margins0,
                    sactive_sensor0,
                    sactive_planes0,
                    sq0_init,
                    sq0_init_g0,
                    skept_candidate_indices,
                ) = refine_q_to_zero_level(
                    point,
                    sensor_candidates,
                    all_chain_specs,
                    q_min,
                    q_max,
                    args,
                    device,
                    target_sensor=sensor_idx,
                )

                if len(sq0) == 0:
                    continue

                ssensor_indices = np.full((len(sq0),), sensor_idx, dtype=np.int16)
                if occlusion_context is not None and occlusion_context["backend"] == "torch":
                    soccluded = compute_q0_occlusion_torch(
                        point,
                        sq0,
                        ssensor_indices,
                        occlusion_context,
                        args,
                        device,
                    )
                else:
                    soccluded = compute_q0_occlusion(
                        point,
                        sq0,
                        ssensor_indices,
                        joint_names,
                        occlusion_context,
                        args,
                    )
                num_sensor_q0_before_occlusion[out_idx, sensor_idx] = len(sq0)
                num_sensor_q0_occluded_candidates[out_idx, sensor_idx] = int(np.count_nonzero(soccluded))
                if occlusion_context is not None and not args.keep_occluded_q0:
                    keep_unoccluded = ~soccluded
                    sq0 = sq0[keep_unoccluded]
                    sg0 = sg0[keep_unoccluded]
                    ssensor_margins0 = ssensor_margins0[keep_unoccluded]
                    sactive_planes0 = sactive_planes0[keep_unoccluded]
                    sq0_init = sq0_init[keep_unoccluded]
                    sq0_init_g0 = sq0_init_g0[keep_unoccluded]
                    skept_candidate_indices = skept_candidate_indices[keep_unoccluded]
                    soccluded = soccluded[keep_unoccluded]
                if len(sq0) == 0:
                    continue

                sk = args.per_sensor_target_q0_per_p
                skeep = greedy_farthest_downsample(sq0, sk, args.seed + 1009 * (out_idx + 1) + sensor_idx)
                sq0 = sq0[skeep]
                sg0 = sg0[skeep]
                ssensor_margins0 = ssensor_margins0[skeep]
                sactive_planes0 = sactive_planes0[skeep]
                sq0_init = sq0_init[skeep]
                sq0_init_g0 = sq0_init_g0[skeep]
                skept_candidate_indices = skept_candidate_indices[skeep]
                soccluded = soccluded[skeep]
                sdelta_q = sq0 - sq0_init
                scount = min(len(sq0), sk)

                sensor_q0_templates[out_idx, sensor_idx, :scount] = sq0[:scount]
                sensor_q0_g[out_idx, sensor_idx, :scount] = sg0[:scount]
                sensor_q0_sensor_margins[out_idx, sensor_idx, :scount] = ssensor_margins0[:scount]
                sensor_q0_active_planes[out_idx, sensor_idx, :scount] = sactive_planes0[:scount]
                sensor_q0_init_templates[out_idx, sensor_idx, :scount] = sq0_init[:scount]
                sensor_q0_init_g[out_idx, sensor_idx, :scount] = sq0_init_g0[:scount]
                sensor_q0_delta_q[out_idx, sensor_idx, :scount] = sdelta_q[:scount]
                sensor_q0_delta_norm_l1[out_idx, sensor_idx, :scount] = np.sum(np.abs(sdelta_q[:scount]), axis=1)
                sensor_q0_delta_norm_l2[out_idx, sensor_idx, :scount] = np.linalg.norm(sdelta_q[:scount], axis=1)
                sensor_q0_source_topk_rank[out_idx, sensor_idx, :scount] = skept_candidate_indices[:scount].astype(np.int32)
                sensor_q0_source_random_index[out_idx, sensor_idx, :scount] = sensor_init_indices[skept_candidate_indices[:scount]].astype(np.int32)
                sensor_q0_occluded[out_idx, sensor_idx, :scount] = soccluded[:scount]
                sensor_q0_valid_with_occlusion[out_idx, sensor_idx, :scount] = ~soccluded[:scount]
                num_sensor_q0[out_idx, sensor_idx] = scount
                num_sensor_q0_visible[out_idx, sensor_idx] = int(
                    np.count_nonzero(sensor_q0_valid_with_occlusion[out_idx, sensor_idx, :scount])
                )

        if args.progress_every > 0 and (out_idx + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            valid_ratio = np.mean(num_q0[:out_idx + 1] > 0)
            avg_q0 = np.mean(num_q0[:out_idx + 1])
            print(
                f"[progress] p {out_idx + 1}/{len(p_indices)} "
                f"valid_ratio {valid_ratio:.3f} avg_q0 {avg_q0:.2f} "
                f"elapsed {elapsed:.1f}s"
            )

    valid_mask = num_q0 > 0
    valid_q0_mask = np.isfinite(q0_g)
    if np.any(valid_q0_mask):
        init_abs_g_mean = float(np.mean(np.abs(q0_init_g[valid_q0_mask])))
        final_abs_g_mean = float(np.mean(np.abs(q0_g[valid_q0_mask])))
        delta_l1_mean = float(np.mean(q0_delta_norm_l1[valid_q0_mask]))
        delta_l2_mean = float(np.mean(q0_delta_norm_l2[valid_q0_mask]))
        delta_l2_max = float(np.max(q0_delta_norm_l2[valid_q0_mask]))
    else:
        init_abs_g_mean = float("nan")
        final_abs_g_mean = float("nan")
        delta_l1_mean = float("nan")
        delta_l2_mean = float("nan")
        delta_l2_max = float("nan")

    save_data = {
        "grid_points": grid_points[p_indices],
        "p_indices": p_indices,
        "q0_templates": q0_templates,
        "q0_g": q0_g,
        "q0_sensor_margins": q0_sensor_margins,
        "q0_active_sensor": q0_active_sensor,
        "q0_active_planes": q0_active_planes,
        "q0_init_templates": q0_init_templates,
        "q0_init_g": q0_init_g,
        "q0_delta_q": q0_delta_q,
        "q0_delta_norm_l1": q0_delta_norm_l1,
        "q0_delta_norm_l2": q0_delta_norm_l2,
        "q0_source_topk_rank": q0_source_topk_rank,
        "q0_source_random_index": q0_source_random_index,
        "q0_active_sensor_occluded": q0_active_sensor_occluded,
        "q0_valid_with_occlusion": q0_valid_with_occlusion,
        "num_q0": num_q0,
        "num_q0_visible": num_q0_visible,
        "num_q0_before_occlusion": num_q0_before_occlusion,
        "num_q0_occluded_candidates": num_q0_occluded_candidates,
        "valid_mask": valid_mask,
        "valid_mask_with_occlusion": num_q0_visible > 0,
        "joint_names": np.asarray(joint_names),
        "sensor_frames": np.asarray(sensor_frames),
        "horizontal_fov_deg": np.asarray(args.horizontal_fov_deg, dtype=np.float32),
        "vertical_fov_deg": np.asarray(args.vertical_fov_deg, dtype=np.float32),
        "z_min": np.asarray(args.z_min, dtype=np.float32),
        "z_max": np.asarray(args.z_max, dtype=np.float32),
        "delta": np.asarray(args.delta, dtype=np.float32),
        "epsilon": np.asarray(args.epsilon, dtype=np.float32),
        "q_init_per_p": np.asarray(args.q_init_per_p, dtype=np.int32),
        "optimize_top_k": np.asarray(args.optimize_top_k, dtype=np.int32),
        "optimize_all_initial": np.asarray(args.optimize_all_initial, dtype=np.bool_),
        "zero_level_optimizer": np.asarray(args.zero_level_optimizer),
        "lbfgs_lr": np.asarray(args.lbfgs_lr, dtype=np.float32),
        "target_q0_per_p": np.asarray(args.target_q0_per_p, dtype=np.int32),
        "extract_per_sensor": np.asarray(args.extract_per_sensor, dtype=np.bool_),
        "per_sensor_optimize_top_k": np.asarray(args.per_sensor_optimize_top_k, dtype=np.int32),
        "per_sensor_target_q0_per_p": np.asarray(args.per_sensor_target_q0_per_p, dtype=np.int32),
        "occlusion_urdf": np.asarray(args.occlusion_urdf),
        "occlusion_backend": np.asarray(occlusion_context["backend"] if occlusion_context is not None else "none"),
        "keep_occluded_q0": np.asarray(args.keep_occluded_q0, dtype=np.bool_),
        "raw_has_occlusion_labels": np.asarray(
            bool(raw["has_occlusion_labels"]) if "has_occlusion_labels" in raw else False,
            dtype=np.bool_,
        ),
        "seed": np.asarray(args.seed, dtype=np.int32),
        "p_seed": np.asarray(args.p_seed, dtype=np.int32),
    }
    if args.extract_per_sensor:
        save_data.update({
            "sensor_q0_templates": sensor_q0_templates,
            "sensor_q0_g": sensor_q0_g,
            "sensor_q0_sensor_margins": sensor_q0_sensor_margins,
            "sensor_q0_active_planes": sensor_q0_active_planes,
            "sensor_q0_init_templates": sensor_q0_init_templates,
            "sensor_q0_init_g": sensor_q0_init_g,
            "sensor_q0_delta_q": sensor_q0_delta_q,
            "sensor_q0_delta_norm_l1": sensor_q0_delta_norm_l1,
            "sensor_q0_delta_norm_l2": sensor_q0_delta_norm_l2,
            "sensor_q0_source_topk_rank": sensor_q0_source_topk_rank,
            "sensor_q0_source_random_index": sensor_q0_source_random_index,
            "sensor_q0_occluded": sensor_q0_occluded,
            "sensor_q0_valid_with_occlusion": sensor_q0_valid_with_occlusion,
            "num_sensor_q0": num_sensor_q0,
            "num_sensor_q0_visible": num_sensor_q0_visible,
            "num_sensor_q0_before_occlusion": num_sensor_q0_before_occlusion,
            "num_sensor_q0_occluded_candidates": num_sensor_q0_occluded_candidates,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save_data)

    elapsed = time.time() - t0
    print("")
    print("=== M4 Summary ===")
    print(f"processed_p:    {len(p_indices)}")
    print(f"valid_p:        {int(valid_mask.sum())}")
    print(f"valid_ratio:    {valid_mask.mean():.6f}")
    print(f"avg_q0_per_p:   {num_q0.mean():.3f}")
    print(f"avg_visible_q0: {num_q0_visible.mean():.3f}")
    print(f"max_q0_per_p:   {num_q0.max() if len(num_q0) else 0}")
    if np.any(np.isfinite(q0_g)):
        global_stored = int(np.count_nonzero(np.isfinite(q0_g)))
        global_visible = int(np.count_nonzero(q0_valid_with_occlusion & np.isfinite(q0_g)))
        global_occluded = int(np.count_nonzero(q0_active_sensor_occluded & np.isfinite(q0_g)))
        global_pre_filter = int(np.sum(num_q0_before_occlusion))
        global_filtered_occ = int(np.sum(num_q0_occluded_candidates))
        print(f"stored global q0: {global_stored}")
        print(f"visible global q0:{global_visible}")
        print(f"occluded global q0:{global_occluded}")
        print(f"pre-filter global q0:{global_pre_filter}")
        print(f"occluded global candidates:{global_filtered_occ}")
    if args.extract_per_sensor:
        sensor_valid_mask = np.isfinite(sensor_q0_g)
        sensor_stored = int(np.count_nonzero(sensor_valid_mask))
        sensor_visible = int(np.count_nonzero(sensor_q0_valid_with_occlusion & sensor_valid_mask))
        sensor_occluded_count = int(np.count_nonzero(sensor_q0_occluded & sensor_valid_mask))
        print(f"stored sensor q0: {sensor_stored}")
        print(f"visible sensor q0:{sensor_visible}")
        print(f"occluded sensor q0:{sensor_occluded_count}")
        print(f"pre-filter sensor q0:{int(np.sum(num_sensor_q0_before_occlusion))}")
        print(f"occluded sensor candidates:{int(np.sum(num_sensor_q0_occluded_candidates))}")
        print("avg visible sensor q0 per p/sensor:")
        for sensor_idx, frame in enumerate(sensor_frames):
            print(f"  {sensor_idx}: {frame:28s} {num_sensor_q0_visible[:, sensor_idx].mean():.3f}")
    print(f"mean |g_init|:  {init_abs_g_mean:.6f}")
    print(f"mean |g_final|: {final_abs_g_mean:.6f}")
    print(f"mean ||dq||_1:  {delta_l1_mean:.6f}")
    print(f"mean ||dq||_2:  {delta_l2_mean:.6f}")
    print(f"max ||dq||_2:   {delta_l2_max:.6f}")
    print(f"elapsed_sec:    {elapsed:.2f}")
    print(f"saved:          {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""URDF-solid backend for offline goal/q0 tools; explicit legacy compatibility.

Coal evaluates primitive pairs, Pinocchio supplies FK. Inflations are Euclidean
distance offsets, never independent axis growth or bounding-sphere substitution.
No change to historical case files or production prior configuration.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_confidence_map/scripts'))
from primitive_point_geometry import load_primitives


@dataclass
class BodyPrimitive:
    link_name: str
    frame_name: str
    center_link: np.ndarray
    rotation_link: np.ndarray
    shape: object
    inflation: float
    collision_index: int
    kind: str
    dimensions: tuple


def add_geometry_arguments(parser):
    parser.add_argument('--geometry-backend', choices=['samples', 'primitive'], default='samples')
    parser.add_argument('--primitive-urdf', type=Path, default=None)


def selected_body_path(args, repo, samples_path):
    return ((args.primitive_urdf or repo/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf').resolve()
            if args.geometry_backend == 'primitive' else samples_path)


def load_body_primitives(path, inflation, risk_only=True):
    import coal
    if not math.isfinite(inflation) or inflation < 0:
        raise ValueError('Invalid body inflation')
    # Preserve these tools' old include_for_risk policy (base excluded, link1 included).
    model = load_primitives(path, ['base_link'] if risk_only else [])
    out = []
    for frame, primitives in model.items():
        for p in primitives:
            shape = {'box': coal.Box, 'cylinder': coal.Cylinder, 'sphere': coal.Sphere}[p.kind](*p.dimensions)
            out.append(BodyPrimitive(frame, frame, np.array(p.center), np.array(p.rotation), shape,
                                     float(inflation), p.collision_index, p.kind, p.dimensions))
    return out


def world_primitives(model, q, primitives):
    import pinocchio as pin
    q = np.asarray(q, dtype=float)
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError('Invalid q')
    data = model.createData(); pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    out = []
    for p in primitives:
        fid = model.getFrameId(p.frame_name)
        if fid >= len(model.frames):
            raise ValueError('Missing primitive frame '+p.frame_name)
        pose = data.oMf[fid]
        out.append((np.asarray(pose.rotation@p.rotation_link), np.asarray(pose.translation+pose.rotation@p.center_link)))
    return out


def pair_clearance(a, pose_a, b_shape, pose_b, inflation_b=0.):
    import coal
    request = coal.DistanceRequest(); request.enable_signed_distance = True
    result = coal.DistanceResult()
    value = coal.distance(a.shape, coal.Transform3s(*pose_a), b_shape, coal.Transform3s(*pose_b), request, result)
    value = float(value)-a.inflation-inflation_b
    if not math.isfinite(value):
        raise ValueError('Nonfinite primitive distance; no fallback')
    return value


def terminal_primitive_clearance(model, q, primitives, boxes):
    import coal
    if not primitives or not boxes:
        raise ValueError('Empty geometry cannot certify clearance')
    best, info = math.inf, {}
    for p, pose in zip(primitives, world_primitives(model, q, primitives)):
        for box in boxes:
            value = pair_clearance(p, pose, coal.Box(*box.size), (box.rotation_world_box, box.center_world))
            if value < best:
                best, info = value, dict(geometry_backend='primitive', link_name=p.link_name,
                    collision_index=p.collision_index, primitive_kind=p.kind, obstacle=box.name, clearance_m=value)
    return best, info


def primitive_self_clearance(primitives, poses, link_order):
    if not primitives or len(primitives) != len(poses):
        raise ValueError('Empty or incomplete primitive poses')
    best, info = math.inf, {}
    for i, a in enumerate(primitives):
        ai = link_order.get(a.link_name)
        for j in range(i+1, len(primitives)):
            b = primitives[j]; bi = link_order.get(b.link_name)
            if ai is None or bi is None or abs(ai-bi) <= 1:
                continue  # Preserve original nonadjacent-link selection policy.
            value = pair_clearance(a, poses[i], b.shape, poses[j], b.inflation)
            if value < best:
                best, info = value, dict(geometry_backend='primitive', link_a=a.link_name, link_b=b.link_name,
                    collision_a=a.collision_index, collision_b=b.collision_index, clearance_m=value)
    return best, info


def aabb_extent(primitive, rotation):
    if primitive.kind == 'box':
        extent = np.abs(rotation)@(np.asarray(primitive.dimensions)/2)
    elif primitive.kind == 'cylinder':
        r, length = primitive.dimensions
        extent = r*np.sqrt(rotation[:, 0]**2+rotation[:, 1]**2)+length/2*np.abs(rotation[:, 2])
    else:
        extent = np.full(3, primitive.dimensions[0])
    return extent+primitive.inflation


def primitive_inside_workspace(primitives, poses, bounds, margin):
    if not primitives or len(primitives) != len(poses):
        raise ValueError('Empty or incomplete primitive poses')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid workspace margin')
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (6,):
        raise ValueError('Workspace requires six bounds')
    lower, upper = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
    if not np.all(np.isfinite(bounds)) or np.any(lower >= upper):
        raise ValueError('Invalid workspace bounds')
    for p, (rotation, center) in zip(primitives, poses):
        if p.link_name in {'base_link', 'link1'}:
            continue
        extent = aabb_extent(p, rotation)+margin
        if np.any(center-extent < lower) or np.any(center+extent > upper):
            return False
    return True

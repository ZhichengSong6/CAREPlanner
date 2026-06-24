#!/usr/bin/env python3

import argparse
import math
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from urdf_parser_py.urdf import URDF


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

PLANE_NAMES = ["left", "right", "bottom", "top", "near", "far"]


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])
    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rz @ ry @ rx


def make_transform(xyz: List[float], rpy: List[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_rot(rpy[0], rpy[1], rpy[2])
    transform[:3, 3] = np.array(xyz, dtype=float)
    return transform


def axis_angle_to_rot(axis: np.ndarray, q: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)

    axis = axis / norm
    x, y, z = axis
    c = math.cos(q)
    s = math.sin(q)
    one_minus_c = 1.0 - c

    return np.array([
        [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
    ])


def joint_origin_transform(joint) -> np.ndarray:
    if joint.origin is None:
        return np.eye(4)

    xyz = joint.origin.xyz if joint.origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = joint.origin.rpy if joint.origin.rpy is not None else [0.0, 0.0, 0.0]
    return make_transform(xyz, rpy)


def joint_motion_transform(joint, q: float) -> np.ndarray:
    transform = np.eye(4)
    axis = np.array(joint.axis if joint.axis is not None else [1.0, 0.0, 0.0], dtype=float)

    if joint.type in ["revolute", "continuous"]:
        transform[:3, :3] = axis_angle_to_rot(axis, q)
    elif joint.type == "prismatic":
        norm = np.linalg.norm(axis)
        if norm > 1e-12:
            axis = axis / norm
        transform[:3, 3] = axis * q

    return transform


def find_chain_joints(robot: URDF, base_link: str, target_link: str):
    parent_joint_of_child = {joint.child: joint for joint in robot.joints}
    chain_reversed = []
    current = target_link

    while current != base_link:
        if current not in parent_joint_of_child:
            raise RuntimeError(
                f"Cannot find chain from {base_link} to {target_link}; stopped at {current}"
            )
        joint = parent_joint_of_child[current]
        chain_reversed.append(joint)
        current = joint.parent

    return list(reversed(chain_reversed))


def get_joint_limits(robot: URDF, joint_names: List[str]) -> Dict[str, Tuple[float, float]]:
    limits = {}
    joint_by_name = {joint.name: joint for joint in robot.joints}

    for name in joint_names:
        joint = joint_by_name[name]
        if joint.type == "continuous":
            limits[name] = (-math.pi, math.pi)
        elif joint.type in ["revolute", "prismatic"]:
            if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
                raise RuntimeError(f"Joint {name} has incomplete limits.")
            limits[name] = (float(joint.limit.lower), float(joint.limit.upper))
        else:
            limits[name] = (0.0, 0.0)

    return limits


def fk_transform(chain_joints, q_map: Dict[str, float]) -> np.ndarray:
    transform = np.eye(4)
    for joint in chain_joints:
        q = q_map.get(joint.name, 0.0)
        transform = transform @ joint_origin_transform(joint) @ joint_motion_transform(joint, q)
    return transform


def sample_points(bounds: np.ndarray, grid_shape: Tuple[int, int, int], random_count: int, seed: int):
    rng = np.random.default_rng(seed)

    if random_count > 0:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return rng.uniform(lower, upper, size=(random_count, 3))

    xs = np.linspace(bounds[0, 0], bounds[0, 1], grid_shape[0])
    ys = np.linspace(bounds[1, 0], bounds[1, 1], grid_shape[1])
    zs = np.linspace(bounds[2, 0], bounds[2, 1], grid_shape[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def sample_q(joint_limits: Dict[str, Tuple[float, float]], joint_names: List[str], seed: int):
    rng = np.random.default_rng(seed)
    q_map = {}
    for name in joint_names:
        lower, upper = joint_limits[name]
        q_map[name] = float(rng.uniform(lower, upper))
    return q_map


def q_map_from_values(joint_names: List[str], q_values: List[float]):
    if len(q_values) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} q values, got {len(q_values)}")
    return {name: float(value) for name, value in zip(joint_names, q_values)}


def fov_plane_margins(point_sensor: np.ndarray, fov_x_deg: float, fov_y_deg: float,
                      z_min: float, z_max: float) -> np.ndarray:
    x, y, z = point_sensor
    ax = math.tan(math.radians(fov_x_deg) * 0.5)
    ay = math.tan(math.radians(fov_y_deg) * 0.5)
    nx = math.sqrt(1.0 + ax * ax)
    ny = math.sqrt(1.0 + ay * ay)

    return np.array([
        (x + z * ax) / nx,
        (-x + z * ax) / nx,
        (y + z * ay) / ny,
        (-y + z * ay) / ny,
        z - z_min,
        z_max - z,
    ])


def sensor_margin(point_base: np.ndarray, transform_base_sensor: np.ndarray,
                  fov_x_deg: float, fov_y_deg: float, z_min: float, z_max: float):
    point_h = np.array([point_base[0], point_base[1], point_base[2], 1.0])
    point_sensor = np.linalg.inv(transform_base_sensor) @ point_h
    margins = fov_plane_margins(point_sensor[:3], fov_x_deg, fov_y_deg, z_min, z_max)
    active_plane = int(np.argmin(margins))
    return float(margins[active_plane]), active_plane, point_sensor[:3]


def compute_visibility(points: np.ndarray, sensor_transforms: List[np.ndarray], args):
    all_margins = np.zeros((len(points), len(sensor_transforms)), dtype=np.float64)
    active_planes = np.zeros((len(points), len(sensor_transforms)), dtype=np.int64)

    for point_idx, point in enumerate(points):
        for sensor_idx, transform in enumerate(sensor_transforms):
            margin, plane_idx, _ = sensor_margin(
                point, transform, args.horizontal_fov_deg, args.vertical_fov_deg,
                args.z_min, args.z_max
            )
            all_margins[point_idx, sensor_idx] = margin
            active_planes[point_idx, sensor_idx] = plane_idx

    best_sensor = np.argmax(all_margins, axis=1)
    best_margin = all_margins[np.arange(len(points)), best_sensor]
    g = best_margin - args.delta
    visible = g >= 0.0
    return all_margins, active_planes, best_sensor, best_margin, g, visible


def print_stats(sensor_margins, best_sensor, best_margin, g, visible, sensor_frames):
    print("")
    print("=== Visibility Oracle Statistics ===")
    print(f"num_points:      {len(g)}")
    print(f"visible_points:  {int(visible.sum())}")
    print(f"visible_ratio:   {visible.mean():.6f}")
    print(f"best_margin min/mean/max: {best_margin.min(): .6f} / {best_margin.mean(): .6f} / {best_margin.max(): .6f}")
    print(f"g min/mean/max:           {g.min(): .6f} / {g.mean(): .6f} / {g.max(): .6f}")
    print(f"sensor_margins min/mean/max: {sensor_margins.min(): .6f} / {sensor_margins.mean(): .6f} / {sensor_margins.max(): .6f}")

    print("")
    print("best_sensor histogram:")
    counts = Counter(best_sensor.tolist())
    for idx, frame in enumerate(sensor_frames):
        print(f"  {idx}: {frame:28s} {counts.get(idx, 0):8d}")


def print_axis_sanity(sensor_transforms, sensor_frames, args):
    print("")
    print("=== Sensor Axis Sanity Check ===")
    z_mid = 0.5 * (args.z_min + args.z_max)
    for idx, transform in enumerate(sensor_transforms):
        front_sensor = np.array([0.0, 0.0, z_mid, 1.0])
        back_sensor = np.array([0.0, 0.0, -z_mid, 1.0])
        front_base = (transform @ front_sensor)[:3]
        back_base = (transform @ back_sensor)[:3]

        front_margin, front_plane, _ = sensor_margin(
            front_base, transform, args.horizontal_fov_deg, args.vertical_fov_deg,
            args.z_min, args.z_max
        )
        back_margin, back_plane, _ = sensor_margin(
            back_base, transform, args.horizontal_fov_deg, args.vertical_fov_deg,
            args.z_min, args.z_max
        )
        print(
            f"  {idx}: {sensor_frames[idx]:28s} "
            f"front_margin={front_margin: .6f} ({PLANE_NAMES[front_plane]}) "
            f"back_margin={back_margin: .6f} ({PLANE_NAMES[back_plane]})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Validate analytic FOV visibility margins for CARE visibility CDF."
    )
    parser.add_argument("--urdf", required=True, help="Path to robot URDF.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--joint-names", nargs="*", default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--sensor-frames", nargs="*", default=DEFAULT_SENSOR_FRAMES)

    parser.add_argument("--q", nargs="*", type=float, default=None,
                        help="Seven joint values. If omitted, all joints are zero.")
    parser.add_argument("--random-q", action="store_true",
                        help="Use one random joint configuration instead of zero q.")

    parser.add_argument("--x-min", type=float, default=-0.95)
    parser.add_argument("--x-max", type=float, default=0.95)
    parser.add_argument("--y-min", type=float, default=-0.95)
    parser.add_argument("--y-max", type=float, default=0.95)
    parser.add_argument("--z-min-bound", type=float, default=0.0)
    parser.add_argument("--z-max-bound", type=float, default=1.15)

    parser.add_argument("--grid", nargs=3, type=int, default=[30, 30, 20],
                        help="Grid shape for p samples. Ignored if --random-points > 0.")
    parser.add_argument("--random-points", type=int, default=0,
                        help="If positive, sample this many random p instead of a grid.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=66.0)
    parser.add_argument("--z-min", type=float, default=0.20,
                        help="Conservative ToF near range in sensor frame.")
    parser.add_argument("--z-max", type=float, default=0.70,
                        help="Conservative ToF far range in sensor frame.")
    parser.add_argument("--delta", type=float, default=0.02,
                        help="Positive visibility margin required for g >= 0.")
    parser.add_argument("--save", default="", help="Optional .npz path for raw M2 samples.")

    args = parser.parse_args()

    robot = URDF.from_xml_file(args.urdf)
    joint_limits = get_joint_limits(robot, args.joint_names)

    if args.q is not None:
        q_map = q_map_from_values(args.joint_names, args.q)
    elif args.random_q:
        q_map = sample_q(joint_limits, args.joint_names, args.seed)
    else:
        q_map = {name: 0.0 for name in args.joint_names}

    sensor_chains = [
        find_chain_joints(robot, args.base_frame, frame)
        for frame in args.sensor_frames
    ]
    sensor_transforms = [fk_transform(chain, q_map) for chain in sensor_chains]

    bounds = np.array([
        [args.x_min, args.x_max],
        [args.y_min, args.y_max],
        [args.z_min_bound, args.z_max_bound],
    ])
    points = sample_points(bounds, tuple(args.grid), args.random_points, args.seed)

    sensor_margins, active_planes, best_sensor, best_margin, g, visible = compute_visibility(
        points, sensor_transforms, args
    )

    print("")
    print("=== Visibility Oracle Config ===")
    print(f"urdf:       {args.urdf}")
    print(f"base_frame: {args.base_frame}")
    print(f"bounds:     x[{args.x_min}, {args.x_max}], y[{args.y_min}, {args.y_max}], z[{args.z_min_bound}, {args.z_max_bound}]")
    print(f"fov:        h={args.horizontal_fov_deg} deg, v={args.vertical_fov_deg} deg")
    print(f"range:      z_min={args.z_min}, z_max={args.z_max}")
    print(f"delta:      {args.delta}")
    print("q:")
    for name in args.joint_names:
        print(f"  {name:20s}: {q_map[name]: .6f}")

    print_stats(sensor_margins, best_sensor, best_margin, g, visible, args.sensor_frames)
    print_axis_sanity(sensor_transforms, args.sensor_frames, args)

    if args.save:
        np.savez_compressed(
            args.save,
            p=points,
            q=np.array([q_map[name] for name in args.joint_names], dtype=np.float64),
            sensor_margins=sensor_margins,
            active_planes=active_planes,
            best_sensor=best_sensor,
            best_margin=best_margin,
            g=g,
            visible=visible,
            sensor_frames=np.array(args.sensor_frames),
            joint_names=np.array(args.joint_names),
        )
        print("")
        print(f"saved: {args.save}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

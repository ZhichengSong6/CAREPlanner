#!/usr/bin/env python3

import argparse
import math
import sys
import random
from typing import Dict, List, Tuple

import numpy as np
import rospy
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
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def axis_angle_to_rot(axis: np.ndarray, q: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)

    axis = axis / norm
    x, y, z = axis

    c = math.cos(q)
    s = math.sin(q)
    C = 1.0 - c

    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def joint_motion_transform(joint, q: float) -> np.ndarray:
    T = np.eye(4)

    axis = np.array(joint.axis if joint.axis is not None else [1.0, 0.0, 0.0], dtype=float)

    if joint.type in ["revolute", "continuous"]:
        T[:3, :3] = axis_angle_to_rot(axis, q)
    elif joint.type == "prismatic":
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-12:
            axis = axis / axis_norm
        T[:3, 3] = axis * q
    elif joint.type == "fixed":
        pass
    else:
        # floating / planar 等这里不处理
        pass

    return T


def get_joint_origin_transform(joint) -> np.ndarray:
    if joint.origin is None:
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
    else:
        xyz = joint.origin.xyz if joint.origin.xyz is not None else [0.0, 0.0, 0.0]
        rpy = joint.origin.rpy if joint.origin.rpy is not None else [0.0, 0.0, 0.0]

    return make_transform(xyz, rpy)


def load_robot(args) -> URDF:
    if args.urdf:
        return URDF.from_xml_file(args.urdf)

    rospy.init_node("compute_ee_workspace_bounds", anonymous=True, disable_signals=True)
    robot_description = rospy.get_param(args.robot_description_param, None)

    if robot_description is None:
        raise RuntimeError(
            f"Could not find robot description param: {args.robot_description_param}. "
            f"Either start robot_state_publisher/Gazebo first, or pass --urdf /path/to/robot.urdf"
        )

    return URDF.from_xml_string(robot_description)


def find_chain_joints(robot: URDF, base_link: str, ee_link: str):
    parent_joint_of_child = {}

    for joint in robot.joints:
        parent_joint_of_child[joint.child] = joint

    chain_reversed = []
    current = ee_link

    while current != base_link:
        if current not in parent_joint_of_child:
            raise RuntimeError(
                f"Cannot find chain from {base_link} to {ee_link}. "
                f"Stopped at link: {current}"
            )

        joint = parent_joint_of_child[current]
        chain_reversed.append(joint)
        current = joint.parent

    return list(reversed(chain_reversed))


def get_joint_limits(joint, user_limits: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    if joint.name in user_limits:
        return user_limits[joint.name]

    if joint.type == "continuous":
        return -math.pi, math.pi

    if joint.type in ["revolute", "prismatic"]:
        if joint.limit is None:
            raise RuntimeError(f"Joint {joint.name} has no limit in URDF.")

        lower = joint.limit.lower
        upper = joint.limit.upper

        if lower is None or upper is None:
            raise RuntimeError(f"Joint {joint.name} limit is incomplete in URDF.")

        return float(lower), float(upper)

    return 0.0, 0.0


def parse_user_joint_limits(items: List[str]) -> Dict[str, Tuple[float, float]]:
    limits = {}

    for item in items:
        # format: joint_name:lower:upper
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid joint limit override: {item}. "
                f"Expected format: joint_name:lower:upper"
            )

        name = parts[0]
        lower = float(parts[1])
        upper = float(parts[2])
        limits[name] = (lower, upper)

    return limits


def sample_joint_values(
    active_chain_joints,
    joint_names: List[str],
    joint_limits: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    q_map = {}

    for joint in active_chain_joints:
        if joint.name in joint_names:
            lower, upper = joint_limits[joint.name]
            q_map[joint.name] = random.uniform(lower, upper)

    return q_map


def fk_ee_position(chain_joints, q_map: Dict[str, float]) -> np.ndarray:
    T = np.eye(4)

    for joint in chain_joints:
        T_origin = get_joint_origin_transform(joint)
        q = q_map.get(joint.name, 0.0)
        T_motion = joint_motion_transform(joint, q)
        T = T @ T_origin @ T_motion

    return T[:3, 3].copy()


def add_limit_corner_samples(
    chain_joints,
    active_joint_names: List[str],
    joint_limits: Dict[str, Tuple[float, float]],
    max_corners: int,
):
    n = len(active_joint_names)
    total_corners = 2 ** n

    samples = []

    if total_corners <= max_corners:
        for mask in range(total_corners):
            q_map = {}
            for i, name in enumerate(active_joint_names):
                lower, upper = joint_limits[name]
                q_map[name] = upper if ((mask >> i) & 1) else lower
            samples.append(q_map)
    else:
        # 如果自由度太多，就随机抽一些 limit corners
        for _ in range(max_corners):
            q_map = {}
            for name in active_joint_names:
                lower, upper = joint_limits[name]
                q_map[name] = upper if random.random() > 0.5 else lower
            samples.append(q_map)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Offline URDF sampling tool for EE workspace bounding box."
    )

    parser.add_argument("--urdf", type=str, default="",
                        help="Path to URDF file. If empty, read from ROS param.")
    parser.add_argument("--robot-description-param", type=str, default="/robot_description",
                        help="ROS parameter name for robot_description.")

    parser.add_argument("--base-frame", type=str, default="base_link")
    parser.add_argument("--ee-frame", type=str, default="EE_link")

    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.10)

    parser.add_argument("--joint-names", nargs="*", default=DEFAULT_JOINT_NAMES,
                        help="Controlled joint names to sample.")
    parser.add_argument("--joint-limit", action="append", default=[],
                        help="Override joint limit. Format: joint_name:lower:upper")

    parser.add_argument("--max-corner-samples", type=int, default=512,
                        help="Also sample joint limit corners, up to this number.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    user_limits = parse_user_joint_limits(args.joint_limit)

    robot = load_robot(args)

    chain_joints = find_chain_joints(robot, args.base_frame, args.ee_frame)

    chain_joint_names = [j.name for j in chain_joints]
    active_joint_names = [
        name for name in args.joint_names
        if name in chain_joint_names
    ]

    if len(active_joint_names) == 0:
        raise RuntimeError(
            "No active joint names are on the chain. "
            f"Given joint_names={args.joint_names}, chain_joints={chain_joint_names}"
        )

    joint_limits = {}
    for joint in chain_joints:
        if joint.name in active_joint_names:
            joint_limits[joint.name] = get_joint_limits(joint, user_limits)

    print("")
    print("=== EE workspace bounds sampling ===")
    print(f"base_frame: {args.base_frame}")
    print(f"ee_frame:   {args.ee_frame}")
    print(f"samples:    {args.samples}")
    print(f"margin:     {args.margin}")
    print("")
    print("chain joints:")
    for joint in chain_joints:
        flag = "sampled" if joint.name in active_joint_names else "fixed/passive"
        print(f"  {joint.name:20s} type={joint.type:10s} parent={joint.parent:20s} child={joint.child:20s} [{flag}]")

    print("")
    print("sampled joint limits:")
    for name in active_joint_names:
        lower, upper = joint_limits[name]
        print(f"  {name:20s}: [{lower:.6f}, {upper:.6f}]")

    points = []

    # Add random samples
    for i in range(args.samples):
        q_map = sample_joint_values(chain_joints, active_joint_names, joint_limits)
        p = fk_ee_position(chain_joints, q_map)
        points.append(p)

    # Add joint-limit corner samples
    corner_samples = add_limit_corner_samples(
        chain_joints,
        active_joint_names,
        joint_limits,
        args.max_corner_samples,
    )

    for q_map in corner_samples:
        p = fk_ee_position(chain_joints, q_map)
        points.append(p)

    points = np.array(points)

    p_min = points.min(axis=0)
    p_max = points.max(axis=0)

    p_min_margin = p_min - args.margin
    p_max_margin = p_max + args.margin

    center = 0.5 * (p_min_margin + p_max_margin)
    size = p_max_margin - p_min_margin

    print("")
    print("=== Raw EE workspace bounds, without margin ===")
    print(f"x: [{p_min[0]: .6f}, {p_max[0]: .6f}]")
    print(f"y: [{p_min[1]: .6f}, {p_max[1]: .6f}]")
    print(f"z: [{p_min[2]: .6f}, {p_max[2]: .6f}]")

    print("")
    print("=== Confidence map bounds, with margin ===")
    print(f"x: [{p_min_margin[0]: .6f}, {p_max_margin[0]: .6f}]")
    print(f"y: [{p_min_margin[1]: .6f}, {p_max_margin[1]: .6f}]")
    print(f"z: [{p_min_margin[2]: .6f}, {p_max_margin[2]: .6f}]")

    print("")
    print("=== YAML snippet ===")
    print("confidence_map:")
    print(f"  x_min: {p_min_margin[0]:.6f}")
    print(f"  x_max: {p_max_margin[0]:.6f}")
    print(f"  y_min: {p_min_margin[1]:.6f}")
    print(f"  y_max: {p_max_margin[1]:.6f}")
    print(f"  z_min: {p_min_margin[2]:.6f}")
    print(f"  z_max: {p_max_margin[2]:.6f}")
    print("  # Suggested values; adjust resolution later for runtime cost.")
    print("  resolution: 0.08")

    print("")
    print("=== Box info ===")
    print(f"center: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print(f"size:   [{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]")
    print(f"num_points_used: {len(points)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("")
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
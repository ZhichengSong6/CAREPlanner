#!/usr/bin/env python3

import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from urdf_parser_py.urdf import URDF
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PLANE_NAMES,
    compute_visibility,
    find_chain_joints,
    fk_transform,
)


def names_from_npz(data, key, default):
    if key in data:
        return [str(name) for name in data[key]]
    return list(default)


def q_row_to_map(joint_names, q_row):
    return {name: float(value) for name, value in zip(joint_names, q_row)}


def transform_point(transform, point_local):
    point_h = np.array([point_local[0], point_local[1], point_local[2], 1.0], dtype=np.float64)
    return (transform @ point_h)[:3]


def to_point(xyz):
    point = Point()
    point.x = float(xyz[0])
    point.y = float(xyz[1])
    point.z = float(xyz[2])
    return point


def color(r, g, b, a):
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def make_marker(marker_id, ns, frame_id, marker_type):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


def make_delete_all(frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.action = Marker.DELETEALL
    return marker


def fov_corners(z, ax, ay):
    return [
        np.array([-z * ax, -z * ay, z], dtype=np.float64),
        np.array([ z * ax, -z * ay, z], dtype=np.float64),
        np.array([ z * ax,  z * ay, z], dtype=np.float64),
        np.array([-z * ax,  z * ay, z], dtype=np.float64),
    ]


def add_line(points, a, b):
    points.append(to_point(a))
    points.append(to_point(b))


def make_fov_marker(marker_id, frame_id, transform_base_sensor, args, active=False):
    ax = math.tan(math.radians(args.horizontal_fov_deg) * 0.5)
    ay = math.tan(math.radians(args.vertical_fov_deg) * 0.5)
    near = [transform_point(transform_base_sensor, p) for p in fov_corners(args.z_min, ax, ay)]
    far = [transform_point(transform_base_sensor, p) for p in fov_corners(args.z_max, ax, ay)]
    origin = transform_point(transform_base_sensor, [0.0, 0.0, 0.0])

    marker = make_marker(marker_id, "visibility_q0/fov", args.base_frame, Marker.LINE_LIST)
    marker.scale.x = 0.006 if active else 0.002
    marker.color = color(1.0, 0.82, 0.05, 0.95) if active else color(0.55, 0.55, 0.55, 0.28)

    for i in range(4):
        add_line(marker.points, near[i], near[(i + 1) % 4])
        add_line(marker.points, far[i], far[(i + 1) % 4])
        add_line(marker.points, near[i], far[i])
    for p in far:
        add_line(marker.points, origin, p)
    return marker


def make_point_marker(marker_id, frame_id, point_base, text="query point"):
    marker = make_marker(marker_id, "visibility_q0/query_point", frame_id, Marker.SPHERE)
    marker.pose.position = to_point(point_base)
    marker.scale.x = 0.045
    marker.scale.y = 0.045
    marker.scale.z = 0.045
    marker.color = color(0.1, 1.0, 0.2, 0.95)

    text_marker = make_marker(marker_id + 1, "visibility_q0/query_text", frame_id, Marker.TEXT_VIEW_FACING)
    text_marker.pose.position = to_point(point_base + np.array([0.0, 0.0, 0.07]))
    text_marker.scale.z = 0.045
    text_marker.color = color(1.0, 1.0, 1.0, 0.95)
    text_marker.text = text
    return [marker, text_marker]


def make_sensor_axes(marker_id, frame_id, transform_base_sensor, scale=0.12):
    origin = transform_point(transform_base_sensor, [0.0, 0.0, 0.0])
    x_end = transform_point(transform_base_sensor, [scale, 0.0, 0.0])
    y_end = transform_point(transform_base_sensor, [0.0, scale, 0.0])
    z_end = transform_point(transform_base_sensor, [0.0, 0.0, scale])

    marker = make_marker(marker_id, "visibility_q0/active_sensor_axes", frame_id, Marker.LINE_LIST)
    marker.scale.x = 0.01
    marker.points = [
        to_point(origin), to_point(x_end),
        to_point(origin), to_point(y_end),
        to_point(origin), to_point(z_end),
    ]
    marker.colors = [
        color(1.0, 0.0, 0.0, 1.0), color(1.0, 0.0, 0.0, 1.0),
        color(0.0, 1.0, 0.0, 1.0), color(0.0, 1.0, 0.0, 1.0),
        color(0.1, 0.4, 1.0, 1.0), color(0.1, 0.4, 1.0, 1.0),
    ]
    return marker


def make_sensor_text(marker_id, frame_id, transform_base_sensor, text):
    origin = transform_point(transform_base_sensor, [0.0, 0.0, 0.0])
    marker = make_marker(marker_id, "visibility_q0/active_sensor_text", frame_id, Marker.TEXT_VIEW_FACING)
    marker.pose.position = to_point(origin + np.array([0.0, 0.0, 0.08]))
    marker.scale.z = 0.04
    marker.color = color(1.0, 0.82, 0.05, 0.95)
    marker.text = text
    return marker


def load_valid_entries(q0_data):
    grid_points = q0_data["grid_points"].astype(np.float64)
    q0 = q0_data["q0_templates"].astype(np.float64)
    q0_g = q0_data["q0_g"]
    active_sensor = q0_data["q0_active_sensor"]
    valid = np.isfinite(q0_g) & (active_sensor >= 0)
    pi, qi = np.where(valid)
    if len(pi) == 0:
        raise RuntimeError("No valid q0 entries found.")
    return grid_points, q0, q0_g, active_sensor, q0_data["q0_active_planes"], pi, qi


def select_entry(pi, qi, args):
    if args.entry >= 0:
        if args.entry >= len(pi):
            raise RuntimeError(f"--entry {args.entry} out of range; valid entries: 0..{len(pi)-1}")
        return int(args.entry)

    matches = np.arange(len(pi))
    if args.p_index >= 0:
        matches = matches[pi[matches] == args.p_index]
    if args.q0_index >= 0:
        matches = matches[qi[matches] == args.q0_index]
    if len(matches) == 0:
        raise RuntimeError("No valid entry matches requested --p-index/--q0-index.")
    rng = np.random.default_rng(args.seed)
    return int(rng.choice(matches))


def publish_joint_state(pub, joint_names, q):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = list(joint_names)
    msg.position = [float(v) for v in q]
    pub.publish(msg)


def make_sample_markers(robot, q0_data, entry_id, args):
    joint_names = names_from_npz(q0_data, "joint_names", DEFAULT_JOINT_NAMES)
    sensor_frames = names_from_npz(q0_data, "sensor_frames", DEFAULT_SENSOR_FRAMES)
    grid_points, q0, q0_g, active_sensor, active_planes, pi, qi = load_valid_entries(q0_data)

    p_row = int(pi[entry_id])
    q_row = int(qi[entry_id])
    p = grid_points[p_row]
    q = q0[p_row, q_row]
    q_map = q_row_to_map(joint_names, q)

    sensor_chains = [find_chain_joints(robot, args.base_frame, frame) for frame in sensor_frames]
    sensor_transforms = [fk_transform(chain, q_map) for chain in sensor_chains]
    oracle_args = SimpleNamespace(
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        z_min=args.z_min,
        z_max=args.z_max,
        delta=args.delta,
    )
    sensor_margins, planes, best_sensor, best_margin, g, visible = compute_visibility(
        p.reshape(1, 3), sensor_transforms, oracle_args
    )

    best_sensor = int(best_sensor[0])
    best_plane = int(planes[0, best_sensor])
    marker_array = MarkerArray()
    marker_array.markers.append(make_delete_all(args.base_frame))

    marker_id = 1
    for sensor_idx, transform in enumerate(sensor_transforms):
        marker_array.markers.append(
            make_fov_marker(marker_id, args.base_frame, transform, args, active=(sensor_idx == best_sensor))
        )
        marker_id += 1

    point_text = (
        f"p row {p_row}, q0 {q_row}\n"
        f"g={float(g[0]):+.5f}, stored={float(q0_g[p_row, q_row]):+.5f}"
    )
    for marker in make_point_marker(marker_id, args.base_frame, p, point_text):
        marker_array.markers.append(marker)
        marker_id += 1

    marker_array.markers.append(make_sensor_axes(marker_id, args.base_frame, sensor_transforms[best_sensor]))
    marker_id += 1

    active_text = (
        f"active sensor {best_sensor}: {sensor_frames[best_sensor]}\n"
        f"active plane {best_plane}: {PLANE_NAMES[best_plane]}\n"
        f"best_margin={float(best_margin[0]):+.5f}, visible={bool(visible[0])}"
    )
    marker_array.markers.append(
        make_sensor_text(marker_id, args.base_frame, sensor_transforms[best_sensor], active_text)
    )

    print("")
    print("=== RViz Visibility Q0 Sample ===")
    print(f"entry:          {entry_id}")
    print(f"p row:          {p_row}")
    print(f"q0 index:       {q_row}")
    print(f"p:              [{p[0]: .6f}, {p[1]: .6f}, {p[2]: .6f}]")
    print("q:")
    for name, value in zip(joint_names, q):
        print(f"  {name:18s}: {value: .6f}")
    print(f"stored g:       {float(q0_g[p_row, q_row]): .9f}")
    print(f"recomputed g:   {float(g[0]): .9f}")
    if "q0_init_templates" in q0_data:
        q_init = q0_data["q0_init_templates"][p_row, q_row]
        q_init_g = float(q0_data["q0_init_g"][p_row, q_row])
        delta_q = q0_data["q0_delta_q"][p_row, q_row]
        delta_l1 = float(q0_data["q0_delta_norm_l1"][p_row, q_row])
        delta_l2 = float(q0_data["q0_delta_norm_l2"][p_row, q_row])
        source_topk_rank = int(q0_data["q0_source_topk_rank"][p_row, q_row])
        source_random_index = int(q0_data["q0_source_random_index"][p_row, q_row])
        print(f"init g:         {q_init_g: .9f}")
        print(f"|g_init| -> |g|:{abs(q_init_g): .9f} -> {abs(float(g[0])): .9f}")
        print(f"||dq||_1:       {delta_l1: .9f}")
        print(f"||dq||_2:       {delta_l2: .9f}")
        print(f"source topk:    {source_topk_rank}")
        print(f"source random:  {source_random_index}")
        print("q init:")
        for name, value in zip(joint_names, q_init):
            print(f"  {name:18s}: {value: .6f}")
        print("delta q:")
        for name, value in zip(joint_names, delta_q):
            print(f"  {name:18s}: {value: .6f}")
    print(f"active sensor:  {best_sensor} ({sensor_frames[best_sensor]})")
    print(f"active plane:   {best_plane} ({PLANE_NAMES[best_plane]})")
    print("sensor margins:")
    for idx, frame in enumerate(sensor_frames):
        plane = int(planes[0, idx])
        print(f"  {idx}: {frame:28s} margin={sensor_margins[0, idx]: .6f} plane={plane} ({PLANE_NAMES[plane]})")

    return joint_names, q, marker_array


def main():
    parser = argparse.ArgumentParser(
        description="Publish RViz markers for visual inspection of M4 visibility q0 samples."
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--q0", required=True, help="M4 zero-level .npz file.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--entry", type=int, default=-1,
                        help="Flat valid q0 entry index. Overrides random selection.")
    parser.add_argument("--p-index", type=int, default=-1,
                        help="Optional q0 file p row to inspect.")
    parser.add_argument("--q0-index", type=int, default=-1,
                        help="Optional q0 slot index to inspect.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds to publish. 0 means publish until Ctrl-C.")
    parser.add_argument("--joint-states-topic", default="/joint_states",
                        help="JointState topic to publish q0 on. Use /care_arm/joint_states only if no Gazebo controller is publishing it.")
    parser.add_argument("--no-joint-states", action="store_true",
                        help="Do not publish /joint_states.")
    parser.add_argument("--horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--vertical-fov-deg", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    q0_data = np.load(args.q0, allow_pickle=True)
    args.horizontal_fov_deg = float(q0_data["horizontal_fov_deg"]) if args.horizontal_fov_deg is None else args.horizontal_fov_deg
    args.vertical_fov_deg = float(q0_data["vertical_fov_deg"]) if args.vertical_fov_deg is None else args.vertical_fov_deg
    args.z_min = float(q0_data["z_min"]) if args.z_min is None else args.z_min
    args.z_max = float(q0_data["z_max"]) if args.z_max is None else args.z_max
    args.delta = float(q0_data["delta"]) if args.delta is None else args.delta

    _, _, _, _, _, pi, qi = load_valid_entries(q0_data)
    entry_id = select_entry(pi, qi, args)

    rospy.init_node("visualize_visibility_q0_rviz", anonymous=True)
    marker_pub = rospy.Publisher("/care_visibility_cdf/rviz_check/markers", MarkerArray, queue_size=1, latch=True)
    joint_pub = rospy.Publisher(args.joint_states_topic, JointState, queue_size=1)

    robot = URDF.from_xml_file(args.urdf)
    joint_names, q, marker_array = make_sample_markers(robot, q0_data, entry_id, args)
    if not args.no_joint_states:
        print(f"publishing q0 JointState on: {args.joint_states_topic}")

    rate = rospy.Rate(args.rate)
    start = rospy.Time.now()
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        for marker in marker_array.markers:
            marker.header.stamp = now
        marker_pub.publish(marker_array)
        if not args.no_joint_states:
            publish_joint_state(joint_pub, joint_names, q)

        if args.duration > 0.0 and (now - start).to_sec() >= args.duration:
            break
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

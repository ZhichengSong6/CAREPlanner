#!/usr/bin/env python3

import argparse
import math
import os
import sys
from collections import Counter

import numpy as np
from urdf_parser_py.urdf import URDF

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    find_chain_joints,
    fk_transform,
    make_transform,
)


def names_from_npz(data, key, default):
    if key in data:
        return [str(name) for name in data[key]]
    return list(default)


def q_row_to_map(joint_names, q_row):
    return {name: float(value) for name, value in zip(joint_names, q_row)}


def origin_transform(origin):
    if origin is None:
        return np.eye(4)
    xyz = origin.xyz if origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = origin.rpy if origin.rpy is not None else [0.0, 0.0, 0.0]
    return make_transform(xyz, rpy)


def load_collision_primitives(robot):
    primitives = []
    for link in robot.links:
        for idx, collision in enumerate(link.collisions):
            geom = collision.geometry
            geom_type = geom.__class__.__name__.lower()
            if geom_type not in {"box", "cylinder", "sphere"}:
                continue
            name = collision.name or f"{link.name}_collision_{idx}"
            primitives.append({
                "link": link.name,
                "name": name,
                "type": geom_type,
                "origin": origin_transform(collision.origin),
                "geometry": geom,
            })
    return primitives


def point_inside_box(point, half_size, eps=1e-10):
    return bool(np.all(point >= -half_size - eps) and np.all(point <= half_size + eps))


def segment_box_intersection(p0, p1, half_size):
    direction = p1 - p0
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if p0[axis] < -half_size[axis] or p0[axis] > half_size[axis]:
                return None
            continue
        inv_d = 1.0 / direction[axis]
        t1 = (-half_size[axis] - p0[axis]) * inv_d
        t2 = ( half_size[axis] - p0[axis]) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min


def point_inside_cylinder(point, radius, half_length, eps=1e-10):
    radial_sq = point[0] * point[0] + point[1] * point[1]
    return bool(radial_sq <= (radius + eps) ** 2 and abs(point[2]) <= half_length + eps)


def segment_cylinder_intersection(p0, p1, radius, half_length):
    d = p1 - p0
    candidates = []

    a = d[0] * d[0] + d[1] * d[1]
    b = 2.0 * (p0[0] * d[0] + p0[1] * d[1])
    c = p0[0] * p0[0] + p0[1] * p0[1] - radius * radius
    if abs(a) > 1e-12:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(max(disc, 0.0))
            for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]:
                if 0.0 <= t <= 1.0:
                    z = p0[2] + t * d[2]
                    if -half_length <= z <= half_length:
                        candidates.append(t)

    if abs(d[2]) > 1e-12:
        for z_cap in [-half_length, half_length]:
            t = (z_cap - p0[2]) / d[2]
            if 0.0 <= t <= 1.0:
                x = p0[0] + t * d[0]
                y = p0[1] + t * d[1]
                if x * x + y * y <= radius * radius:
                    candidates.append(t)

    if not candidates:
        return None
    return min(candidates)


def point_inside_sphere(point, radius, eps=1e-10):
    return bool(np.dot(point, point) <= (radius + eps) ** 2)


def segment_sphere_intersection(p0, p1, radius):
    d = p1 - p0
    a = float(np.dot(d, d))
    if a < 1e-12:
        return None
    b = 2.0 * float(np.dot(p0, d))
    c = float(np.dot(p0, p0) - radius * radius)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(max(disc, 0.0))
    candidates = []
    for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]:
        if 0.0 <= t <= 1.0:
            candidates.append(t)
    if not candidates:
        return None
    return min(candidates)


def transform_point(transform, point):
    point_h = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return (transform @ point_h)[:3]


def collision_hit_t(local_p0, local_p1, primitive, ignore_start_inside):
    geom = primitive["geometry"]
    geom_type = primitive["type"]

    if geom_type == "box":
        half_size = 0.5 * np.asarray(geom.size, dtype=np.float64)
        if ignore_start_inside and point_inside_box(local_p0, half_size):
            return None
        return segment_box_intersection(local_p0, local_p1, half_size)

    if geom_type == "cylinder":
        radius = float(geom.radius)
        half_length = 0.5 * float(geom.length)
        if ignore_start_inside and point_inside_cylinder(local_p0, radius, half_length):
            return None
        return segment_cylinder_intersection(local_p0, local_p1, radius, half_length)

    if geom_type == "sphere":
        radius = float(geom.radius)
        if ignore_start_inside and point_inside_sphere(local_p0, radius):
            return None
        return segment_sphere_intersection(local_p0, local_p1, radius)

    return None


def raycast_self_occlusion(robot, collision_chains, primitives, sensor_transform, point_base,
                           q_map, args):
    sensor_origin = sensor_transform[:3, 3]
    vec = point_base - sensor_origin
    length = float(np.linalg.norm(vec))
    if length < args.min_ray_length:
        return False, None

    direction = vec / length
    start = sensor_origin + args.ray_start_offset * direction
    end = point_base - args.point_end_offset * direction
    segment_len = float(np.linalg.norm(end - start))
    if segment_len < args.min_ray_length:
        return False, None

    best = None
    for primitive, chain in zip(primitives, collision_chains):
        if primitive["link"] in args.ignore_links:
            continue

        link_transform = fk_transform(chain, q_map)
        collision_transform = link_transform @ primitive["origin"]
        inv_collision_transform = np.linalg.inv(collision_transform)
        local_start = transform_point(inv_collision_transform, start)
        local_end = transform_point(inv_collision_transform, end)

        t = collision_hit_t(local_start, local_end, primitive, args.ignore_start_inside)
        if t is None:
            continue
        if t < -1e-9 or t > 1.0 + 1e-9:
            continue

        distance = max(0.0, min(1.0, float(t))) * segment_len + args.ray_start_offset
        if distance < args.min_hit_distance:
            continue
        if best is None or distance < best["distance"]:
            best = {
                "distance": distance,
                "link": primitive["link"],
                "collision": primitive["name"],
                "type": primitive["type"],
            }

    return best is not None, best


def load_q0_entries(q0_data, num_samples, seed):
    grid_points = q0_data["grid_points"].astype(np.float64)
    q0 = q0_data["q0_templates"].astype(np.float64)
    q0_g = q0_data["q0_g"]
    active_sensor = q0_data["q0_active_sensor"]

    valid = np.isfinite(q0_g) & (active_sensor >= 0)
    pi, qi = np.where(valid)
    if len(pi) == 0:
        raise RuntimeError("No valid q0 entries found.")

    if num_samples > 0:
        rng = np.random.default_rng(seed)
        count = min(num_samples, len(pi))
        pick = rng.choice(len(pi), size=count, replace=False)
        pi = pi[pick]
        qi = qi[pick]

    return grid_points, q0, q0_g, active_sensor, pi, qi


def main():
    parser = argparse.ArgumentParser(
        description="Check self-occlusion for M4 visibility q0 samples using simplified URDF collision primitives."
    )
    parser.add_argument("--urdf", required=True, help="Simplified collision URDF.")
    parser.add_argument("--q0", required=True, help="M4 zero-level .npz file.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of valid q0 entries to sample. Use 0 for all valid q0.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ray-start-offset", type=float, default=0.03,
                        help="Move ray start this far from sensor origin toward point.")
    parser.add_argument("--point-end-offset", type=float, default=0.005,
                        help="Move ray end this far before the query point.")
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--ignore-start-inside", action="store_true", default=True)
    parser.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    parser.add_argument("--ignore-links", nargs="*", default=[],
                        help="Collision links to ignore for all raycasts.")
    parser.add_argument("--print-examples", type=int, default=10)
    args = parser.parse_args()

    q0_data = np.load(args.q0, allow_pickle=True)
    joint_names = names_from_npz(q0_data, "joint_names", DEFAULT_JOINT_NAMES)
    sensor_frames = names_from_npz(q0_data, "sensor_frames", DEFAULT_SENSOR_FRAMES)
    grid_points, q0, q0_g, active_sensor, pi, qi = load_q0_entries(q0_data, args.num_samples, args.seed)

    robot = URDF.from_xml_file(args.urdf)
    primitives = load_collision_primitives(robot)
    if not primitives:
        raise RuntimeError("No supported collision primitives found. Expected box/cylinder/sphere.")

    sensor_chains = [find_chain_joints(robot, args.base_frame, frame) for frame in sensor_frames]
    collision_chains = [find_chain_joints(robot, args.base_frame, primitive["link"]) for primitive in primitives]

    active_occluded = 0
    hit_counter = Counter()
    hit_type_counter = Counter()
    active_sensor_counter = Counter()
    active_sensor_occluded_counter = Counter()
    examples = []

    print("")
    print("=== Self-Occlusion Check Config ===")
    print(f"urdf:              {args.urdf}")
    print(f"q0:                {args.q0}")
    print(f"num_checked:       {len(pi)}")
    print(f"collision_shapes:  {len(primitives)}")
    print(f"ray_start_offset:  {args.ray_start_offset}")
    print(f"point_end_offset:  {args.point_end_offset}")
    print(f"ignore_start_inside: {args.ignore_start_inside}")
    print(f"ignore_links:      {args.ignore_links}")

    for sample_idx, (p_idx, q_idx) in enumerate(zip(pi, qi)):
        p = grid_points[p_idx]
        q = q0[p_idx, q_idx]
        sensor_idx = int(active_sensor[p_idx, q_idx])
        q_map = q_row_to_map(joint_names, q)
        sensor_transform = fk_transform(sensor_chains[sensor_idx], q_map)
        occluded, hit = raycast_self_occlusion(
            robot,
            collision_chains,
            primitives,
            sensor_transform,
            p,
            q_map,
            args,
        )

        active_sensor_counter[sensor_idx] += 1
        if occluded:
            active_occluded += 1
            active_sensor_occluded_counter[sensor_idx] += 1
            hit_counter[hit["link"]] += 1
            hit_type_counter[hit["type"]] += 1
            if len(examples) < args.print_examples:
                examples.append((sample_idx, int(p_idx), int(q_idx), sensor_idx, float(q0_g[p_idx, q_idx]), hit))

    total = len(pi)
    print("")
    print("=== Self-Occlusion Summary ===")
    print(f"checked q0:                 {total}")
    print(f"active sensor occluded:     {active_occluded}")
    print(f"active occluded ratio:      {active_occluded / max(total, 1):.6f}")
    print(f"active sensor clear:        {total - active_occluded}")

    print("")
    print("active sensor histogram:")
    for idx, frame in enumerate(sensor_frames):
        count = active_sensor_counter[idx]
        occ = active_sensor_occluded_counter[idx]
        ratio = occ / max(count, 1)
        print(f"  {idx}: {frame:28s} count={count:6d} occluded={occ:6d} ratio={ratio:.4f}")

    print("")
    print("occluding link histogram:")
    for link, count in hit_counter.most_common():
        print(f"  {link:28s} {count:8d}")

    print("")
    print("occluding primitive type histogram:")
    for primitive_type, count in hit_type_counter.most_common():
        print(f"  {primitive_type:12s} {count:8d}")

    if examples:
        print("")
        print("occluded examples:")
        for sample_idx, p_idx, q_idx, sensor_idx, g_value, hit in examples:
            print(
                f"  sample={sample_idx:5d} p_row={p_idx:5d} q0={q_idx:2d} "
                f"sensor={sensor_idx} g={g_value:+.6f} "
                f"hit={hit['link']}/{hit['collision']} type={hit['type']} dist={hit['distance']:.4f}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

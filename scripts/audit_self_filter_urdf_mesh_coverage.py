#!/usr/bin/env python3
"""
Configuration-aware GLOBAL-UNION audit for CAREPlanner ToF self filtering.

This script mirrors the runtime semantics in tof_fusion_self_filter_node.cpp:

    for every rendered visual-surface point p at robot configuration q:
        d_global(p,q) = min over ALL dedicated self-filter primitives
                        d(p, G_link,primitive(q))

    covered  <=> d_global <= containment_tolerance

Therefore a visual point belonging to link2 is considered covered if it lies
inside ANY self-filter primitive, including a primitive attached to link1,
link3, wrist links, etc.  This is intentionally different from the old
per-link/anchor audit.

Reference rendered robot:
    src/arm_description/urdf/Arm.urdf

Dedicated self-filter geometry:
    src/arm_description/urdf/Arm_with_self_filter_collision.urdf

The default configuration is all movable joints at q=0, matching the Phase-E
zero initial configuration.  Arbitrary configurations can be supplied with:

    --joint-values "joint1=0.2,joint2=-0.4,wrist_joint1=0.1"

No runtime padding is applied.

Outputs:
    self_filter_mesh_coverage.csv
    self_filter_mesh_coverage.json
    self_filter_mesh_coverage_worst_points.csv
    self_filter_mesh_unmapped_visual_links.csv

The worst-points CSV preserves x_anchor/y_anchor/z_anchor in the SOURCE visual
link frame so the existing RViz coverage marker can display the point on the
correct robot link.  It also reports world/base coordinates and the globally
nearest collision link/primitive.
"""

import argparse
import csv
import heapq
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------------

def parse_vec(text, n=3, default=None):
    if default is None:
        default = [0.0] * n
    if text is None:
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in str(text).split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} values, got {vals}")
    return np.asarray(vals, dtype=np.float64)


def rpy_to_matrix(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # URDF fixed-axis RPY: Rz(yaw) * Ry(pitch) * Rx(roll).
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-15:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.asarray([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def origin_transform(elem):
    if elem is None:
        return np.eye(3), np.zeros(3)
    return (
        rpy_to_matrix(parse_vec(elem.attrib.get("rpy"))),
        parse_vec(elem.attrib.get("xyz")),
    )


def compose(T_ab, T_bc):
    """Return T_ac for p_a = T_ab(T_bc(p_c))."""
    Ra, ta = T_ab
    Rb, tb = T_bc
    return Ra @ Rb, Ra @ tb + ta


def apply_transform(points, T):
    R, t = T
    return points @ R.T + t


# ---------------------------------------------------------------------------
# STL sampling
# ---------------------------------------------------------------------------

def resolve_package_uri(uri, repo_root):
    prefix = "package://"
    if not uri.startswith(prefix):
        p = Path(uri)
        return p if p.is_absolute() else (repo_root / p)

    rest = uri[len(prefix):]
    package, rel = rest.split("/", 1)
    candidate = repo_root / "src" / package / rel
    if candidate.exists():
        return candidate

    matches = list((repo_root / "src").glob(f"**/{package}/{rel}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve {uri} from {repo_root}")


def load_stl_triangles(path):
    data = Path(path).read_bytes()

    # Binary STL.
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            tri = np.empty((n, 3, 3), dtype=np.float64)
            off = 84
            for i in range(n):
                vals = struct.unpack_from("<12fH", data, off)
                tri[i, 0, :] = vals[3:6]
                tri[i, 1, :] = vals[6:9]
                tri[i, 2, :] = vals[9:12]
                off += 50
            return tri

    # ASCII STL fallback.
    verts = []
    for raw in data.decode("utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if s.lower().startswith("vertex "):
            toks = s.split()
            if len(toks) >= 4:
                verts.append(
                    [float(toks[1]), float(toks[2]), float(toks[3])])

    if not verts or len(verts) % 3 != 0:
        raise ValueError(f"unsupported/corrupt STL: {path}")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def triangle_surface_samples(tri, spacing):
    a, b, c = tri
    max_edge = max(
        np.linalg.norm(b - a),
        np.linalg.norm(c - b),
        np.linalg.norm(a - c),
    )
    n = max(1, int(math.ceil(max_edge / spacing)))

    out = np.empty(((n + 1) * (n + 2) // 2, 3), dtype=np.float64)
    k = 0
    inv = 1.0 / n
    for i in range(n + 1):
        for j in range(n + 1 - i):
            u = i * inv
            v = j * inv
            out[k] = a + u * (b - a) + v * (c - a)
            k += 1
    return out


# ---------------------------------------------------------------------------
# Robot model / FK
# ---------------------------------------------------------------------------

def parse_joint_values(text):
    out = {}
    text = (text or "").strip()
    if not text:
        return out

    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"invalid --joint-values token {token!r}; expected name=value")
        name, value = token.split("=", 1)
        out[name.strip()] = float(value)
    return out


def parse_reference_robot(path):
    root = ET.parse(path).getroot()

    links = {link.attrib.get("name", "") for link in root.findall("link")}
    links.discard("")

    visuals_by_link = {}
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        visuals = []

        for vi, visual in enumerate(link.findall("visual")):
            geom = visual.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue

            visuals.append({
                "name": visual.attrib.get("name", f"visual_{vi}"),
                "filename": mesh.attrib["filename"],
                "scale": parse_vec(
                    mesh.attrib.get("scale"),
                    default=[1.0, 1.0, 1.0]),
                "T_link_visual": origin_transform(visual.find("origin")),
            })

        if visuals:
            visuals_by_link[link_name] = visuals

    parent_joint = {}
    movable_joint_names = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue

        jtype = joint.attrib.get("type", "fixed")
        name = joint.attrib.get("name", "")
        child_name = child.attrib["link"]

        axis_elem = joint.find("axis")
        axis = (
            parse_vec(axis_elem.attrib.get("xyz"))
            if axis_elem is not None
            else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        )

        parent_joint[child_name] = {
            "name": name,
            "type": jtype,
            "parent": parent.attrib["link"],
            "T_parent_joint": origin_transform(joint.find("origin")),
            "axis": axis,
        }

        if jtype in ("revolute", "continuous", "prismatic"):
            movable_joint_names.append(name)

    children = set(parent_joint.keys())
    roots = sorted(links - children)
    if len(roots) != 1:
        raise RuntimeError(f"expected one URDF root, got {roots}")

    return (
        visuals_by_link,
        parent_joint,
        roots[0],
        sorted(set(movable_joint_names)),
        links,
    )


def joint_motion_transform(joint, q):
    jtype = joint["type"]
    if jtype in ("fixed", "floating", "planar"):
        if jtype not in ("fixed",):
            raise RuntimeError(
                f"unsupported joint type {jtype} for {joint['name']}")
        return np.eye(3), np.zeros(3)

    if jtype in ("revolute", "continuous"):
        return axis_angle_matrix(joint["axis"], q), np.zeros(3)

    if jtype == "prismatic":
        axis = np.asarray(joint["axis"], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-15:
            axis = axis / norm
        return np.eye(3), axis * q

    raise RuntimeError(
        f"unsupported joint type {jtype} for {joint['name']}")


def compute_world_fk(
        links,
        parent_joint,
        root_link,
        joint_values):
    cache = {
        root_link: (np.eye(3), np.zeros(3)),
    }

    def solve(link):
        if link in cache:
            return cache[link]

        joint = parent_joint.get(link)
        if joint is None:
            raise RuntimeError(f"no parent joint for non-root link {link}")

        T_world_parent = solve(joint["parent"])
        q = float(joint_values.get(joint["name"], 0.0))
        T_parent_child = compose(
            joint["T_parent_joint"],
            joint_motion_transform(joint, q))
        cache[link] = compose(T_world_parent, T_parent_child)
        return cache[link]

    for link in links:
        solve(link)

    return cache


# ---------------------------------------------------------------------------
# Dedicated self-filter primitives
# ---------------------------------------------------------------------------

class Primitive:
    def __init__(
            self, link, name, kind, R, t,
            size=None, radius=None, length=None):
        self.link = link
        self.name = name
        self.kind = kind
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.size = size
        self.radius = radius
        self.length = length

    def transformed(self, T_world_link):
        T_world_primitive = compose(
            T_world_link,
            (self.R, self.t))
        return Primitive(
            self.link,
            self.name,
            self.kind,
            T_world_primitive[0],
            T_world_primitive[1],
            size=self.size,
            radius=self.radius,
            length=self.length)

    def signed_distance(self, points_world):
        # self.R/self.t represent T_world_primitive.
        # Row-vector equivalent of p_primitive = R^T (p_world - t).
        q = (points_world - self.t) @ self.R

        if self.kind == "box":
            d = np.abs(q) - 0.5 * self.size
            outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            inside = np.minimum(np.max(d, axis=1), 0.0)
            return outside + inside

        if self.kind == "cylinder":
            radial = np.linalg.norm(q[:, :2], axis=1) - self.radius
            axial = np.abs(q[:, 2]) - 0.5 * self.length
            outside = np.hypot(
                np.maximum(radial, 0.0),
                np.maximum(axial, 0.0))
            inside = np.minimum(np.maximum(radial, axial), 0.0)
            return outside + inside

        if self.kind == "sphere":
            return np.linalg.norm(q, axis=1) - self.radius

        raise RuntimeError(self.kind)


def parse_self_filter_collisions(path):
    root = ET.parse(path).getroot()
    collisions_by_link = {}

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        prims = []

        for ci, collision in enumerate(link.findall("collision")):
            geom = collision.find("geometry")
            if geom is None:
                continue

            R, t = origin_transform(collision.find("origin"))
            cname = collision.attrib.get("name", f"collision_{ci}")

            box = geom.find("box")
            cyl = geom.find("cylinder")
            sph = geom.find("sphere")

            if box is not None:
                prims.append(Primitive(
                    link_name, cname, "box", R, t,
                    size=parse_vec(box.attrib["size"])))
            elif cyl is not None:
                prims.append(Primitive(
                    link_name, cname, "cylinder", R, t,
                    radius=float(cyl.attrib["radius"]),
                    length=float(cyl.attrib["length"])))
            elif sph is not None:
                prims.append(Primitive(
                    link_name, cname, "sphere", R, t,
                    radius=float(sph.attrib["radius"])))
            else:
                raise RuntimeError(
                    f"{link_name}/{cname}: dedicated self-filter URDF "
                    "contains unsupported non-primitive collision geometry")

        if prims:
            collisions_by_link[link_name] = prims

    return collisions_by_link


def build_global_primitives(collisions_by_link, world_fk):
    out = []
    missing_links = []

    for link_name in sorted(collisions_by_link):
        if link_name not in world_fk:
            missing_links.append(link_name)
            continue
        T_world_link = world_fk[link_name]
        for primitive in collisions_by_link[link_name]:
            out.append(primitive.transformed(T_world_link))

    if missing_links:
        raise RuntimeError(
            "self-filter collision links missing from reference URDF FK: "
            + ", ".join(missing_links))
    if not out:
        raise RuntimeError("no global self-filter primitives")
    return out


# ---------------------------------------------------------------------------
# Global-union coverage evaluation
# ---------------------------------------------------------------------------

def evaluate_global_union(points_world, global_primitives):
    d_union = np.full(
        points_world.shape[0],
        np.inf,
        dtype=np.float64)
    nearest_idx = np.full(
        points_world.shape[0],
        -1,
        dtype=np.int32)

    for pi, prim in enumerate(global_primitives):
        d = prim.signed_distance(points_world)
        better = d < d_union
        d_union[better] = d[better]
        nearest_idx[better] = pi

    return d_union, nearest_idx


def audit_visual_link_global(
        source_link,
        visuals,
        T_world_source,
        global_primitives,
        repo_root,
        spacing,
        tolerance,
        top_k,
        batch_target=100000):
    total = 0
    outside = 0
    outside_distances = []
    worst_heap = []
    heap_serial = 0

    def consume_batch(
            samples_source,
            triangle_ids,
            visual_name,
            mesh_path):
        nonlocal total, outside, heap_serial

        if not samples_source:
            return

        source_pts = np.concatenate(samples_source, axis=0)
        tri_ids = np.concatenate(triangle_ids, axis=0)
        world_pts = apply_transform(source_pts, T_world_source)

        d_union, nearest_idx = evaluate_global_union(
            world_pts,
            global_primitives)

        total += len(d_union)
        mask = d_union > tolerance
        if not np.any(mask):
            return

        idxs = np.flatnonzero(mask)
        vals = d_union[idxs]
        outside += len(idxs)
        outside_distances.extend(vals.tolist())

        # Preserve only the worst top_k samples to keep diagnostic files small.
        for local_i, d in zip(idxs, vals):
            prim = global_primitives[int(nearest_idx[local_i])]
            p_source = source_pts[local_i]
            p_world = world_pts[local_i]
            rec = {
                "source_link": source_link,
                # Existing RViz publisher uses anchor_link as marker frame.
                # For a global-union audit, source_link is the correct frame
                # for the rendered visual point.
                "anchor_link": source_link,
                "visual": visual_name,
                "mesh": str(mesh_path),
                "triangle": int(tri_ids[local_i]),
                "x_anchor": float(p_source[0]),
                "y_anchor": float(p_source[1]),
                "z_anchor": float(p_source[2]),
                "x_world": float(p_world[0]),
                "y_world": float(p_world[1]),
                "z_world": float(p_world[2]),
                "outside_m": float(d),
                "nearest_collision_link": prim.link,
                "nearest_primitive": prim.name,
                "nearest_type": prim.kind,
                "fixed_chain": "",
            }

            item = (float(d), heap_serial, rec)
            heap_serial += 1
            if len(worst_heap) < top_k:
                heapq.heappush(worst_heap, item)
            elif d > worst_heap[0][0]:
                heapq.heapreplace(worst_heap, item)

    for vis in visuals:
        mesh_path = resolve_package_uri(vis["filename"], repo_root)
        triangles = load_stl_triangles(mesh_path)
        T_source_visual = vis["T_link_visual"]

        sample_chunks = []
        tri_chunks = []
        pending = 0

        for tri_idx, tri_mesh in enumerate(triangles):
            tri_scaled = tri_mesh * vis["scale"]
            samples_visual = triangle_surface_samples(
                tri_scaled,
                spacing)
            samples_source = apply_transform(
                samples_visual,
                T_source_visual)

            sample_chunks.append(samples_source)
            tri_chunks.append(
                np.full(
                    len(samples_source),
                    tri_idx,
                    dtype=np.int32))
            pending += len(samples_source)

            if pending >= batch_target:
                consume_batch(
                    sample_chunks,
                    tri_chunks,
                    vis["name"],
                    mesh_path)
                sample_chunks = []
                tri_chunks = []
                pending = 0

        consume_batch(
            sample_chunks,
            tri_chunks,
            vis["name"],
            mesh_path)

    worst = [
        item[2]
        for item in sorted(
            worst_heap,
            key=lambda item: item[0],
            reverse=True)
    ]

    if outside_distances:
        arr = np.asarray(outside_distances, dtype=np.float64)
        max_out = float(np.max(arr))
        p95_out = float(np.percentile(arr, 95))
        mean_out = float(np.mean(arr))
    else:
        max_out = p95_out = mean_out = 0.0

    return {
        "source_link": source_link,
        "anchor_link": source_link,
        "fixed_chain": "",
        "visual_count": len(visuals),
        "global_primitive_count": len(global_primitives),
        "surface_samples": int(total),
        "outside_samples": int(outside),
        "outside_fraction": outside / total if total else float("nan"),
        "max_outside_m": max_out,
        "p95_outside_m": p95_out,
        "mean_outside_m": mean_out,
        "contained": bool(outside == 0),
        "worst": worst,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        default="/home/zhicheng/Project/CAREPlanner")
    ap.add_argument(
        "--reference-urdf",
        default="src/arm_description/urdf/Arm.urdf",
        help="actual Gazebo/rendered robot visual geometry")
    ap.add_argument(
        "--self-filter-urdf",
        default=(
            "src/arm_description/urdf/"
            "Arm_with_self_filter_collision.urdf"))
    ap.add_argument(
        "--surface-spacing",
        type=float,
        default=0.002,
        help="approximate visual-mesh surface sampling spacing in meters")
    ap.add_argument(
        "--containment-tolerance",
        type=float,
        default=1e-4,
        help="global d <= tolerance is treated as covered (default 0.1 mm)")
    ap.add_argument(
        "--joint-values",
        default="",
        help=(
            "comma-separated movable joint configuration, e.g. "
            "'joint1=0.1,joint2=-0.2,wrist_joint1=0.3'; "
            "unspecified movable joints default to zero"))
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument(
        "--output-dir",
        default="outputs/self_filter_mesh_coverage")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()

    reference_urdf = Path(args.reference_urdf)
    if not reference_urdf.is_absolute():
        reference_urdf = repo_root / reference_urdf

    self_filter_urdf = Path(args.self_filter_urdf)
    if not self_filter_urdf.is_absolute():
        self_filter_urdf = repo_root / self_filter_urdf

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.surface_spacing <= 0:
        raise SystemExit("--surface-spacing must be positive")
    if args.containment_tolerance < 0:
        raise SystemExit("--containment-tolerance must be nonnegative")
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    joint_values = parse_joint_values(args.joint_values)

    (
        visuals,
        parent_joint,
        root_link,
        movable_joint_names,
        all_links,
    ) = parse_reference_robot(reference_urdf)

    unknown_joint_names = sorted(
        set(joint_values) - set(movable_joint_names))
    if unknown_joint_names:
        raise SystemExit(
            "unknown movable joint(s) in --joint-values: "
            + ", ".join(unknown_joint_names))

    # Make the evaluated configuration explicit and reproducible.
    resolved_joint_values = {
        name: float(joint_values.get(name, 0.0))
        for name in movable_joint_names
    }

    world_fk = compute_world_fk(
        all_links,
        parent_joint,
        root_link,
        resolved_joint_values)

    collisions = parse_self_filter_collisions(self_filter_urdf)
    global_primitives = build_global_primitives(
        collisions,
        world_fk)

    results = []
    worst_rows = []
    unmapped = []

    for source_link in sorted(visuals):
        if source_link not in world_fk:
            unmapped.append({
                "source_link": source_link,
                "reason": "source visual link missing from FK",
                "fixed_chain": "",
            })
            continue

        result = audit_visual_link_global(
            source_link,
            visuals[source_link],
            world_fk[source_link],
            global_primitives,
            repo_root,
            args.surface_spacing,
            args.containment_tolerance,
            args.top_k)
        results.append(result)
        worst_rows.extend(result["worst"])

    # Sort primary table from largest true global-union leak to smallest.
    results.sort(
        key=lambda r: r["max_outside_m"],
        reverse=True)
    worst_rows.sort(
        key=lambda r: r["outside_m"],
        reverse=True)

    csv_path = out_dir / "self_filter_mesh_coverage.csv"
    json_path = out_dir / "self_filter_mesh_coverage.json"
    worst_path = out_dir / "self_filter_mesh_coverage_worst_points.csv"
    unmapped_path = out_dir / "self_filter_mesh_unmapped_visual_links.csv"

    fields = [
        "source_link", "anchor_link", "fixed_chain",
        "visual_count", "global_primitive_count",
        "surface_samples", "outside_samples", "outside_fraction",
        "max_outside_m", "p95_outside_m", "mean_outside_m",
        "contained",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    worst_fields = [
        "source_link", "anchor_link", "visual", "mesh", "triangle",
        "x_anchor", "y_anchor", "z_anchor",
        "x_world", "y_world", "z_world",
        "outside_m",
        "nearest_collision_link", "nearest_primitive", "nearest_type",
        "fixed_chain",
    ]
    with worst_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        w.writerows(worst_rows)

    with unmapped_path.open("w", newline="") as f:
        fields_u = ["source_link", "reason", "fixed_chain"]
        w = csv.DictWriter(f, fieldnames=fields_u)
        w.writeheader()
        w.writerows(unmapped)

    all_mapped = len(unmapped) == 0
    all_contained = all(r["contained"] for r in results)

    payload = {
        "audit_semantics": "configuration_aware_global_collision_union",
        "reference_urdf": str(reference_urdf),
        "self_filter_urdf": str(self_filter_urdf),
        "root_link": root_link,
        "surface_spacing_m": args.surface_spacing,
        "containment_tolerance_m": args.containment_tolerance,
        "joint_values": resolved_joint_values,
        "global_primitive_count": len(global_primitives),
        "reference_visual_link_count": len(visuals),
        "self_filter_collision_link_count": len(collisions),
        "mapped_visual_link_count": len(results),
        "unmapped_visual_link_count": len(unmapped),
        "all_visual_links_mapped": all_mapped,
        "all_visual_surfaces_globally_contained": all_contained,
        "strict_whole_robot_pass": bool(all_mapped and all_contained),
        "results": results,
        "unmapped": unmapped,
    }
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8")

    print("=" * 124)
    print("CONFIGURATION-AWARE GLOBAL-UNION SELF-FILTER AUDIT")
    print(f"reference robot : {reference_urdf}")
    print(f"self-filter URDF: {self_filter_urdf}")
    print(f"surface spacing : {args.surface_spacing*1000.0:.2f} mm")
    print(
        f"tolerance       : "
        f"{args.containment_tolerance*1000.0:.3f} mm")
    print(f"global primitives: {len(global_primitives)}")
    print("joint configuration:")
    for name in movable_joint_names:
        print(f"  {name}={resolved_joint_values[name]:.9f}")
    print("-" * 124)
    print(
        f"{'source visual link':30s} "
        f"{'samples':>10s} {'outside':>10s} "
        f"{'out%':>9s} {'max(mm)':>10s} "
        f"{'p95(mm)':>10s} {'contained':>10s}")

    for r in results:
        frac = (
            100.0 * r["outside_fraction"]
            if math.isfinite(r["outside_fraction"])
            else float("nan"))
        print(
            f"{r['source_link']:30s} "
            f"{r['surface_samples']:10d} "
            f"{r['outside_samples']:10d} "
            f"{frac:9.5f} "
            f"{1000.0*r['max_outside_m']:10.4f} "
            f"{1000.0*r['p95_outside_m']:10.4f} "
            f"{str(r['contained']):>10s}")

    print("-" * 124)
    print(f"unmapped_visual_links={len(unmapped)}")
    print(
        "all_visual_surfaces_globally_contained="
        f"{all_contained}")
    print(
        "strict_whole_robot_pass="
        f"{payload['strict_whole_robot_pass']}")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    print(f"WORST:    {worst_path}")
    print(f"UNMAPPED: {unmapped_path}")

    leaking = [r for r in results if not r["contained"]]
    if leaking:
        print("\nGLOBAL-UNION LEAKING LINKS (worst -> smallest):")
        for rank, r in enumerate(leaking, 1):
            print(
                f"  {rank:2d}. {r['source_link']}: "
                f"max={1000.0*r['max_outside_m']:.3f} mm, "
                f"p95={1000.0*r['p95_outside_m']:.3f} mm, "
                f"outside={r['outside_samples']}/"
                f"{r['surface_samples']} "
                f"({100.0*r['outside_fraction']:.5f}%)")
    else:
        print("\nGLOBAL-UNION LEAKING LINKS: none")

    if worst_rows:
        print("\nTop global-union outside visual-surface samples:")
        for r in worst_rows[:15]:
            print(
                f"  source={r['source_link']} "
                f"p_source=[{r['x_anchor']:.6f},"
                f"{r['y_anchor']:.6f},"
                f"{r['z_anchor']:.6f}] "
                f"p_world=[{r['x_world']:.6f},"
                f"{r['y_world']:.6f},"
                f"{r['z_world']:.6f}] "
                f"outside={1000.0*r['outside_m']:.3f} mm "
                f"nearest={r['nearest_collision_link']}/"
                f"{r['nearest_primitive']}")

    return 0 if payload["strict_whole_robot_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

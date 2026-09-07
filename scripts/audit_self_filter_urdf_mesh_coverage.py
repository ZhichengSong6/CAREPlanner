#!/usr/bin/env python3
"""
Offline coverage audit for CAREPlanner's dedicated self-filter URDF.

Question answered:
    Does the union of each link's dedicated self-filter collision primitives
    completely contain that link's visual mesh surface?

This script intentionally does NOT use body_samples.yaml.  It audits the
geometry that tof_fusion_self_filter_node.cpp now uses directly:
    src/arm_description/urdf/Arm_with_self_filter_collision.urdf

For each visual STL triangle, the surface is sampled at approximately
--surface-spacing (default 2 mm), including vertices, edges, and triangle
interiors.  Every sample is transformed by the URDF visual origin/scale and
evaluated against the exact union of that same link's collision
box/cylinder/sphere primitives.

Signed-distance convention:
    d <= 0 : covered by at least one self-filter primitive
    d >  0 : visual surface lies outside the self-filter union

Outputs:
    self_filter_mesh_coverage.csv
    self_filter_mesh_coverage.json
    self_filter_mesh_coverage_worst_points.csv
"""

import argparse
import csv
import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


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


def origin_transform(elem):
    if elem is None:
        return np.eye(3), np.zeros(3)
    return (
        rpy_to_matrix(parse_vec(elem.attrib.get("rpy"))),
        parse_vec(elem.attrib.get("xyz")),
    )


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

    # Defensive fallback for nested workspaces.
    matches = list((repo_root / "src").glob(f"**/{package}/{rel}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve {uri} from repo root {repo_root}")


def load_stl_triangles(path):
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + 50 * n
        if expected == len(data):
            tri = np.empty((n, 3, 3), dtype=np.float64)
            off = 84
            for i in range(n):
                # normal(12 bytes), vertices(36), attr(2)
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
        if not s.lower().startswith("vertex "):
            continue
        toks = s.split()
        if len(toks) >= 4:
            verts.append([float(toks[1]), float(toks[2]), float(toks[3])])
    if len(verts) % 3 != 0 or not verts:
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

    # Uniform barycentric lattice over the full triangle.
    out = []
    inv = 1.0 / n
    for i in range(n + 1):
        for j in range(n + 1 - i):
            u = i * inv
            v = j * inv
            out.append(a + u * (b - a) + v * (c - a))
    return np.asarray(out, dtype=np.float64)


class Primitive:
    def __init__(self, link, name, kind, R, t, size=None, radius=None, length=None):
        self.link = link
        self.name = name
        self.kind = kind
        self.R = R
        self.t = t
        self.size = size
        self.radius = radius
        self.length = length

    def signed_distance(self, points_link):
        # Row-vector equivalent of p_local = R^T (p_link - t).
        q = (points_link - self.t) @ self.R

        if self.kind == "box":
            d = np.abs(q) - 0.5 * self.size
            outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            inside = np.minimum(np.max(d, axis=1), 0.0)
            return outside + inside

        if self.kind == "cylinder":
            radial = np.linalg.norm(q[:, :2], axis=1) - self.radius
            axial = np.abs(q[:, 2]) - 0.5 * self.length
            outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
            inside = np.minimum(np.maximum(radial, axial), 0.0)
            return outside + inside

        if self.kind == "sphere":
            return np.linalg.norm(q, axis=1) - self.radius

        raise RuntimeError(self.kind)


def parse_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    links = {}

    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        visuals = []
        collisions = []

        for vi, visual in enumerate(link.findall("visual")):
            geom = visual.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            R, t = origin_transform(visual.find("origin"))
            scale = parse_vec(mesh.attrib.get("scale"), default=[1.0, 1.0, 1.0])
            visuals.append({
                "name": visual.attrib.get("name", f"visual_{vi}"),
                "filename": mesh.attrib["filename"],
                "scale": scale,
                "R": R,
                "t": t,
            })

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
                collisions.append(Primitive(
                    name, cname, "box", R, t,
                    size=parse_vec(box.attrib["size"])))
            elif cyl is not None:
                collisions.append(Primitive(
                    name, cname, "cylinder", R, t,
                    radius=float(cyl.attrib["radius"]),
                    length=float(cyl.attrib["length"])))
            elif sph is not None:
                collisions.append(Primitive(
                    name, cname, "sphere", R, t,
                    radius=float(sph.attrib["radius"])))
            else:
                raise RuntimeError(
                    f"{name}/{cname}: audit supports only box/cylinder/sphere")

        if visuals or collisions:
            links[name] = {"visuals": visuals, "collisions": collisions}

    return links


def audit_link(link_name, data, repo_root, spacing, tolerance, top_k):
    primitives = data["collisions"]
    visuals = data["visuals"]
    if not visuals:
        return None
    if not primitives:
        return {
            "link": link_name,
            "visual_count": len(visuals),
            "primitive_count": 0,
            "surface_samples": 0,
            "outside_samples": 0,
            "outside_fraction": float("nan"),
            "max_outside_m": float("inf"),
            "p95_outside_m": float("inf"),
            "mean_outside_m": float("inf"),
            "contained": False,
            "reason": "visual mesh present but no collision primitive",
            "worst": [],
        }

    total = 0
    outside = 0
    outside_d = []
    worst = []

    for vis in visuals:
        mesh_path = resolve_package_uri(vis["filename"], repo_root)
        triangles = load_stl_triangles(mesh_path)
        scale = vis["scale"]
        Rv, tv = vis["R"], vis["t"]

        for tri_idx, tri_mesh in enumerate(triangles):
            samples = triangle_surface_samples(tri_mesh * scale, spacing)
            samples_link = samples @ Rv.T + tv

            d_union = np.full(samples_link.shape[0], np.inf, dtype=np.float64)
            nearest_idx = np.full(samples_link.shape[0], -1, dtype=np.int32)
            for pi, prim in enumerate(primitives):
                d = prim.signed_distance(samples_link)
                better = d < d_union
                d_union[better] = d[better]
                nearest_idx[better] = pi

            total += len(d_union)
            mask = d_union > tolerance
            if np.any(mask):
                idxs = np.flatnonzero(mask)
                outside += len(idxs)
                vals = d_union[idxs]
                outside_d.extend(vals.tolist())

                for local_i, d in zip(idxs, vals):
                    p = samples_link[local_i]
                    prim = primitives[int(nearest_idx[local_i])]
                    rec = {
                        "link": link_name,
                        "visual": vis["name"],
                        "mesh": str(mesh_path),
                        "triangle": int(tri_idx),
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "z": float(p[2]),
                        "outside_m": float(d),
                        "nearest_primitive": prim.name,
                        "nearest_type": prim.kind,
                    }
                    worst.append(rec)

    worst.sort(key=lambda r: r["outside_m"], reverse=True)
    worst = worst[:top_k]

    if outside_d:
        arr = np.asarray(outside_d, dtype=np.float64)
        max_out = float(np.max(arr))
        p95_out = float(np.percentile(arr, 95))
        mean_out = float(np.mean(arr))
    else:
        max_out = 0.0
        p95_out = 0.0
        mean_out = 0.0

    return {
        "link": link_name,
        "visual_count": len(visuals),
        "primitive_count": len(primitives),
        "surface_samples": int(total),
        "outside_samples": int(outside),
        "outside_fraction": (outside / total) if total else float("nan"),
        "max_outside_m": max_out,
        "p95_outside_m": p95_out,
        "mean_outside_m": mean_out,
        "contained": bool(outside == 0),
        "reason": "",
        "worst": worst,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        default="/home/zhicheng/Project/CAREPlanner")
    ap.add_argument(
        "--urdf",
        default="src/arm_description/urdf/Arm_with_self_filter_collision.urdf")
    ap.add_argument(
        "--surface-spacing",
        type=float,
        default=0.002,
        help="approximate mesh-surface sampling spacing in meters")
    ap.add_argument(
        "--containment-tolerance",
        type=float,
        default=1e-4,
        help="positive signed distance <= this is treated as numerical boundary")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument(
        "--output-dir",
        default="outputs/self_filter_mesh_coverage")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = repo_root / urdf_path

    if args.surface_spacing <= 0:
        raise SystemExit("--surface-spacing must be positive")
    if args.containment_tolerance < 0:
        raise SystemExit("--containment-tolerance must be nonnegative")

    links = parse_urdf(urdf_path)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    worst_rows = []
    for link_name in sorted(links):
        result = audit_link(
            link_name,
            links[link_name],
            repo_root,
            args.surface_spacing,
            args.containment_tolerance,
            args.top_k)
        if result is None:
            continue
        results.append(result)
        worst_rows.extend(result["worst"])

    csv_path = out_dir / "self_filter_mesh_coverage.csv"
    json_path = out_dir / "self_filter_mesh_coverage.json"
    worst_path = out_dir / "self_filter_mesh_coverage_worst_points.csv"

    fields = [
        "link", "visual_count", "primitive_count", "surface_samples",
        "outside_samples", "outside_fraction", "max_outside_m",
        "p95_outside_m", "mean_outside_m", "contained", "reason",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    payload = {
        "urdf": str(urdf_path),
        "surface_spacing_m": args.surface_spacing,
        "containment_tolerance_m": args.containment_tolerance,
        "all_contained": all(r["contained"] for r in results),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    worst_fields = [
        "link", "visual", "mesh", "triangle", "x", "y", "z",
        "outside_m", "nearest_primitive", "nearest_type",
    ]
    with worst_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        w.writerows(worst_rows)

    print("=" * 96)
    print("SELF-FILTER URDF VISUAL-MESH COVERAGE AUDIT")
    print(f"URDF: {urdf_path}")
    print(f"surface spacing: {args.surface_spacing * 1000.0:.2f} mm")
    print(f"containment tolerance: {args.containment_tolerance * 1000.0:.3f} mm")
    print("-" * 96)
    print(
        f"{'link':18s} {'samples':>10s} {'outside':>10s} "
        f"{'outside%':>10s} {'max_out(mm)':>12s} {'contained':>10s}")
    for r in results:
        frac = 100.0 * r["outside_fraction"] if math.isfinite(r["outside_fraction"]) else float("nan")
        print(
            f"{r['link']:18s} {r['surface_samples']:10d} "
            f"{r['outside_samples']:10d} {frac:10.5f} "
            f"{1000.0*r['max_outside_m']:12.4f} "
            f"{str(r['contained']):>10s}")

    print("-" * 96)
    print(f"all_contained={payload['all_contained']}")
    print(f"CSV:   {csv_path}")
    print(f"JSON:  {json_path}")
    print(f"WORST: {worst_path}")

    if worst_rows:
        print("\nTop outside samples:")
        for r in sorted(worst_rows, key=lambda x: x["outside_m"], reverse=True)[:10]:
            print(
                f"  {r['link']} p=[{r['x']:.6f},{r['y']:.6f},{r['z']:.6f}] "
                f"outside={1000.0*r['outside_m']:.3f} mm "
                f"nearest={r['nearest_primitive']}")

    return 0 if payload["all_contained"] else 1


if __name__ == "__main__":
    sys.exit(main())

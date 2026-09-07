#!/usr/bin/env python3
"""
Audit whether the dedicated CAREPlanner self-filter geometry contains the
visual geometry of the ACTUAL Gazebo robot.

Reference visual geometry:
    src/arm_description/urdf/Arm.urdf

Self-filter geometry:
    src/arm_description/urdf/Arm_with_self_filter_collision.urdf

This distinction matters because Arm.urdf contains auxiliary fixed sensor,
camera, ToF and EE visual links that are intentionally stripped from the
dedicated self-filter URDF.  For every visual link in Arm.urdf, this script
walks upward through fixed joints to the nearest ancestor that has dedicated
self-filter collision primitives, transforms that visual mesh into the
ancestor frame, and evaluates it against the exact union of those primitives.

Thus the audit answers the actual runtime question:
    Is every rendered robot visual surface that can be seen by Gazebo depth
    cameras inside the dedicated self-filter envelope?

Surface audit:
  * binary or ASCII STL supported without trimesh;
  * triangle vertices, edges and interiors are sampled on a barycentric lattice;
  * default surface spacing = 2 mm;
  * exact signed distance to box/cylinder/sphere primitives;
  * no runtime padding is assumed.

Signed-distance convention:
    d <= 0 : covered by dedicated self-filter geometry
    d >  0 : rendered visual surface lies outside the self-filter union

Outputs:
    self_filter_mesh_coverage.csv
    self_filter_mesh_coverage.json
    self_filter_mesh_coverage_worst_points.csv
    self_filter_mesh_unmapped_visual_links.csv
"""

import argparse
import csv
import json
import math
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
    # URDF fixed-axis RPY = Rz(yaw) Ry(pitch) Rx(roll).
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


def compose(T_ab, T_bc):
    """Return T_ac for p_a = T_ab(T_bc(p_c))."""
    Ra, ta = T_ab
    Rb, tb = T_bc
    return Ra @ Rb, Ra @ tb + ta


def apply_transform(points, T):
    R, t = T
    return points @ R.T + t


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


class Primitive:
    def __init__(
            self, link, name, kind, R, t,
            size=None, radius=None, length=None):
        self.link = link
        self.name = name
        self.kind = kind
        self.R = R
        self.t = t
        self.size = size
        self.radius = radius
        self.length = length

    def signed_distance(self, points_anchor):
        # Primitive origin T_anchor_primitive.
        # Row-vector local coordinates:
        # p_primitive = (p_anchor - t) @ R.
        q = (points_anchor - self.t) @ self.R

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


def parse_reference_robot(path):
    root = ET.parse(path).getroot()

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
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue

        child_name = child.attrib["link"]
        parent_joint[child_name] = {
            "name": joint.attrib.get("name", ""),
            "type": joint.attrib.get("type", ""),
            "parent": parent.attrib["link"],
            "T_parent_child": origin_transform(joint.find("origin")),
        }

    return visuals_by_link, parent_joint


def find_self_filter_anchor(
        source_link,
        self_filter_links,
        parent_joint):
    """
    Map a rendered reference link into a dedicated self-filter link.

    Only fixed joints may be collapsed.  If a non-fixed joint is encountered
    before reaching a dedicated collision link, the mapping is invalid.
    """
    current = source_link
    T_current_source = (np.eye(3), np.zeros(3))
    chain = []

    while True:
        if current in self_filter_links:
            return {
                "ok": True,
                "anchor": current,
                "T_anchor_source": T_current_source,
                "fixed_chain": chain,
                "reason": "",
            }

        joint = parent_joint.get(current)
        if joint is None:
            return {
                "ok": False,
                "anchor": "",
                "T_anchor_source": None,
                "fixed_chain": chain,
                "reason": "no parent joint before self-filter anchor",
            }

        if joint["type"] != "fixed":
            return {
                "ok": False,
                "anchor": "",
                "T_anchor_source": None,
                "fixed_chain": chain,
                "reason": (
                    f"encountered non-fixed joint {joint['name']} "
                    f"type={joint['type']} before self-filter anchor"),
            }

        # T_parent_source = T_parent_current * T_current_source.
        T_current_source = compose(
            joint["T_parent_child"],
            T_current_source)
        chain.append(joint["name"])
        current = joint["parent"]


def audit_visual_link(
        source_link,
        visuals,
        mapping,
        primitives,
        repo_root,
        spacing,
        tolerance,
        top_k):
    total = 0
    outside = 0
    outside_distances = []
    worst = []

    T_anchor_source = mapping["T_anchor_source"]

    for vis in visuals:
        mesh_path = resolve_package_uri(vis["filename"], repo_root)
        triangles = load_stl_triangles(mesh_path)

        # Full mesh -> visual origin -> source link -> fixed-chain anchor.
        T_anchor_visual = compose(
            T_anchor_source,
            vis["T_link_visual"])

        for tri_idx, tri_mesh in enumerate(triangles):
            tri_scaled = tri_mesh * vis["scale"]
            samples_visual = triangle_surface_samples(
                tri_scaled,
                spacing)
            samples_anchor = apply_transform(
                samples_visual,
                T_anchor_visual)

            d_union = np.full(
                samples_anchor.shape[0],
                np.inf,
                dtype=np.float64)
            nearest_idx = np.full(
                samples_anchor.shape[0],
                -1,
                dtype=np.int32)

            for pi, prim in enumerate(primitives):
                d = prim.signed_distance(samples_anchor)
                better = d < d_union
                d_union[better] = d[better]
                nearest_idx[better] = pi

            total += len(d_union)
            mask = d_union > tolerance
            if not np.any(mask):
                continue

            idxs = np.flatnonzero(mask)
            vals = d_union[idxs]
            outside += len(idxs)
            outside_distances.extend(vals.tolist())

            for local_i, d in zip(idxs, vals):
                p = samples_anchor[local_i]
                prim = primitives[int(nearest_idx[local_i])]
                worst.append({
                    "source_link": source_link,
                    "anchor_link": mapping["anchor"],
                    "visual": vis["name"],
                    "mesh": str(mesh_path),
                    "triangle": int(tri_idx),
                    "x_anchor": float(p[0]),
                    "y_anchor": float(p[1]),
                    "z_anchor": float(p[2]),
                    "outside_m": float(d),
                    "nearest_primitive": prim.name,
                    "nearest_type": prim.kind,
                    "fixed_chain": ";".join(mapping["fixed_chain"]),
                })

    worst.sort(key=lambda r: r["outside_m"], reverse=True)
    worst = worst[:top_k]

    if outside_distances:
        arr = np.asarray(outside_distances, dtype=np.float64)
        max_out = float(np.max(arr))
        p95_out = float(np.percentile(arr, 95))
        mean_out = float(np.mean(arr))
    else:
        max_out = p95_out = mean_out = 0.0

    return {
        "source_link": source_link,
        "anchor_link": mapping["anchor"],
        "fixed_chain": ";".join(mapping["fixed_chain"]),
        "visual_count": len(visuals),
        "primitive_count": len(primitives),
        "surface_samples": int(total),
        "outside_samples": int(outside),
        "outside_fraction": outside / total if total else float("nan"),
        "max_outside_m": max_out,
        "p95_outside_m": p95_out,
        "mean_outside_m": mean_out,
        "contained": bool(outside == 0),
        "worst": worst,
    }


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
        help="d <= tolerance is treated as contained (default 0.1 mm)")
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

    collisions = parse_self_filter_collisions(self_filter_urdf)
    visuals, parent_joint = parse_reference_robot(reference_urdf)

    results = []
    worst_rows = []
    unmapped = []

    for source_link in sorted(visuals):
        mapping = find_self_filter_anchor(
            source_link,
            set(collisions),
            parent_joint)

        if not mapping["ok"]:
            unmapped.append({
                "source_link": source_link,
                "reason": mapping["reason"],
                "fixed_chain": ";".join(mapping["fixed_chain"]),
            })
            continue

        result = audit_visual_link(
            source_link,
            visuals[source_link],
            mapping,
            collisions[mapping["anchor"]],
            repo_root,
            args.surface_spacing,
            args.containment_tolerance,
            args.top_k)
        results.append(result)
        worst_rows.extend(result["worst"])

    csv_path = out_dir / "self_filter_mesh_coverage.csv"
    json_path = out_dir / "self_filter_mesh_coverage.json"
    worst_path = out_dir / "self_filter_mesh_coverage_worst_points.csv"
    unmapped_path = out_dir / "self_filter_mesh_unmapped_visual_links.csv"

    fields = [
        "source_link", "anchor_link", "fixed_chain",
        "visual_count", "primitive_count",
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
        "outside_m", "nearest_primitive", "nearest_type", "fixed_chain",
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
        "reference_urdf": str(reference_urdf),
        "self_filter_urdf": str(self_filter_urdf),
        "surface_spacing_m": args.surface_spacing,
        "containment_tolerance_m": args.containment_tolerance,
        "reference_visual_link_count": len(visuals),
        "self_filter_collision_link_count": len(collisions),
        "mapped_visual_link_count": len(results),
        "unmapped_visual_link_count": len(unmapped),
        "all_visual_links_mapped": all_mapped,
        "all_mapped_visual_surfaces_contained": all_contained,
        "strict_whole_robot_pass": bool(all_mapped and all_contained),
        "results": results,
        "unmapped": unmapped,
    }
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8")

    print("=" * 112)
    print("DEDICATED SELF-FILTER VS ACTUAL GAZEBO VISUAL-MESH AUDIT")
    print(f"reference robot : {reference_urdf}")
    print(f"self-filter URDF: {self_filter_urdf}")
    print(f"surface spacing : {args.surface_spacing*1000.0:.2f} mm")
    print(
        f"tolerance       : "
        f"{args.containment_tolerance*1000.0:.3f} mm")
    print("-" * 112)
    print(
        f"{'source visual link':30s} {'anchor':15s} "
        f"{'samples':>10s} {'outside':>10s} "
        f"{'out%':>9s} {'max(mm)':>10s} {'contained':>10s}")

    for r in results:
        frac = (
            100.0 * r["outside_fraction"]
            if math.isfinite(r["outside_fraction"])
            else float("nan"))
        print(
            f"{r['source_link']:30s} "
            f"{r['anchor_link']:15s} "
            f"{r['surface_samples']:10d} "
            f"{r['outside_samples']:10d} "
            f"{frac:9.5f} "
            f"{1000.0*r['max_outside_m']:10.4f} "
            f"{str(r['contained']):>10s}")

    print("-" * 112)
    print(f"unmapped_visual_links={len(unmapped)}")
    for r in unmapped:
        print(
            f"  UNMAPPED {r['source_link']}: {r['reason']} "
            f"chain={r['fixed_chain']}")

    print(
        "all_mapped_visual_surfaces_contained="
        f"{all_contained}")
    print(f"strict_whole_robot_pass={payload['strict_whole_robot_pass']}")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    print(f"WORST:    {worst_path}")
    print(f"UNMAPPED: {unmapped_path}")

    if worst_rows:
        print("\nTop outside visual-surface samples:")
        for r in sorted(
                worst_rows,
                key=lambda x: x["outside_m"],
                reverse=True)[:15]:
            print(
                f"  source={r['source_link']} "
                f"anchor={r['anchor_link']} "
                f"p=[{r['x_anchor']:.6f},"
                f"{r['y_anchor']:.6f},"
                f"{r['z_anchor']:.6f}] "
                f"outside={1000.0*r['outside_m']:.3f} mm "
                f"nearest={r['nearest_primitive']}")

    return 0 if payload["strict_whole_robot_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

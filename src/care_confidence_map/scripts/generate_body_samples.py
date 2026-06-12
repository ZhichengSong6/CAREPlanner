#!/usr/bin/env python3

import argparse
import math
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_vec(s, n=3):
    if s is None:
        return [0.0] * n

    vals = [float(x) for x in s.replace(",", " ").split()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} values, got {len(vals)} from {s!r}")

    return vals


def rpy_to_R(rpy):
    r, p, y = rpy

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def mat_vec(R, v):
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def cylinder_samples(radius, length, spacing, margin, radius_scale, center_span_scale):
    # URDF cylinder axis is local +Z.
    #
    # We first compute the number of samples from the original cylinder length
    # and spacing. This keeps the sample count approximately unchanged.
    #
    # Then we place those N samples inside a shortened axial span:
    #
    #   center_span = length * center_span_scale
    #
    # This makes center-to-center spacing smaller while keeping:
    #   - sample count mostly unchanged
    #   - sample radius unchanged
    #
    # It also avoids placing samples too close to cylinder endpoints, which was
    # the main reason endpoint regions looked overly conservative.
    if length <= 1e-9:
        return [([0.0, 0.0, 0.0], radius * radius_scale + margin)]

    center_span_scale = clamp(center_span_scale, 0.05, 1.0)

    n = max(1, int(math.ceil(length / spacing)))

    center_span = length * center_span_scale
    dz = center_span / n
    z_min = -0.5 * center_span

    sample_radius = radius * radius_scale + margin

    out = []
    for i in range(n):
        z = z_min + (i + 0.5) * dz
        out.append(([0.0, 0.0, z], sample_radius))

    return out


def box_samples(size, cell, margin, radius_scale):
    sx, sy, sz = size

    # Sparse voxel-like samples inside the box.
    #
    # The number of samples is controlled by box_cell_size.
    # The radius is controlled by box_radius_scale.
    #
    # We keep box_cell_size unchanged to preserve sample count, and tune only
    # box_radius_scale to make the visualization/risk samples slightly smaller.
    nx = max(1, int(math.ceil(sx / cell)))
    ny = max(1, int(math.ceil(sy / cell)))
    nz = max(1, int(math.ceil(sz / cell)))

    dx, dy, dz = sx / nx, sy / ny, sz / nz

    radius = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
    radius = radius * radius_scale + margin

    xs = [(-0.5 * sx + (i + 0.5) * dx) for i in range(nx)]
    ys = [(-0.5 * sy + (i + 0.5) * dy) for i in range(ny)]
    zs = [(-0.5 * sz + (i + 0.5) * dz) for i in range(nz)]

    out = []
    for x in xs:
        for y in ys:
            for z in zs:
                out.append(([x, y, z], radius))

    return out


def should_include_for_risk(link_name):
    # Fixed base is useful for visualization/debug, but should not be counted
    # in whole-body trajectory risk.
    return link_name != "base_link"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True)

    # Cylinder parameters.
    parser.add_argument("--cylinder-spacing", type=float, default=0.06)
    parser.add_argument("--cylinder-safety-margin", type=float, default=0.005)
    parser.add_argument("--cylinder-radius-scale", type=float, default=1.0)

    # New:
    #   1.0 = use the full cylinder axial span for sample centers
    #   0.8 = place the same number of samples inside the middle 80% span
    #
    # This reduces endpoint conservativeness and shrinks center spacing without
    # changing the sample count.
    parser.add_argument("--cylinder-center-span-scale", type=float, default=0.80)

    # Box parameters.
    parser.add_argument("--box-cell-size", type=float, default=0.06)
    parser.add_argument("--box-safety-margin", type=float, default=0.002)
    parser.add_argument("--box-radius-scale", type=float, default=0.80)

    args = parser.parse_args()

    tree = ET.parse(args.urdf)
    root = tree.getroot()

    links_yaml = []
    total = 0
    total_risk = 0

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        collisions = link.findall("collision")

        if not collisions:
            continue

        link_samples = []

        for cidx, col in enumerate(collisions):
            origin = col.find("origin")

            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None)
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None)

            R = rpy_to_R(rpy)

            geom = col.find("geometry")
            if geom is None:
                continue

            local_samples = []
            source_type = None

            cyl = geom.find("cylinder")
            box = geom.find("box")

            if cyl is not None:
                source_type = "cylinder"

                radius = float(cyl.attrib["radius"])
                length = float(cyl.attrib["length"])

                local_samples = cylinder_samples(
                    radius=radius,
                    length=length,
                    spacing=args.cylinder_spacing,
                    margin=args.cylinder_safety_margin,
                    radius_scale=args.cylinder_radius_scale,
                    center_span_scale=args.cylinder_center_span_scale,
                )

            elif box is not None:
                source_type = "box"

                size = parse_vec(box.attrib["size"])

                local_samples = box_samples(
                    size=size,
                    cell=args.box_cell_size,
                    margin=args.box_safety_margin,
                    radius_scale=args.box_radius_scale,
                )

            else:
                continue

            for local_center, sample_radius in local_samples:
                center_link = add(xyz, mat_vec(R, local_center))

                link_samples.append(
                    {
                        "center": [
                            round(center_link[0], 6),
                            round(center_link[1], 6),
                            round(center_link[2], 6),
                        ],
                        "radius": round(sample_radius, 6),
                        "source_type": source_type,
                        "source_collision_index": cidx,
                    }
                )

        if link_samples:
            include = should_include_for_risk(link_name)

            links_yaml.append(
                {
                    "link_name": link_name,
                    "frame": link_name,
                    "include_for_risk": bool(include),
                    "samples": link_samples,
                }
            )

            total += len(link_samples)

            if include:
                total_risk += len(link_samples)

    data = {
        "body_sampling": {
            "frame_convention": "sample centers are expressed in each link frame",
            "source_urdf": str(Path(args.urdf).name),

            "cylinder_spacing": args.cylinder_spacing,
            "cylinder_safety_margin": args.cylinder_safety_margin,
            "cylinder_radius_scale": args.cylinder_radius_scale,
            "cylinder_center_span_scale": args.cylinder_center_span_scale,
            "cylinder_endpoint_sampling": False,

            "box_cell_size": args.box_cell_size,
            "box_safety_margin": args.box_safety_margin,
            "box_radius_scale": args.box_radius_scale,

            "total_samples": total,
            "total_risk_samples": total_risk,
            "links": links_yaml,
        }
    }

    with open(args.output, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {args.output}")
    print(f"total_samples={total}, total_risk_samples={total_risk}")

    for l in links_yaml:
        print(
            f"  {l['link_name']}: {len(l['samples'])}, "
            f"risk={l['include_for_risk']}"
        )


if __name__ == "__main__":
    main()


#     --cylinder-spacing 0.06
# --cylinder-safety-margin 0.005
# --cylinder-radius-scale 1.0
# --cylinder-center-span-scale 0.70
# --box-cell-size 0.06
# --box-safety-margin 0.002
# --box-radius-scale 0.80
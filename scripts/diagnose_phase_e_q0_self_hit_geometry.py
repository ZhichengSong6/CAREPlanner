#!/usr/bin/env python3
"""Identify which URDF visual mesh produces a watched ToF return at q=0.

This is a planner-independent Phase-E diagnostic. It subscribes directly to one
raw Gazebo ToF PointCloud2 topic, selects finite returns whose transformed
base-frame endpoints fall inside a configured 5-cm watched voxel, and ray-casts
those measurements against every robot visual STL referenced by Arm.urdf.

For each watched depth ray it reports:
  - raw sensor-frame point and base-frame endpoint
  - measured range
  - first robot visual mesh intersection along that ray
  - intersection distance and error relative to the measured return
  - ground-plane intersection as a control

If a robot visual mesh intersects at nearly the measured range, that mesh is
what Gazebo is rendering into the ToF image. If no robot visual matches, the
return is not explained by the URDF visual geometry and we should investigate
Gazebo sensor/rendering behavior instead.

No planner, confidence-map, GCDF, VBC, or occupancy semantics are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


@dataclass
class VisualMesh:
    link_name: str
    mesh_uri: str
    mesh_path: Path
    xyz: np.ndarray
    rpy: np.ndarray
    scale: np.ndarray


def parse_vec(text: Optional[str], default: Tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    vals = [float(x) for x in text.split()]
    if len(vals) != 3:
        raise ValueError(f"expected 3-vector, got {text!r}")
    return np.asarray(vals, dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = [float(x) for x in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def transform_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_matrix(rpy)
    T[:3, 3] = xyz
    return T


def quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=float)
    n = float(np.dot(q, q))
    if n < 1e-16:
        return np.eye(3, dtype=float)
    q *= math.sqrt(2.0 / n)
    Q = np.outer(q, q)
    return np.asarray([
        [1.0 - Q[1, 1] - Q[2, 2], Q[0, 1] - Q[2, 3], Q[0, 2] + Q[1, 3]],
        [Q[0, 1] + Q[2, 3], 1.0 - Q[0, 0] - Q[2, 2], Q[1, 2] - Q[0, 3]],
        [Q[0, 2] - Q[1, 3], Q[1, 2] + Q[0, 3], 1.0 - Q[0, 0] - Q[1, 1]],
    ], dtype=float)


def tf_to_matrix(msg) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    T = np.eye(4, dtype=float)
    T[:3, :3] = quat_matrix(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def resolve_mesh_uri(uri: str, repo: Path) -> Path:
    prefix = "package://arm_description/"
    if uri.startswith(prefix):
        return (repo / "src" / "arm_description" / uri[len(prefix):]).resolve()
    if uri.startswith("file://"):
        return Path(uri[len("file://"):]).resolve()
    return Path(uri).resolve()


def load_visual_meshes(urdf_path: Path, repo: Path) -> List[VisualMesh]:
    root = ET.parse(str(urdf_path)).getroot()
    out: List[VisualMesh] = []
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            uri = mesh.attrib.get("filename", "")
            if not uri:
                continue
            origin = visual.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0))
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0))
            scale = parse_vec(mesh.attrib.get("scale"), (1, 1, 1))
            path = resolve_mesh_uri(uri, repo)
            if path.is_file():
                out.append(VisualMesh(link_name, uri, path, xyz, rpy, scale))
    return out


def load_stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    # Binary STL: 80-byte header + uint32 count + 50 bytes per triangle.
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            tris = np.empty((n, 3, 3), dtype=np.float64)
            off = 84
            for i in range(n):
                vals = struct.unpack_from("<12fH", data, off)
                tris[i, 0] = vals[3:6]
                tris[i, 1] = vals[6:9]
                tris[i, 2] = vals[9:12]
                off += 50
            return tris

    # ASCII STL fallback.
    vertices: List[List[float]] = []
    for raw in data.decode("utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("vertex "):
            vals = line.split()
            if len(vals) >= 4:
                vertices.append([float(vals[1]), float(vals[2]), float(vals[3])])
    if len(vertices) % 3 != 0 or not vertices:
        raise RuntimeError(f"could not parse STL: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape((-1, 3, 3))


def ray_aabb_hit(origin: np.ndarray, direction: np.ndarray,
                 bmin: np.ndarray, bmax: np.ndarray) -> bool:
    tmin, tmax = -np.inf, np.inf
    for k in range(3):
        if abs(direction[k]) < 1e-12:
            if origin[k] < bmin[k] or origin[k] > bmax[k]:
                return False
            continue
        t1 = (bmin[k] - origin[k]) / direction[k]
        t2 = (bmax[k] - origin[k]) / direction[k]
        lo, hi = min(t1, t2), max(t1, t2)
        tmin = max(tmin, lo)
        tmax = min(tmax, hi)
        if tmax < max(tmin, 0.0):
            return False
    return True


def ray_triangles_first(origin: np.ndarray, direction: np.ndarray,
                        tris: np.ndarray) -> Optional[float]:
    if tris.size == 0:
        return None
    v0 = tris[:, 0, :]
    e1 = tris[:, 1, :] - v0
    e2 = tris[:, 2, :] - v0
    d = np.broadcast_to(direction.reshape(1, 3), e2.shape)
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    mask = np.abs(det) > 1e-10
    if not np.any(mask):
        return None
    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]
    tvec = origin.reshape(1, 3) - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    mask &= (u >= -1e-9) & (u <= 1.0 + 1e-9)
    if not np.any(mask):
        return None
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,ij->i", d, qvec) * inv_det
    mask &= (v >= -1e-9) & ((u + v) <= 1.0 + 1e-9)
    if not np.any(mask):
        return None
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    mask &= t > 1e-6
    if not np.any(mask):
        return None
    return float(np.min(t[mask]))


class Diagnostic:
    def __init__(self, args):
        self.args = args
        self.repo = Path(args.repo).resolve()
        self.urdf = Path(args.urdf).resolve()
        self.visuals = load_visual_meshes(self.urdf, self.repo)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.mesh_cache: Dict[Path, np.ndarray] = {}
        self.records: List[dict] = []
        self.done = False
        self.start_wall = time.time()
        self.sub = rospy.Subscriber(args.topic, PointCloud2, self.cb, queue_size=1)
        rospy.logwarn("[q0-self-hit] loaded %d robot visual meshes from %s",
                      len(self.visuals), self.urdf)

    def load_mesh(self, visual: VisualMesh) -> np.ndarray:
        if visual.mesh_path not in self.mesh_cache:
            tris = load_stl_triangles(visual.mesh_path)
            tris = tris * visual.scale.reshape(1, 1, 3)
            self.mesh_cache[visual.mesh_path] = tris
        return self.mesh_cache[visual.mesh_path]

    def lookup(self, target: str, source: str, stamp) -> Optional[np.ndarray]:
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, stamp, rospy.Duration(self.args.tf_timeout))
            return tf_to_matrix(tf)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "[q0-self-hit] TF %s <- %s failed: %s",
                                   target, source, exc)
            return None

    def mesh_intersections(self, o_base: np.ndarray, d_base: np.ndarray,
                           stamp) -> List[dict]:
        hits: List[dict] = []
        for visual in self.visuals:
            T_base_link = self.lookup(self.args.base_frame, visual.link_name, stamp)
            if T_base_link is None:
                continue
            T_link_visual = transform_matrix(visual.xyz, visual.rpy)
            T_base_mesh = T_base_link @ T_link_visual
            R = T_base_mesh[:3, :3]
            t = T_base_mesh[:3, 3]
            o_mesh = R.T @ (o_base - t)
            d_mesh = R.T @ d_base
            tris = self.load_mesh(visual)
            flat = tris.reshape((-1, 3))
            bmin = np.min(flat, axis=0)
            bmax = np.max(flat, axis=0)
            if not ray_aabb_hit(o_mesh, d_mesh, bmin, bmax):
                continue
            dist = ray_triangles_first(o_mesh, d_mesh, tris)
            if dist is None:
                continue
            p_base = o_base + dist * d_base
            hits.append({
                "link": visual.link_name,
                "mesh": str(visual.mesh_path),
                "mesh_uri": visual.mesh_uri,
                "distance_m": dist,
                "point_base": p_base.tolist(),
            })
        hits.sort(key=lambda x: x["distance_m"])
        return hits

    def cb(self, msg: PointCloud2):
        if self.done:
            return
        source_frame = msg.header.frame_id or self.args.sensor_frame
        T_base_sensor = self.lookup(self.args.base_frame, source_frame, msg.header.stamp)
        if T_base_sensor is None:
            return
        o_base = T_base_sensor[:3, 3].copy()
        R = T_base_sensor[:3, :3]

        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            ps = np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=float)
            if not np.all(np.isfinite(ps)):
                continue
            pb = R @ ps + o_base
            if np.any(np.abs(pb - self.args.watch_center) > self.args.watch_half_extent):
                continue

            measured_range = float(np.linalg.norm(ps))
            if measured_range <= 1e-9:
                continue
            d_base = (pb - o_base) / measured_range
            hits = self.mesh_intersections(o_base, d_base, msg.header.stamp)

            ground_dist = None
            if abs(d_base[2]) > 1e-12:
                tg = -o_base[2] / d_base[2]
                if tg > 0.0:
                    ground_dist = float(tg)

            rec = {
                "stamp": msg.header.stamp.to_sec(),
                "source_frame": source_frame,
                "raw_sensor": ps.tolist(),
                "endpoint_base": pb.tolist(),
                "sensor_origin_base": o_base.tolist(),
                "measured_range_m": measured_range,
                "measured_forward_depth_m": float(ps[2]),
                "ground_intersection_distance_m": ground_dist,
                "robot_visual_hits": hits[: self.args.top_hits],
            }
            if hits:
                rec["first_robot_visual"] = hits[0]
                rec["first_robot_visual_error_m"] = hits[0]["distance_m"] - measured_range
            else:
                rec["first_robot_visual"] = None
                rec["first_robot_visual_error_m"] = None
            self.records.append(rec)

            first = hits[0] if hits else None
            print("\n[Q0_SELF_HIT_RAY]")
            print(f"stamp={rec['stamp']:.6f}")
            print(f"raw_sensor={rec['raw_sensor']}")
            print(f"endpoint_base={rec['endpoint_base']}")
            print(f"measured_range_m={measured_range:.6f}")
            print(f"forward_depth_m={ps[2]:.6f}")
            if first:
                print("FIRST_ROBOT_VISUAL "
                      f"link={first['link']} "
                      f"mesh={Path(first['mesh']).name} "
                      f"distance_m={first['distance_m']:.6f} "
                      f"error_vs_depth_m={first['distance_m'] - measured_range:+.6f}")
            else:
                print("FIRST_ROBOT_VISUAL none")
            print(f"ground_distance_m={ground_dist}")
            for rank, hit in enumerate(hits[: self.args.top_hits], 1):
                print(f"  hit#{rank}: link={hit['link']} "
                      f"mesh={Path(hit['mesh']).name} "
                      f"distance={hit['distance_m']:.6f}")

            if len(self.records) >= self.args.max_rays:
                self.done = True
                break

    def run(self):
        rate = rospy.Rate(50)
        deadline = time.time() + self.args.duration
        while not rospy.is_shutdown() and time.time() < deadline and not self.done:
            rate.sleep()
        self.done = True

        out = {
            "topic": self.args.topic,
            "sensor_frame": self.args.sensor_frame,
            "base_frame": self.args.base_frame,
            "watch_center": self.args.watch_center.tolist(),
            "watch_half_extent": self.args.watch_half_extent,
            "ray_count": len(self.records),
            "records": self.records,
        }
        Path(self.args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(self.args.output_json).write_text(json.dumps(out, indent=2))

        print("\n================ Q0 SELF-HIT GEOMETRY SUMMARY ================")
        print(f"watched_rays={len(self.records)}")
        print(f"output_json={self.args.output_json}")
        counts: Dict[str, int] = {}
        errors: Dict[str, List[float]] = {}
        for rec in self.records:
            first = rec.get("first_robot_visual")
            if not first:
                key = "NONE"
            else:
                key = f"{first['link']}::{Path(first['mesh']).name}"
                err = rec.get("first_robot_visual_error_m")
                if err is not None:
                    errors.setdefault(key, []).append(float(err))
            counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if key in errors and errors[key]:
                e = np.asarray(errors[key], dtype=float)
                print(f"{key}: count={count} "
                      f"median_error_m={np.median(e):+.6f} "
                      f"max_abs_error_m={np.max(np.abs(e)):.6f}")
            else:
                print(f"{key}: count={count}")
        print("===============================================================")


def main():
    repo_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(repo_default))
    ap.add_argument("--urdf", default=str(repo_default / "src/arm_description/urdf/Arm.urdf"))
    ap.add_argument("--topic", default="/link2_sensor2/tof/cloud")
    ap.add_argument("--sensor-frame", default="link2_sensor2_tof_link")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--watch-x", type=float, default=0.20)
    ap.add_argument("--watch-y", type=float, default=-0.05)
    ap.add_argument("--watch-z", type=float, default=0.45)
    ap.add_argument("--watch-half-extent", type=float, default=0.025)
    ap.add_argument("--max-rays", type=int, default=12)
    ap.add_argument("--top-hits", type=int, default=5)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--tf-timeout", type=float, default=0.05)
    ap.add_argument("--output-json",
                    default=str(repo_default / "outputs/phase_e_q0_self_hit_geometry.json"))
    args = ap.parse_args()
    args.watch_center = np.asarray([args.watch_x, args.watch_y, args.watch_z], dtype=float)

    rospy.init_node("phase_e_q0_self_hit_geometry_diagnostic", anonymous=True)
    Diagnostic(args).run()


if __name__ == "__main__":
    main()

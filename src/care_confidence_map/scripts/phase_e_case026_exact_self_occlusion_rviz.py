#!/usr/bin/env python3
"""Case-026 exact self-occlusion RViz diagnostic.

Diagnostic-only: publishes the frozen q_vis pose, S4 FOV, target ray, the
self-filter primitive hit, and an exact ray/triangle intersection against the
reference STL visual meshes.  It does not start Gazebo, perception, or planner.

The exact-mesh test answers the narrow question left by the conservative
primitive test:
  - primitive hit + exact STL hit -> real self-occlusion is confirmed
  - primitive hit + exact STL clear -> primitive approximation is too conservative
"""

import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rospy
import tf.transformations as tft

from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from urdf_parser_py.urdf import URDF


SRC_DIR = Path(__file__).resolve().parents[2]
VIS_SCRIPT_DIR = SRC_DIR / "care_visibility_cdf" / "scripts"
if str(VIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_SCRIPT_DIR))

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    find_chain_joints,
    fk_transform,
    make_transform,
    sensor_margin,
)
from check_visibility_self_occlusion import (  # noqa: E402
    load_collision_primitives,
    q_row_to_map,
    raycast_self_occlusion,
)


CASE026_Q_VIS = [
    -0.26947852969169617,
    0.750813901424408,
    -0.2667731046676636,
    -1.8708069324493408,
    0.1902204304933548,
    -0.1728929728269577,
    -0.3792421519756317,
]
CASE026_TARGET = [
    0.10000000149011612,
    0.05000000074505806,
    0.15000000596046448,
]
CASE026_SENSOR = "link4_sensor1_tof_link"
BODY_VISUAL_LINKS = [
    "base_link", "link1", "link2", "link3", "link4",
    "wrist_link1", "wrist_link2", "wrist_link3",
]


def rgba(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def point(v):
    p = Point()
    p.x, p.y, p.z = map(float, v)
    return p


def vec(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in text.split()]
    if len(vals) != 3:
        raise ValueError(text)
    return np.asarray(vals, dtype=np.float64)


def resolve_mesh_uri(uri, repo):
    if uri.startswith("package://"):
        rest = uri[len("package://"):]
        pkg, rel = rest.split("/", 1)
        candidate = os.path.join(repo, "src", pkg, rel)
        if os.path.isfile(candidate):
            return candidate
    if uri.startswith("file://"):
        p = uri[len("file://"):]
        if os.path.isfile(p):
            return p
    if os.path.isabs(uri) and os.path.isfile(uri):
        return uri
    raise RuntimeError("cannot resolve mesh URI: " + uri)


def load_stl_triangles(path):
    """Load binary or ASCII STL into [N,3,3] float64 triangles."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)

    is_binary = False
    if len(head) >= 84:
        n = struct.unpack("<I", head[80:84])[0]
        if 84 + 50 * n == size:
            is_binary = True

    if is_binary:
        triangles = np.empty((n, 3, 3), dtype=np.float64)
        with open(path, "rb") as f:
            f.seek(84)
            for i in range(n):
                rec = f.read(50)
                if len(rec) != 50:
                    raise RuntimeError("truncated binary STL: " + path)
                vals = struct.unpack("<12fH", rec)
                triangles[i, 0, :] = vals[3:6]
                triangles[i, 1, :] = vals[6:9]
                triangles[i, 2, :] = vals[9:12]
        return triangles

    verts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip().split()
            if len(s) == 4 and s[0].lower() == "vertex":
                verts.append([float(s[1]), float(s[2]), float(s[3])])
    if len(verts) % 3 != 0 or not verts:
        raise RuntimeError("failed to parse STL: " + path)
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def transform_triangles(tris, scale, transform):
    scaled = tris * np.asarray(scale, dtype=np.float64).reshape(1, 1, 3)
    r = transform[:3, :3]
    t = transform[:3, 3]
    return np.einsum("ij,tkj->tki", r, scaled) + t.reshape(1, 1, 3)


def segment_triangle_first_hit(start, end, triangles):
    """Vectorized Moller-Trumbore; return distance and hit point."""
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    seg_len = float(np.linalg.norm(delta))
    if seg_len < 1e-9 or len(triangles) == 0:
        return None
    d = delta / seg_len
    v0 = triangles[:, 0, :]
    e1 = triangles[:, 1, :] - v0
    e2 = triangles[:, 2, :] - v0
    pvec = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    good = np.abs(det) > 1e-10
    inv = np.zeros_like(det)
    inv[good] = 1.0 / det[good]

    tvec = np.asarray(start, dtype=np.float64).reshape(1, 3) - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", d, qvec) * inv
    dist = np.einsum("ij,ij->i", e2, qvec) * inv

    good &= u >= -1e-9
    good &= v >= -1e-9
    good &= (u + v) <= 1.0 + 1e-9
    good &= dist >= -1e-9
    good &= dist <= seg_len + 1e-9
    ids = np.where(good)[0]
    if len(ids) == 0:
        return None
    j = ids[np.argmin(dist[ids])]
    hit_dist = float(max(0.0, dist[j]))
    return hit_dist, np.asarray(start, dtype=np.float64) + d * hit_dist, int(j)


def load_visual_mesh_entries(urdf_path, links):
    root = ET.parse(urdf_path).getroot()
    wanted = set(links)
    out = []
    for link in root.findall("link"):
        lname = link.attrib.get("name", "")
        if lname not in wanted:
            continue
        for idx, vis in enumerate(link.findall("visual")):
            geom = vis.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            origin = vis.find("origin")
            xyz = vec(origin.attrib.get("xyz")) if origin is not None else vec(None)
            rpy = vec(origin.attrib.get("rpy")) if origin is not None else vec(None)
            scale = vec(mesh.attrib.get("scale"), (1.0, 1.0, 1.0))
            out.append({
                "link": lname,
                "index": idx,
                "uri": mesh.attrib.get("filename", ""),
                "scale": scale,
                "T_link_visual": make_transform(xyz.tolist(), rpy.tolist()),
            })
    return out


def exact_visual_mesh_raycast(
        robot, visual_entries, q_map, repo, ray_start, target):
    chain_cache = {}
    best = None
    details = []
    for entry in visual_entries:
        lname = entry["link"]
        if lname not in chain_cache:
            chain_cache[lname] = find_chain_joints(robot, "base_link", lname)
        t_base_link = fk_transform(chain_cache[lname], q_map)
        t_base_visual = t_base_link @ entry["T_link_visual"]
        path = resolve_mesh_uri(entry["uri"], repo)
        tris = load_stl_triangles(path)
        tris_base = transform_triangles(tris, entry["scale"], t_base_visual)
        hit = segment_triangle_first_hit(ray_start, target, tris_base)
        details.append({
            "link": lname,
            "uri": entry["uri"],
            "triangle_count": int(len(tris)),
            "hit": hit is not None,
            "hit_distance_m": float(hit[0]) if hit is not None else None,
        })
        if hit is None:
            continue
        if best is None or hit[0] < best["distance_m"]:
            best = {
                "distance_m": float(hit[0]),
                "point_base": hit[1],
                "triangle_index": int(hit[2]),
                "link": lname,
                "uri": entry["uri"],
                "T_base_visual": t_base_visual,
                "scale": entry["scale"],
            }
    return best, details


def marker(ns, mid, typ):
    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = rospy.Time(0)
    m.ns = ns
    m.id = mid
    m.type = typ
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def matrix_pose(m, transform):
    q = tft.quaternion_from_matrix(transform)
    m.pose.position = point(transform[:3, 3])
    m.pose.orientation.x = q[0]
    m.pose.orientation.y = q[1]
    m.pose.orientation.z = q[2]
    m.pose.orientation.w = q[3]


def primitive_marker(arr, mid, primitive, q_map, robot, strong):
    chain = find_chain_joints(robot, "base_link", primitive["link"])
    t = fk_transform(chain, q_map) @ primitive["origin"]
    m = marker("self_filter_primitive", mid, {
        "box": Marker.CUBE,
        "cylinder": Marker.CYLINDER,
        "sphere": Marker.SPHERE,
    }[primitive["type"]])
    matrix_pose(m, t)
    g = primitive["geometry"]
    if primitive["type"] == "box":
        m.scale.x, m.scale.y, m.scale.z = map(float, g.size)
    elif primitive["type"] == "cylinder":
        m.scale.x = m.scale.y = 2.0 * float(g.radius)
        m.scale.z = float(g.length)
    else:
        m.scale.x = m.scale.y = m.scale.z = 2.0 * float(g.radius)
    m.color = rgba(0.0, 0.85, 1.0, 0.55 if strong else 0.16)
    arr.markers.append(m)
    return mid + 1


class RayArgs:
    min_ray_length = 1e-4
    ray_start_offset = 0.03
    point_end_offset = 0.005
    ignore_links = []
    ignore_start_inside = True
    min_hit_distance = 0.0


def main():
    rospy.init_node("phase_e_case026_exact_self_occlusion_rviz")

    repo = rospy.get_param("~repo", "/home/zhicheng/Project/CAREPlanner")
    reference_urdf = rospy.get_param(
        "~reference_urdf",
        os.path.join(repo, "src/arm_description/urdf/Arm.urdf"))
    self_filter_urdf = rospy.get_param(
        "~self_filter_urdf",
        os.path.join(
            repo, "src/arm_description/urdf/Arm_with_self_filter_collision.urdf"))
    marker_topic = rospy.get_param(
        "~marker_topic", "/care_planner/debug/markers")
    sensor_frame = rospy.get_param("~sensor_frame", CASE026_SENSOR)

    q = np.asarray([
        float(rospy.get_param("~q1", CASE026_Q_VIS[0])),
        float(rospy.get_param("~q2", CASE026_Q_VIS[1])),
        float(rospy.get_param("~q3", CASE026_Q_VIS[2])),
        float(rospy.get_param("~q4", CASE026_Q_VIS[3])),
        float(rospy.get_param("~q5", CASE026_Q_VIS[4])),
        float(rospy.get_param("~q6", CASE026_Q_VIS[5])),
        float(rospy.get_param("~q7", CASE026_Q_VIS[6])),
    ], dtype=np.float64)
    target = np.asarray([
        float(rospy.get_param("~target_x", CASE026_TARGET[0])),
        float(rospy.get_param("~target_y", CASE026_TARGET[1])),
        float(rospy.get_param("~target_z", CASE026_TARGET[2])),
    ], dtype=np.float64)

    hfov = float(rospy.get_param("~horizontal_fov_deg", 55.0))
    vfov = float(rospy.get_param("~vertical_fov_deg", 72.0))
    z_min = float(rospy.get_param("~z_min", 0.15))
    z_max = float(rospy.get_param("~z_max", 0.75))
    visual_links = [
        x.strip() for x in rospy.get_param(
            "~exact_visual_links", ",".join(BODY_VISUAL_LINKS)).split(",")
        if x.strip()]

    reference_robot = URDF.from_xml_file(reference_urdf)
    self_filter_robot = URDF.from_xml_file(self_filter_urdf)
    q_map = q_row_to_map(DEFAULT_JOINT_NAMES, q)

    sensor_chain = find_chain_joints(reference_robot, "base_link", sensor_frame)
    t_base_sensor = fk_transform(sensor_chain, q_map)
    sensor_origin = t_base_sensor[:3, 3]

    vec_to_target = target - sensor_origin
    ray_len = float(np.linalg.norm(vec_to_target))
    if ray_len < 1e-8:
        raise RuntimeError("sensor and target coincide")
    ray_dir = vec_to_target / ray_len
    ray_start = sensor_origin + 0.03 * ray_dir
    ray_end = target - 0.005 * ray_dir

    nominal_margin, nominal_plane, p_sensor = sensor_margin(
        target, t_base_sensor, hfov, vfov, z_min, z_max)

    primitives = load_collision_primitives(self_filter_robot)
    collision_chains = [
        find_chain_joints(self_filter_robot, "base_link", p["link"])
        for p in primitives]
    primitive_occluded, primitive_hit = raycast_self_occlusion(
        self_filter_robot, collision_chains, primitives, t_base_sensor,
        target, q_map, RayArgs())

    visual_entries = load_visual_mesh_entries(reference_urdf, visual_links)
    exact_hit, mesh_details = exact_visual_mesh_raycast(
        reference_robot, visual_entries, q_map, repo, ray_start, ray_end)

    exact_occluded = exact_hit is not None
    exact_distance_from_sensor = (
        0.03 + exact_hit["distance_m"] if exact_hit is not None else None)
    if primitive_occluded and exact_occluded:
        classification = "EXACT_MESH_SELF_OCCLUSION_CONFIRMED"
    elif primitive_occluded and not exact_occluded:
        classification = "PRIMITIVE_ONLY__SELF_FILTER_TOO_CONSERVATIVE"
    elif not primitive_occluded and exact_occluded:
        classification = "EXACT_MESH_HIT_MISSED_BY_SELF_FILTER"
    else:
        classification = "NO_GEOMETRIC_SELF_OCCLUSION"

    rospy.logwarn("============================================================")
    rospy.logwarn("[CASE026 EXACT SELF-OCCLUSION]")
    rospy.logwarn("q_vis=%s", np.array2string(q, precision=5))
    rospy.logwarn("sensor=%s origin=%s", sensor_frame, np.array2string(sensor_origin, precision=5))
    rospy.logwarn("target=%s", np.array2string(target, precision=5))
    rospy.logwarn(
        "target_sensor=%s nominal_margin=%+.6f plane=%d",
        np.array2string(p_sensor, precision=5), nominal_margin, nominal_plane)
    rospy.logwarn(
        "PRIMITIVE_OCCLUDED=%d hit=%s",
        int(primitive_occluded), str(primitive_hit))
    if exact_hit is None:
        rospy.logwarn("EXACT_MESH_OCCLUDED=0")
    else:
        rospy.logwarn(
            "EXACT_MESH_OCCLUDED=1 link=%s dist_from_sensor=%.6f m "
            "dist_from_30mm_start=%.6f m point=%s triangle=%d",
            exact_hit["link"], exact_distance_from_sensor,
            exact_hit["distance_m"],
            np.array2string(exact_hit["point_base"], precision=6),
            exact_hit["triangle_index"])
    rospy.logwarn("CLASSIFICATION=%s", classification)

    output_json = rospy.get_param(
        "~output_json",
        os.path.join(
            repo, "outputs/phase_e_case026_exact_self_occlusion",
            "case026_exact_self_occlusion.json"))
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    report = {
        "case_id": "phase_e_goal_026",
        "obligation_id": 7,
        "q_vis": [float(v) for v in q],
        "sensor_frame": sensor_frame,
        "sensor_origin_base": [float(v) for v in sensor_origin],
        "target_base": [float(v) for v in target],
        "target_sensor": [float(v) for v in p_sensor],
        "nominal_fov_margin_m": float(nominal_margin),
        "primitive_occluded": bool(primitive_occluded),
        "primitive_hit": primitive_hit,
        "exact_mesh_occluded": bool(exact_occluded),
        "exact_mesh_hit": (
            None if exact_hit is None else {
                "link": exact_hit["link"],
                "mesh_uri": exact_hit["uri"],
                "distance_from_sensor_m": float(exact_distance_from_sensor),
                "distance_from_30mm_start_m": float(exact_hit["distance_m"]),
                "point_base": [float(v) for v in exact_hit["point_base"]],
                "triangle_index": int(exact_hit["triangle_index"]),
            }),
        "mesh_tests": mesh_details,
        "classification": classification,
    }
    with open(output_json, "w") as out:
        json.dump(report, out, indent=2)
    rospy.logwarn("OUTPUT_JSON=%s", output_json)
    rospy.logwarn("============================================================")

    pub_markers = rospy.Publisher(marker_topic, MarkerArray, queue_size=1, latch=True)
    pub_joint = rospy.Publisher("/joint_states", JointState, queue_size=1, latch=True)
    rospy.sleep(0.5)

    arr = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    mid = 1

    # Sensor origin.
    m = marker("case026_sensor", mid, Marker.SPHERE); mid += 1
    m.pose.position = point(sensor_origin)
    m.scale.x = m.scale.y = m.scale.z = 0.025
    m.color = rgba(1.0, 0.85, 0.0, 1.0)
    arr.markers.append(m)

    # Target point.
    m = marker("case026_target", mid, Marker.SPHERE); mid += 1
    m.pose.position = point(target)
    m.scale.x = m.scale.y = m.scale.z = 0.032
    m.color = rgba(1.0, 0.0, 1.0, 1.0)
    arr.markers.append(m)

    # Full sensor -> target ray.
    m = marker("case026_ray", mid, Marker.LINE_LIST); mid += 1
    m.scale.x = 0.006
    m.color = rgba(0.1, 1.0, 0.2, 1.0)
    m.points = [point(sensor_origin), point(target)]
    arr.markers.append(m)

    # Nominal FOV frustum.
    fr = marker("case026_fov", mid, Marker.LINE_LIST); mid += 1
    fr.scale.x = 0.003
    fr.color = rgba(1.0, 0.85, 0.0, 0.9)
    hx = math.tan(math.radians(hfov) * 0.5)
    hy = math.tan(math.radians(vfov) * 0.5)

    def tfp(v):
        h = np.ones(4)
        h[:3] = v
        return (t_base_sensor @ h)[:3]

    near = [
        tfp([-z_min * hx, -z_min * hy, z_min]),
        tfp([ z_min * hx, -z_min * hy, z_min]),
        tfp([ z_min * hx,  z_min * hy, z_min]),
        tfp([-z_min * hx,  z_min * hy, z_min]),
    ]
    far = [
        tfp([-z_max * hx, -z_max * hy, z_max]),
        tfp([ z_max * hx, -z_max * hy, z_max]),
        tfp([ z_max * hx,  z_max * hy, z_max]),
        tfp([-z_max * hx,  z_max * hy, z_max]),
    ]
    for i in range(4):
        fr.points.extend([point(near[i]), point(near[(i + 1) % 4])])
        fr.points.extend([point(far[i]), point(far[(i + 1) % 4])])
        fr.points.extend([point(near[i]), point(far[i])])
    arr.markers.append(fr)

    # Highlight all self-filter primitives on the primitive-hit link, with the
    # actually hit primitive stronger.
    if primitive_hit is not None:
        hit_link = primitive_hit["link"]
        hit_name = primitive_hit["collision"]
        for p in primitives:
            if p["link"] != hit_link:
                continue
            mid = primitive_marker(
                arr, mid, p, q_map, self_filter_robot,
                strong=(p["name"] == hit_name))

        hit_dist = float(primitive_hit["distance"])
        primitive_hit_point = sensor_origin + ray_dir * hit_dist
        m = marker("case026_primitive_hit", mid, Marker.SPHERE); mid += 1
        m.pose.position = point(primitive_hit_point)
        m.scale.x = m.scale.y = m.scale.z = 0.022
        m.color = rgba(0.0, 0.85, 1.0, 1.0)
        arr.markers.append(m)

    # Highlight exact STL mesh that actually intersects the ray.
    if exact_hit is not None:
        m = marker("case026_exact_mesh", mid, Marker.MESH_RESOURCE); mid += 1
        m.mesh_resource = exact_hit["uri"]
        m.mesh_use_embedded_materials = False
        matrix_pose(m, exact_hit["T_base_visual"])
        m.scale.x, m.scale.y, m.scale.z = map(float, exact_hit["scale"])
        m.color = rgba(1.0, 0.25, 0.0, 0.55)
        arr.markers.append(m)

        m = marker("case026_exact_hit", mid, Marker.SPHERE); mid += 1
        m.pose.position = point(exact_hit["point_base"])
        m.scale.x = m.scale.y = m.scale.z = 0.026
        m.color = rgba(1.0, 0.05, 0.0, 1.0)
        arr.markers.append(m)

    # Text summary next to target.
    txt = marker("case026_summary", mid, Marker.TEXT_VIEW_FACING); mid += 1
    txt.pose.position = point(target + np.array([0.0, 0.0, 0.07]))
    txt.scale.z = 0.035
    txt.color = rgba(1.0, 1.0, 1.0, 1.0)
    exact_desc = (
        "none" if exact_hit is None
        else "%s @ %.3fm" % (exact_hit["link"], exact_distance_from_sensor))
    prim_desc = (
        "none" if primitive_hit is None
        else "%s/%s @ %.3fm" % (
            primitive_hit["link"], primitive_hit["collision"],
            primitive_hit["distance"]))
    txt.text = (
        "Case026 S4 -> target\n"
        "FOV margin = %+.3f m\n"
        "primitive = %s\n"
        "exact STL = %s\n"
        "%s" % (nominal_margin, prim_desc, exact_desc, classification))
    arr.markers.append(txt)

    pub_markers.publish(arr)

    js = JointState()
    js.name = list(DEFAULT_JOINT_NAMES)
    js.position = [float(v) for v in q]

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        js.header.stamp = rospy.Time.now()
        pub_joint.publish(js)
        pub_markers.publish(arr)
        rate.sleep()


if __name__ == "__main__":
    main()

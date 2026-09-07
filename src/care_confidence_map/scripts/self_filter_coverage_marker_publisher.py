#!/usr/bin/env python3
import csv
import math
import os
import xml.etree.ElementTree as ET

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def parse_vec(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return tuple(default)
    vals = [float(x) for x in text.split()]
    if len(vals) != 3:
        raise ValueError(text)
    return tuple(vals)


def rpy_to_quat(rpy):
    import tf.transformations as tft
    return tft.quaternion_from_euler(*rpy)


def load_primitives(urdf_path):
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        lname = link.attrib.get("name", "")
        plist = []
        for i, col in enumerate(link.findall("collision")):
            geom = col.find("geometry")
            if geom is None:
                continue
            origin = col.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz")) if origin is not None else (0,0,0)
            rpy = parse_vec(origin.attrib.get("rpy")) if origin is not None else (0,0,0)
            name = col.attrib.get("name", f"collision_{i}")
            box = geom.find("box")
            cyl = geom.find("cylinder")
            sph = geom.find("sphere")
            if box is not None:
                plist.append(("box", name, xyz, rpy, parse_vec(box.attrib["size"])))
            elif cyl is not None:
                plist.append(("cylinder", name, xyz, rpy,
                              (2.0*float(cyl.attrib["radius"]),
                               2.0*float(cyl.attrib["radius"]),
                               float(cyl.attrib["length"]))))
            elif sph is not None:
                d = 2.0*float(sph.attrib["radius"])
                plist.append(("sphere", name, xyz, rpy, (d,d,d)))
        if plist:
            out[lname] = plist
    return out


def load_visual_meshes(urdf_path):
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        lname = link.attrib.get("name", "")
        visuals = []
        for i, vis in enumerate(link.findall("visual")):
            geom = vis.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue
            origin = vis.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz")) if origin is not None else (0,0,0)
            rpy = parse_vec(origin.attrib.get("rpy")) if origin is not None else (0,0,0)
            scale = parse_vec(mesh.attrib.get("scale"), (1.0,1.0,1.0))
            visuals.append((
                vis.attrib.get("name", f"visual_{i}"),
                mesh.attrib.get("filename", ""),
                xyz, rpy, scale))
        if visuals:
            out[lname] = visuals
    return out


def load_worst_points(csv_path, selected_links, max_per_link, min_outside_mm):
    rows = []
    with open(csv_path, newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            source = row.get("source_link", "")
            if selected_links and source not in selected_links:
                continue
            try:
                outside_m = float(row["outside_m"])
                x = float(row["x_anchor"])
                y = float(row["y_anchor"])
                z = float(row["z_anchor"])
            except Exception:
                continue
            if outside_m * 1000.0 < min_outside_mm:
                continue
            rows.append({
                "source_link": source,
                "anchor_link": row.get("anchor_link", source),
                "x": x, "y": y, "z": z,
                "outside_m": outside_m,
                "nearest_primitive": row.get("nearest_primitive", ""),
            })

    grouped = {}
    for r in rows:
        grouped.setdefault(r["source_link"], []).append(r)
    for key in grouped:
        grouped[key].sort(key=lambda r: r["outside_m"], reverse=True)
        grouped[key] = grouped[key][:max_per_link]
    return grouped


def make_color(r, g, b, a):
    from std_msgs.msg import ColorRGBA
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


def main():
    rospy.init_node("self_filter_coverage_marker_publisher")

    csv_path = rospy.get_param(
        "~coverage_csv",
        "/home/zhicheng/Project/CAREPlanner/outputs/self_filter_mesh_coverage/"
        "self_filter_mesh_coverage_worst_points.csv")
    urdf_path = rospy.get_param(
        "~self_filter_urdf",
        "/home/zhicheng/Project/CAREPlanner/src/arm_description/urdf/"
        "Arm_with_self_filter_collision.urdf")
    reference_urdf = rospy.get_param(
        "~reference_urdf",
        "/home/zhicheng/Project/CAREPlanner/src/arm_description/urdf/Arm.urdf")
    selected = rospy.get_param("~links", "link2,link1,link3,link4,wrist_link1,wrist_link3")
    selected_links = {x.strip() for x in selected.split(",") if x.strip()}
    max_per_link = int(rospy.get_param("~max_points_per_link", 30))
    min_outside_mm = float(rospy.get_param("~min_outside_mm", 0.1))
    show_geometry = bool(rospy.get_param("~show_filter_geometry", True))
    point_size = float(rospy.get_param("~point_size_m", 0.010))
    worst_size = float(rospy.get_param("~worst_point_size_m", 0.025))
    topic = rospy.get_param("~marker_topic", "/care_planner/debug/markers")

    if not os.path.isfile(csv_path):
        raise RuntimeError("coverage CSV missing: " + csv_path)
    if not os.path.isfile(urdf_path):
        raise RuntimeError("self-filter URDF missing: " + urdf_path)
    if not os.path.isfile(reference_urdf):
        raise RuntimeError("reference URDF missing: " + reference_urdf)

    grouped = load_worst_points(
        csv_path, selected_links, max_per_link, min_outside_mm)
    primitives = load_primitives(urdf_path)
    visual_meshes = load_visual_meshes(reference_urdf)

    pub = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
    rospy.sleep(0.5)

    arr = MarkerArray()
    marker_id = 0

    # Clear stale namespaces from earlier link selections.  Without DELETEALL,
    # RViz keeps old wrist/link markers when this publisher is relaunched with
    # a different links:= selection.
    clear = Marker()
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)

    # Explicitly overlay the selected ACTUAL visual mesh(es) in yellow so the
    # user never has to infer which part of the full RobotModel is being
    # audited.
    for source in sorted(selected_links):
        for _, mesh_uri, xyz, rpy, scale in visual_meshes.get(source, []):
            if not mesh_uri:
                continue
            m = Marker()
            m.header.frame_id = source
            m.header.stamp = rospy.Time(0)
            m.frame_locked = True
            m.ns = "selected_visual_mesh"
            m.id = marker_id
            marker_id += 1
            m.action = Marker.ADD
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = mesh_uri
            m.mesh_use_embedded_materials = False
            m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
            q = rpy_to_quat(rpy)
            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color = make_color(1.0, 0.75, 0.0, 0.55)
            arr.markers.append(m)

    # Dedicated self-filter primitives: translucent cyan.
    if show_geometry:
        anchors = sorted({
            r["anchor_link"]
            for rows in grouped.values()
            for r in rows
        })
        for anchor in anchors:
            for kind, name, xyz, rpy, scale in primitives.get(anchor, []):
                m = Marker()
                m.header.frame_id = anchor
                m.header.stamp = rospy.Time(0)
                m.frame_locked = True
                m.ns = "self_filter_geometry"
                m.id = marker_id
                marker_id += 1
                m.action = Marker.ADD
                m.type = {
                    "box": Marker.CUBE,
                    "cylinder": Marker.CYLINDER,
                    "sphere": Marker.SPHERE,
                }[kind]
                m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
                q = rpy_to_quat(rpy)
                m.pose.orientation.x = q[0]
                m.pose.orientation.y = q[1]
                m.pose.orientation.z = q[2]
                m.pose.orientation.w = q[3]
                m.scale.x, m.scale.y, m.scale.z = scale
                m.color = make_color(0.0, 0.8, 1.0, 0.22)
                arr.markers.append(m)

    # Outside visual-surface samples: red points, per source visual link.
    for source in sorted(grouped):
        rows = grouped[source]
        if not rows:
            continue
        anchor = rows[0]["anchor_link"]

        pts = Marker()
        pts.header.frame_id = anchor
        pts.header.stamp = rospy.Time(0)
        pts.frame_locked = True
        pts.ns = "outside_points_" + source
        pts.id = marker_id
        marker_id += 1
        pts.type = Marker.POINTS
        pts.action = Marker.ADD
        pts.pose.orientation.w = 1.0
        pts.scale.x = point_size
        pts.scale.y = point_size
        pts.color = make_color(1.0, 0.05, 0.05, 1.0)
        for r in rows:
            p = Point()
            p.x, p.y, p.z = r["x"], r["y"], r["z"]
            pts.points.append(p)
        arr.markers.append(pts)

        # Largest violation for this visual link: larger magenta sphere + text.
        worst = rows[0]
        w = Marker()
        w.header.frame_id = anchor
        w.header.stamp = rospy.Time(0)
        w.frame_locked = True
        w.ns = "worst_point_" + source
        w.id = marker_id
        marker_id += 1
        w.type = Marker.SPHERE
        w.action = Marker.ADD
        w.pose.position.x = worst["x"]
        w.pose.position.y = worst["y"]
        w.pose.position.z = worst["z"]
        w.pose.orientation.w = 1.0
        w.scale.x = w.scale.y = w.scale.z = worst_size
        w.color = make_color(1.0, 0.0, 1.0, 1.0)
        arr.markers.append(w)

        txt = Marker()
        txt.header.frame_id = anchor
        txt.header.stamp = rospy.Time(0)
        txt.frame_locked = True
        txt.ns = "outside_text_" + source
        txt.id = marker_id
        marker_id += 1
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = worst["x"]
        txt.pose.position.y = worst["y"]
        txt.pose.position.z = worst["z"] + 0.035
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.028
        txt.color = make_color(1.0, 1.0, 1.0, 1.0)
        txt.text = (
            f"{source}: {1000.0*worst['outside_m']:.2f} mm\n"
            f"nearest={worst['nearest_primitive']}")
        arr.markers.append(txt)

    pub.publish(arr)

    rospy.loginfo(
        "[self_filter_coverage_marker] published %d markers to %s; "
        "links=%s",
        len(arr.markers), topic, ",".join(sorted(grouped.keys())))
    rospy.loginfo(
        "[self_filter_coverage_marker] yellow=selected actual visual mesh, "
        "cyan=dedicated filter primitives, red=outside visual samples, "
        "magenta=worst sample")

    rospy.spin()


if __name__ == "__main__":
    main()

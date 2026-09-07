#!/usr/bin/env python3
"""RViz diagnostic for Phase-E runtime self-hit leakage.

This node is diagnostic-only. It does not change point clouds, self-filter
decisions, confidence-map state, or planning.

For one selected raw ToF sensor it reproduces the exact endpoint-containment
test used by tof_fusion_self_filter_node on the same cloud timestamp, then
visualizes:
  cyan   : exact dedicated-URDF self-filter primitives
  yellow : selected ToF origin + FOV frustum
  green  : hotspot finite endpoints classified as self
  red    : hotspot finite endpoints classified as non-self (map HIT candidates)
  magenta: rays from the ToF origin to those suspect endpoints
  blue   : first exact self-primitive intersection along a suspect ray
  orange : 5-cm confidence-map voxels receiving suspect endpoints

The node also computes an exact segment/primitive intersection after the ToF
near clip. This is the key discriminator:
  ray_crosses_self=1 -> the measured endpoint lies beyond a robot surface
                        along the same ray (depth-edge / endpoint overshoot)
  ray_crosses_self=0 -> investigate sensor/TF/render pose mismatch instead.

When execution GCDF HARD_HOLD becomes true, the most recent diagnostic frame
is frozen and repeatedly published so RViz shows the configuration that
created the blocker.
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tft
import tf2_ros

from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


EPS = 1e-10


def vec(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in text.split()]
    if len(vals) != 3:
        raise ValueError(text)
    return np.asarray(vals, dtype=np.float64)


def T_xyz_rpy(xyz, rpy):
    T = tft.euler_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = xyz
    return T


def T_from_tf(msg):
    q = msg.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    p = msg.transform.translation
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def signed_distance_local(points, prim):
    if prim["kind"] == "box":
        d = np.abs(points) - 0.5 * prim["size"]
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(np.max(d, axis=1), 0.0)
        return outside + inside
    if prim["kind"] == "cylinder":
        radial = np.linalg.norm(points[:, :2], axis=1) - prim["radius"]
        axial = np.abs(points[:, 2]) - 0.5 * prim["length"]
        outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return outside + inside
    if prim["kind"] == "sphere":
        return np.linalg.norm(points, axis=1) - prim["radius"]
    raise RuntimeError(prim["kind"])


def ray_box(o, d, size, max_t):
    lo = -0.5 * size
    hi = 0.5 * size
    t0, t1 = 0.0, max_t
    for k in range(3):
        if abs(d[k]) < EPS:
            if o[k] < lo[k] or o[k] > hi[k]:
                return None
            continue
        a = (lo[k] - o[k]) / d[k]
        b = (hi[k] - o[k]) / d[k]
        if a > b:
            a, b = b, a
        t0 = max(t0, a)
        t1 = min(t1, b)
        if t0 > t1:
            return None
    return t0 if 0.0 <= t0 <= max_t else None


def ray_sphere(o, d, radius, max_t):
    c = float(np.dot(o, o) - radius * radius)
    if c <= 0.0:
        return 0.0
    b = 2.0 * float(np.dot(o, d))
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    s = math.sqrt(max(0.0, disc))
    vals = [(-b - s) * 0.5, (-b + s) * 0.5]
    vals = [t for t in vals if 0.0 <= t <= max_t]
    return min(vals) if vals else None


def ray_cylinder(o, d, radius, length, max_t):
    h = 0.5 * length
    if o[0] * o[0] + o[1] * o[1] <= radius * radius and abs(o[2]) <= h:
        return 0.0
    hits = []
    a = d[0] * d[0] + d[1] * d[1]
    if a > EPS:
        b = 2.0 * (o[0] * d[0] + o[1] * d[1])
        c = o[0] * o[0] + o[1] * o[1] - radius * radius
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            s = math.sqrt(max(0.0, disc))
            for t in ((-b - s) / (2.0 * a), (-b + s) / (2.0 * a)):
                if 0.0 <= t <= max_t:
                    z = o[2] + t * d[2]
                    if -h <= z <= h:
                        hits.append(t)
    if abs(d[2]) > EPS:
        for zcap in (-h, h):
            t = (zcap - o[2]) / d[2]
            if 0.0 <= t <= max_t:
                x = o[0] + t * d[0]
                y = o[1] + t * d[1]
                if x * x + y * y <= radius * radius:
                    hits.append(t)
    return min(hits) if hits else None


def rgba(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


def point(v):
    p = Point()
    p.x, p.y, p.z = map(float, v)
    return p


class RuntimeSelfHitDiag:
    def __init__(self):
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.raw_topic = rospy.get_param("~raw_topic", "/link4_sensor2/tof/cloud")
        self.sensor_frame = rospy.get_param(
            "~sensor_frame", "link4_sensor2_tof_link")
        self.sensor_id = int(rospy.get_param("~sensor_id", 5))
        self.sensor_name = rospy.get_param("~sensor_name", "link4_sensor2")
        self.urdf_path = rospy.get_param("~self_filter_urdf")
        self.marker_topic = rospy.get_param(
            "~marker_topic", "/care_planner/debug/markers")
        self.hard_hold_topic = rospy.get_param(
            "~hard_hold_topic", "/care_planner/execution_gcdf/hard_hold")
        self.joint_topic = rospy.get_param(
            "~joint_topic", "/care_arm/joint_states")

        self.stride = int(rospy.get_param("~pixel_stride", 8))
        self.near_clip = float(rospy.get_param("~near_clip", 0.15))
        self.far_clip = float(rospy.get_param("~far_clip", 0.75))
        self.hfov = math.radians(float(rospy.get_param("~horizontal_fov_deg", 55.0)))
        self.vfov = math.radians(float(rospy.get_param("~vertical_fov_deg", 72.0)))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.03))

        self.hotspot = np.asarray([
            float(rospy.get_param("~hotspot_x_min", -0.025)),
            float(rospy.get_param("~hotspot_x_max", 0.075)),
            float(rospy.get_param("~hotspot_y_min", -0.025)),
            float(rospy.get_param("~hotspot_y_max", 0.125)),
            float(rospy.get_param("~hotspot_z_min", 0.200)),
            float(rospy.get_param("~hotspot_z_max", 0.425)),
        ])

        self.map_min = np.asarray([
            float(rospy.get_param("~map_x_min", -0.95)),
            float(rospy.get_param("~map_y_min", -0.95)),
            float(rospy.get_param("~map_z_min", 0.0)),
        ])
        self.map_res = float(rospy.get_param("~map_resolution", 0.05))

        self.primitives = self.load_primitives(self.urdf_path)
        self.link_names = sorted({p["link"] for p in self.primitives})

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)
        self.pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        self.q_names = [
            "joint1", "joint2", "joint3", "joint4",
            "wrist_joint1", "wrist_joint2", "wrist_joint3"]
        self.latest_q = None
        self.latest_markers = None
        self.latest_summary = ""
        self.frozen = False

        rospy.Subscriber(self.joint_topic, JointState, self.on_joint, queue_size=1)
        rospy.Subscriber(self.hard_hold_topic, Bool, self.on_hold, queue_size=1)
        rospy.Subscriber(self.raw_topic, rospy.AnyMsg, self.on_any_cloud, queue_size=1)
        # Replace AnyMsg subscriber after type discovery; this avoids startup
        # ordering assumptions while still using a typed PointCloud2 callback.
        self.any_sub = None
        self.cloud_sub = None
        self.timer = rospy.Timer(rospy.Duration(0.2), self.on_timer)

        rospy.logwarn(
            "[RUNTIME_SELF_HIT_DIAG] ready sensor_id=%d sensor=%s raw=%s "
            "urdf=%s; no mapping/planning semantics changed",
            self.sensor_id, self.sensor_name, self.raw_topic, self.urdf_path)

    def on_any_cloud(self, _msg):
        if self.cloud_sub is not None:
            return
        try:
            if self.any_sub is not None:
                self.any_sub.unregister()
        except Exception:
            pass
        self.cloud_sub = rospy.Subscriber(
            self.raw_topic,
            __import__("sensor_msgs.msg", fromlist=["PointCloud2"]).PointCloud2,
            self.on_cloud,
            queue_size=1)

    def on_joint(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in self.q_names):
            self.latest_q = [float(msg.position[idx[n]]) for n in self.q_names]

    def on_hold(self, msg):
        if msg.data and not self.frozen:
            self.frozen = True
            rospy.logerr(
                "[RUNTIME_SELF_HIT_DIAG_FREEZE] HARD_HOLD; frozen snapshot: %s",
                self.latest_summary or "no suspect frame captured yet")

    def on_timer(self, _evt):
        if self.latest_markers is not None:
            self.pub.publish(self.latest_markers)

    def load_primitives(self, path):
        if not os.path.isfile(path):
            raise RuntimeError("missing self-filter URDF: " + path)
        root = ET.parse(path).getroot()
        out = []
        for link in root.findall("link"):
            lname = link.attrib.get("name", "")
            for ci, col in enumerate(link.findall("collision")):
                geom = col.find("geometry")
                if geom is None:
                    continue
                org = col.find("origin")
                xyz = vec(org.attrib.get("xyz")) if org is not None else vec(None)
                rpy = vec(org.attrib.get("rpy")) if org is not None else vec(None)
                p = {
                    "link": lname,
                    "name": col.attrib.get("name", "collision_%d" % ci),
                    "T_link_prim": T_xyz_rpy(xyz, rpy),
                }
                box = geom.find("box")
                cyl = geom.find("cylinder")
                sph = geom.find("sphere")
                if box is not None:
                    p["kind"] = "box"
                    p["size"] = vec(box.attrib["size"])
                elif cyl is not None:
                    p["kind"] = "cylinder"
                    p["radius"] = float(cyl.attrib["radius"])
                    p["length"] = float(cyl.attrib["length"])
                elif sph is not None:
                    p["kind"] = "sphere"
                    p["radius"] = float(sph.attrib["radius"])
                else:
                    continue
                out.append(p)
        if not out:
            raise RuntimeError("no supported primitives in " + path)
        return out

    def timed_primitives(self, stamp):
        link_T = {}
        for lname in self.link_names:
            try:
                tfm = self.tfbuf.lookup_transform(
                    self.base_frame, lname, stamp, rospy.Duration(self.tf_timeout))
            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0, "[RUNTIME_SELF_HIT_DIAG] TF unavailable %s: %s" %
                    (lname, exc))
                return None
            link_T[lname] = T_from_tf(tfm)

        timed = []
        for p in self.primitives:
            q = dict(p)
            q["T_base_prim"] = link_T[p["link"]].dot(p["T_link_prim"])
            q["T_prim_base"] = np.linalg.inv(q["T_base_prim"])
            timed.append(q)
        return timed

    def classify(self, P, timed):
        n = len(P)
        best = np.full(n, np.inf, dtype=np.float64)
        best_i = np.full(n, -1, dtype=np.int32)
        Ph = np.ones((n, 4), dtype=np.float64)
        Ph[:, :3] = P
        for i, p in enumerate(timed):
            L = Ph.dot(p["T_prim_base"].T)[:, :3]
            d = signed_distance_local(L, p)
            m = d < best
            best[m] = d[m]
            best_i[m] = i
        return best, best_i

    def first_ray_self_intersection(self, origin, endpoint, timed):
        delta = endpoint - origin
        measured = float(np.linalg.norm(delta))
        if measured <= self.near_clip + 1e-6:
            return None
        dbase = delta / measured
        start = self.near_clip
        ostart = origin + dbase * start
        seg_len = measured - start

        best = None
        for i, p in enumerate(timed):
            T = p["T_prim_base"]
            R = T[:3, :3]
            o = R.dot(ostart) + T[:3, 3]
            dl = R.dot(dbase)
            if p["kind"] == "box":
                t = ray_box(o, dl, p["size"], seg_len)
            elif p["kind"] == "cylinder":
                t = ray_cylinder(o, dl, p["radius"], p["length"], seg_len)
            else:
                t = ray_sphere(o, dl, p["radius"], seg_len)
            if t is not None:
                tabs = start + float(t)
                if best is None or tabs < best[0]:
                    best = (tabs, i, origin + dbase * tabs)
        return best

    def voxel_center(self, p):
        idx = np.rint((p - self.map_min) / self.map_res)
        return self.map_min + idx * self.map_res

    def add_marker(self, arr, ns, mid, typ, stamp, lifetime=0.35):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = typ
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = rospy.Duration(lifetime)
        arr.markers.append(m)
        return m

    def make_markers(self, stamp, Tbs, timed, self_pts, suspects, ray_hits,
                     nearest_labels, nearest_d, pixels):
        arr = MarkerArray()
        mid = 0

        # Exact self-filter primitive union at the cloud stamp.
        for p in timed:
            m = self.add_marker(arr, "runtime_self_primitives", mid,
                                {"box": Marker.CUBE,
                                 "cylinder": Marker.CYLINDER,
                                 "sphere": Marker.SPHERE}[p["kind"]],
                                stamp)
            mid += 1
            T = p["T_base_prim"]
            q = tft.quaternion_from_matrix(T)
            m.pose.position = point(T[:3, 3])
            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]
            if p["kind"] == "box":
                m.scale.x, m.scale.y, m.scale.z = map(float, p["size"])
            elif p["kind"] == "cylinder":
                m.scale.x = m.scale.y = 2.0 * p["radius"]
                m.scale.z = p["length"]
            else:
                m.scale.x = m.scale.y = m.scale.z = 2.0 * p["radius"]
            m.color = rgba(0.0, 0.8, 1.0, 0.18)

        origin = Tbs[:3, 3]

        # Sensor origin.
        m = self.add_marker(arr, "runtime_sensor", mid, Marker.SPHERE, stamp)
        mid += 1
        m.pose.position = point(origin)
        m.scale.x = m.scale.y = m.scale.z = 0.025
        m.color = rgba(1.0, 0.85, 0.0, 1.0)

        # FOV frustum (+z forward in tof_link / optical frame).
        fr = self.add_marker(arr, "runtime_sensor_fov", mid, Marker.LINE_LIST, stamp)
        mid += 1
        fr.scale.x = 0.004
        fr.color = rgba(1.0, 0.75, 0.0, 0.9)
        hx = math.tan(0.5 * self.hfov)
        hy = math.tan(0.5 * self.vfov)

        def tfp(v):
            h = np.ones(4)
            h[:3] = v
            return Tbs.dot(h)[:3]

        far = [
            tfp([-self.far_clip * hx, -self.far_clip * hy, self.far_clip]),
            tfp([ self.far_clip * hx, -self.far_clip * hy, self.far_clip]),
            tfp([ self.far_clip * hx,  self.far_clip * hy, self.far_clip]),
            tfp([-self.far_clip * hx,  self.far_clip * hy, self.far_clip]),
        ]
        for c in far:
            fr.points.extend([point(origin), point(c)])
        for i in range(4):
            fr.points.extend([point(far[i]), point(far[(i + 1) % 4])])

        if len(self_pts):
            m = self.add_marker(arr, "runtime_self_hits", mid, Marker.POINTS, stamp)
            mid += 1
            m.scale.x = m.scale.y = 0.009
            m.color = rgba(0.15, 1.0, 0.15, 0.9)
            m.points = [point(x) for x in self_pts]

        if len(suspects):
            m = self.add_marker(arr, "runtime_suspect_hits", mid, Marker.POINTS, stamp)
            mid += 1
            m.scale.x = m.scale.y = 0.014
            m.color = rgba(1.0, 0.05, 0.05, 1.0)
            m.points = [point(x) for x in suspects]

            rays = self.add_marker(arr, "runtime_suspect_rays", mid,
                                   Marker.LINE_LIST, stamp)
            mid += 1
            rays.scale.x = 0.003
            rays.color = rgba(1.0, 0.0, 1.0, 0.85)
            for p in suspects:
                rays.points.extend([point(origin), point(p)])

            ints = [r[2] for r in ray_hits if r is not None]
            if ints:
                m = self.add_marker(arr, "runtime_first_self_intersection", mid,
                                    Marker.POINTS, stamp)
                mid += 1
                m.scale.x = m.scale.y = 0.012
                m.color = rgba(0.1, 0.45, 1.0, 1.0)
                m.points = [point(x) for x in ints]

            voxels = {}
            for p in suspects:
                c = self.voxel_center(p)
                voxels[tuple(np.round(c, 6))] = c
            for c in voxels.values():
                m = self.add_marker(arr, "runtime_suspect_voxels", mid,
                                    Marker.CUBE, stamp)
                mid += 1
                m.pose.position = point(c)
                m.scale.x = m.scale.y = m.scale.z = self.map_res
                m.color = rgba(1.0, 0.35, 0.0, 0.25)

        # Diagnostic text. Place it above the selected sensor.
        qtxt = "q=unavailable"
        if self.latest_q is not None:
            qtxt = "q=[" + ",".join("%.3f" % x for x in self.latest_q) + "]"
        cross = sum(1 for r in ray_hits if r is not None)
        txt = self.add_marker(arr, "runtime_self_hit_text", mid,
                              Marker.TEXT_VIEW_FACING, stamp)
        txt.pose.position = point(origin + np.asarray([0.0, 0.0, 0.08]))
        txt.scale.z = 0.025
        txt.color = rgba(1.0, 1.0, 1.0, 1.0)
        txt.text = (
            "sensor %d: %s\n"
            "stamp=%.3f  hotspot self=%d suspect=%d\n"
            "ray_crosses_self=%d/%d\n%s" %
            (self.sensor_id, self.sensor_name, stamp.to_sec(),
             len(self_pts), len(suspects), cross, len(suspects), qtxt))

        if len(suspects):
            k = int(np.argmax(nearest_d))
            label = nearest_labels[k]
            uv = pixels[k]
            extra = self.add_marker(arr, "runtime_worst_text", mid,
                                    Marker.TEXT_VIEW_FACING, stamp)
            extra.pose.position = point(suspects[k] + np.asarray([0, 0, 0.035]))
            extra.scale.z = 0.021
            extra.color = rgba(1.0, 0.8, 0.8, 1.0)
            extra.text = (
                "red endpoint pixel=(%d,%d)\nnearest=%s  outside=%.1f mm" %
                (uv[0], uv[1], label, 1000.0 * nearest_d[k]))

        return arr

    def on_cloud(self, msg):
        if self.frozen:
            return
        if msg.width <= 0 or msg.height <= 1:
            return

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        try:
            tfm = self.tfbuf.lookup_transform(
                self.base_frame, self.sensor_frame, stamp,
                rospy.Duration(self.tf_timeout))
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0, "[RUNTIME_SELF_HIT_DIAG] sensor TF unavailable: %s" % exc)
            return
        Tbs = T_from_tf(tfm)
        timed = self.timed_primitives(stamp)
        if timed is None:
            return

        uvs = [(u, v)
               for v in range(0, int(msg.height), self.stride)
               for u in range(0, int(msg.width), self.stride)]
        vals = list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=False, uvs=uvs))
        if not vals:
            return

        raw = np.asarray(vals, dtype=np.float64)
        finite = np.isfinite(raw).all(axis=1)
        if not np.any(finite):
            return
        raw = raw[finite]
        uvf = [uv for uv, keep in zip(uvs, finite.tolist()) if keep]

        H = np.ones((len(raw), 4), dtype=np.float64)
        H[:, :3] = raw
        P = H.dot(Tbs.T)[:, :3]

        hs = (
            (P[:, 0] >= self.hotspot[0]) & (P[:, 0] <= self.hotspot[1]) &
            (P[:, 1] >= self.hotspot[2]) & (P[:, 1] <= self.hotspot[3]) &
            (P[:, 2] >= self.hotspot[4]) & (P[:, 2] <= self.hotspot[5]))
        if not np.any(hs):
            return

        P = P[hs]
        uvh = [uv for uv, keep in zip(uvf, hs.tolist()) if keep]
        best, best_i = self.classify(P, timed)
        self_mask = best <= 0.0
        suspect_mask = ~self_mask
        self_pts = P[self_mask]
        suspects = P[suspect_mask]

        nearest_labels = []
        nearest_d = []
        pixels = []
        ray_hits = []
        origin = Tbs[:3, 3]
        suspect_indices = np.flatnonzero(suspect_mask)
        for j in suspect_indices:
            pi = int(best_i[j])
            nearest_labels.append(
                "%s/%s" % (timed[pi]["link"], timed[pi]["name"]))
            nearest_d.append(float(best[j]))
            pixels.append(uvh[j])
            ray_hits.append(
                self.first_ray_self_intersection(origin, P[j], timed))

        nearest_d = np.asarray(nearest_d, dtype=np.float64)
        self.latest_markers = self.make_markers(
            stamp, Tbs, timed, self_pts, suspects, ray_hits,
            nearest_labels, nearest_d, pixels)

        if len(suspects):
            cross = sum(r is not None for r in ray_hits)
            k = int(np.argmax(nearest_d))
            r = ray_hits[k]
            rtxt = "none"
            if r is not None:
                p = timed[r[1]]
                measured = float(np.linalg.norm(suspects[k] - origin))
                rtxt = "%s/%s entry=%.3fm overshoot=%.1fmm" % (
                    p["link"], p["name"], r[0],
                    1000.0 * (measured - r[0]))
            self.latest_summary = (
                "stamp=%.3f sensor=%d:%s q=%s suspect=%d self=%d "
                "ray_cross_self=%d/%d worst_pixel=%s worst_endpoint=%s "
                "nearest=%s outside=%.2fmm ray_first=%s" %
                (stamp.to_sec(), self.sensor_id, self.sensor_name,
                 ("[" + ",".join("%.5f" % x for x in self.latest_q) + "]")
                 if self.latest_q is not None else "unavailable",
                 len(suspects), len(self_pts), cross, len(suspects),
                 pixels[k], np.array2string(suspects[k], precision=5),
                 nearest_labels[k], 1000.0 * nearest_d[k], rtxt))
            rospy.logwarn_throttle(
                1.0, "[RUNTIME_SELF_HIT_DIAG] " + self.latest_summary)


def main():
    rospy.init_node("runtime_self_hit_rviz_diagnostic")
    RuntimeSelfHitDiag()
    rospy.spin()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare Gazebo's actual simulated link / ToF render poses against ROS TF.

Diagnostic only. No mapping, filtering, planner, or controller semantics change.

The key comparison is for link4_sensor2_tof_gz_link:
  - Gazebo /gazebo/link_states gives the frame that physically carries the
    depth sensor in simulation.
  - ROS TF gives the frame used by the self-filter / point-cloud transform.

If those differ while the arm is static, a robot self-return can be rendered
from one pose but transformed / filtered against another pose.

The node prints both vector and angular deltas and publishes RViz markers:
  yellow sphere : TF optical origin (link4_sensor2_tof_link)
  purple sphere : Gazebo actual render-link origin
  white line    : position delta between them
"""

import math
import numpy as np
import rospy
import tf.transformations as tft
import tf2_ros

from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def T_from_pose(pose):
    q = pose.orientation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def T_from_tf(msg):
    q = msg.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    p = msg.transform.translation
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def angle_deg(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    return math.degrees(math.acos(c))


def mk_point(v):
    p = Point()
    p.x, p.y, p.z = map(float, v)
    return p


def color(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


class Node:
    def __init__(self):
        self.base = rospy.get_param("~base_frame", "base_link")
        self.model = rospy.get_param("~model_name", "care_arm")
        self.raw_topic = rospy.get_param(
            "~raw_topic", "/link4_sensor2/tof/cloud")
        self.optical = rospy.get_param(
            "~optical_frame", "link4_sensor2_tof_link")
        self.render = rospy.get_param(
            "~render_frame", "link4_sensor2_tof_gz_link")
        self.links = rospy.get_param(
            "~links", ["link1", "link2", "link3", "link4"])
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.05))
        self.log_period = float(rospy.get_param("~log_period", 1.0))

        self.buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buf)
        self.gz = {}
        self.last_log = rospy.Time(0)

        self.pub = rospy.Publisher(
            "/care_planner/debug/frame_delta_markers",
            MarkerArray, queue_size=1)

        rospy.Subscriber(
            "/gazebo/link_states", LinkStates, self.on_links, queue_size=1)
        rospy.Subscriber(
            self.raw_topic, PointCloud2, self.on_cloud, queue_size=1)

        rospy.logwarn(
            "[GAZEBO_TF_DELTA] ready optical=%s render=%s raw=%s",
            self.optical, self.render, self.raw_topic)

    def on_links(self, msg):
        poses = {}
        for name, pose in zip(msg.name, msg.pose):
            if "::" not in name:
                continue
            model, short = name.split("::", 1)
            if model != self.model:
                continue
            poses[short] = T_from_pose(pose)
        if self.base not in poses:
            return
        T_bw = np.linalg.inv(poses[self.base])
        self.gz = {name: T_bw.dot(T) for name, T in poses.items()}

    def tf(self, source, stamp):
        m = self.buf.lookup_transform(
            self.base, source, stamp, rospy.Duration(self.tf_timeout))
        return T_from_tf(m)

    def compare(self, name, stamp):
        if name not in self.gz:
            return None
        try:
            Ttf = self.tf(name, stamp)
        except Exception:
            return None
        Tgz = self.gz[name]
        d = Tgz[:3, 3] - Ttf[:3, 3]
        a = angle_deg(Ttf[:3, :3].T.dot(Tgz[:3, :3]))
        return Ttf, Tgz, d, float(np.linalg.norm(d)), a

    def publish_sensor_delta(self, stamp, optical_cmp, render_cmp):
        arr = MarkerArray()
        if optical_cmp is None or render_cmp is None:
            return

        Topt = optical_cmp[0]
        Trender_gz = render_cmp[1]

        m = Marker()
        m.header.frame_id = self.base
        m.header.stamp = stamp
        m.ns = "frame_delta"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.pose.position = mk_point(Topt[:3, 3])
        m.scale.x = m.scale.y = m.scale.z = 0.025
        m.color = color(1.0, 0.85, 0.0, 1.0)
        m.lifetime = rospy.Duration(1.2)
        arr.markers.append(m)

        m = Marker()
        m.header.frame_id = self.base
        m.header.stamp = stamp
        m.ns = "frame_delta"
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.pose.position = mk_point(Trender_gz[:3, 3])
        m.scale.x = m.scale.y = m.scale.z = 0.020
        m.color = color(0.75, 0.2, 1.0, 1.0)
        m.lifetime = rospy.Duration(1.2)
        arr.markers.append(m)

        m = Marker()
        m.header.frame_id = self.base
        m.header.stamp = stamp
        m.ns = "frame_delta"
        m.id = 2
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.006
        m.color = color(1.0, 1.0, 1.0, 1.0)
        m.points = [
            mk_point(Topt[:3, 3]),
            mk_point(Trender_gz[:3, 3])]
        m.lifetime = rospy.Duration(1.2)
        arr.markers.append(m)

        self.pub.publish(arr)

    def on_cloud(self, msg):
        if not self.gz:
            return
        stamp = msg.header.stamp
        if stamp == rospy.Time():
            return

        try:
            optical_cmp = self.compare(self.optical, stamp)
            render_cmp = self.compare(self.render, stamp)
        except Exception:
            return

        self.publish_sensor_delta(stamp, optical_cmp, render_cmp)

        now = rospy.Time.now()
        if (not self.last_log.is_zero() and
                (now - self.last_log).to_sec() < self.log_period):
            return
        self.last_log = now

        parts = []
        for name in list(self.links) + [self.render]:
            cmpv = self.compare(name, stamp)
            if cmpv is None:
                parts.append("%s=unavailable" % name)
                continue
            _, _, d, n, a = cmpv
            parts.append(
                "%s dxyz_mm=[%.3f,%.3f,%.3f] norm_mm=%.3f deg=%.4f" %
                (name, 1000*d[0], 1000*d[1], 1000*d[2],
                 1000*n, a))

        # The render link is rotated relative to the optical frame, so compare
        # translations only for the two origins (the fixed joint has zero xyz).
        sensor_origin_delta = "unavailable"
        if optical_cmp is not None and render_cmp is not None:
            Topt_tf = optical_cmp[0]
            Trender_gz = render_cmp[1]
            d = Trender_gz[:3, 3] - Topt_tf[:3, 3]
            sensor_origin_delta = (
                "render_origin_minus_tf_optical_mm=[%.3f,%.3f,%.3f] "
                "norm_mm=%.3f" %
                (1000*d[0], 1000*d[1], 1000*d[2],
                 1000*np.linalg.norm(d)))

        rospy.logwarn(
            "[GAZEBO_TF_DELTA] stamp=%.6f %s %s",
            stamp.to_sec(), " | ".join(parts), sensor_origin_delta)


def main():
    rospy.init_node("diagnose_gazebo_tf_frame_delta")
    Node()
    rospy.spin()


if __name__ == "__main__":
    main()

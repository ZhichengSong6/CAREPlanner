#!/usr/bin/env python3
"""Robust static Gazebo-vs-TF pose snapshot for Phase-E self-hit debugging.

This diagnostic is intentionally service-based rather than relying on
/gazebo/link_states callbacks.  The robot is static when it is called, so we
can compare latest ROS TF against Gazebo's /gazebo/get_link_state result
without any cloud-timestamp ambiguity.

It reports, repeatedly:
  Gazebo(actual link pose relative to base_link)
  minus
  ROS TF(base_link -> same link)

for link1..link4 and the preserved ToF render link
link4_sensor2_tof_gz_link.

It also subscribes to the raw ToF cloud and stores a deterministic stride-8
depth signature so repeated static startups can be compared directly.

No controller, mapping, filtering, or planner semantics are modified.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tft
import tf2_ros

from gazebo_msgs.srv import GetLinkState
from sensor_msgs.msg import PointCloud2


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


class Capture:
    def __init__(self, args):
        self.args = args
        self.buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buf)
        rospy.wait_for_service("/gazebo/get_link_state", timeout=10.0)
        self.get_link = rospy.ServiceProxy(
            "/gazebo/get_link_state", GetLinkState)

        self.cloud_signatures = []
        self.sub = rospy.Subscriber(
            args.raw_topic, PointCloud2, self.on_cloud, queue_size=1)

    def gazebo_world_T(self, link):
        # Gazebo service accepts either scoped or unscoped names, but use the
        # scoped form explicitly so fixed-joint-preserved links are unambiguous.
        resp = self.get_link(
            self.args.model_name + "::" + link, "world")
        if not resp.success:
            raise RuntimeError(
                "get_link_state failed for %s: %s" %
                (link, resp.status_message))
        return T_from_pose(resp.link_state.pose)

    def tf_base_T(self, link):
        msg = self.buf.lookup_transform(
            self.args.base_frame, link, rospy.Time(0),
            rospy.Duration(self.args.tf_timeout))
        return T_from_tf(msg)

    def snapshot(self):
        T_world_base = self.gazebo_world_T(self.args.base_frame)
        T_base_world = np.linalg.inv(T_world_base)

        out = {}
        for link in self.args.links:
            try:
                T_gz = T_base_world.dot(self.gazebo_world_T(link))
                T_tf = self.tf_base_T(link)
                d = T_gz[:3, 3] - T_tf[:3, 3]
                a = angle_deg(T_tf[:3, :3].T.dot(T_gz[:3, :3]))
                out[link] = {
                    "gazebo_xyz": T_gz[:3, 3].tolist(),
                    "tf_xyz": T_tf[:3, 3].tolist(),
                    "delta_xyz_m": d.tolist(),
                    "delta_norm_m": float(np.linalg.norm(d)),
                    "delta_angle_deg": float(a),
                }
            except Exception as exc:
                out[link] = {"error": str(exc)}
        return out

    def on_cloud(self, msg):
        if len(self.cloud_signatures) >= self.args.max_clouds:
            return
        if msg.width <= 0 or msg.height <= 1:
            return

        stride = self.args.pixel_stride
        uvs = [
            (u, v)
            for v in range(0, int(msg.height), stride)
            for u in range(0, int(msg.width), stride)
        ]
        vals = list(pc2.read_points(
            msg, field_names=("x", "y", "z"),
            skip_nans=False, uvs=uvs))
        arr = np.asarray(vals, dtype=np.float64)
        finite = np.isfinite(arr).all(axis=1)

        # Quantize to micrometres before hashing.  This avoids irrelevant
        # float-format differences while still making different rendered
        # depth patterns obvious.
        q = np.full(arr.shape, np.iinfo(np.int32).min, dtype=np.int32)
        q[finite] = np.rint(arr[finite] * 1e6).astype(np.int32)
        digest = hashlib.sha256(q.tobytes()).hexdigest()

        z = arr[finite, 2] if np.any(finite) else np.asarray([])
        self.cloud_signatures.append({
            "stamp": msg.header.stamp.to_sec(),
            "width": int(msg.width),
            "height": int(msg.height),
            "stride": int(stride),
            "finite_count": int(np.count_nonzero(finite)),
            "depth_z_min_m": float(np.min(z)) if z.size else None,
            "depth_z_median_m": float(np.median(z)) if z.size else None,
            "depth_z_max_m": float(np.max(z)) if z.size else None,
            "sha256_stride8_xyz_um": digest,
        })

    def run(self):
        records = []
        for k in range(self.args.samples):
            rec = {
                "sample": k,
                "ros_time": rospy.Time.now().to_sec(),
                "links": self.snapshot(),
            }
            records.append(rec)

            print("[STATIC_GZ_TF_SNAPSHOT] sample=%d" % k, flush=True)
            for name, v in rec["links"].items():
                if "error" in v:
                    print("  %s ERROR %s" % (name, v["error"]), flush=True)
                else:
                    d = v["delta_xyz_m"]
                    print(
                        "  %s dxyz_mm=[%.3f,%.3f,%.3f] "
                        "norm_mm=%.3f deg=%.5f" %
                        (name, 1000*d[0], 1000*d[1], 1000*d[2],
                         1000*v["delta_norm_m"],
                         v["delta_angle_deg"]),
                        flush=True)
            rospy.sleep(self.args.period)

        # Give the raw-cloud callback a short chance to collect signatures.
        deadline = time.time() + 2.0
        while (len(self.cloud_signatures) < self.args.max_clouds and
               time.time() < deadline and not rospy.is_shutdown()):
            rospy.sleep(0.02)

        result = {
            "base_frame": self.args.base_frame,
            "model_name": self.args.model_name,
            "raw_topic": self.args.raw_topic,
            "links": self.args.links,
            "pose_samples": records,
            "raw_cloud_signatures": self.cloud_signatures,
        }
        Path(self.args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(self.args.output).write_text(json.dumps(result, indent=2))

        print("[STATIC_GZ_TF_SNAPSHOT] output=%s" % self.args.output,
              flush=True)
        for s in self.cloud_signatures:
            print(
                "  cloud stamp=%.6f finite=%d hash=%s" %
                (s["stamp"], s["finite_count"],
                 s["sha256_stride8_xyz_um"][:16]),
                flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--model-name", default="care_arm")
    ap.add_argument("--raw-topic", default="/link4_sensor2/tof/cloud")
    ap.add_argument(
        "--links", nargs="+",
        default=[
            "link1", "link2", "link3", "link4",
            "link4_sensor2_tof_gz_link"])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--period", type=float, default=0.20)
    ap.add_argument("--tf-timeout", type=float, default=0.10)
    ap.add_argument("--pixel-stride", type=int, default=8)
    ap.add_argument("--max-clouds", type=int, default=8)
    args = ap.parse_args()

    rospy.init_node(
        "capture_static_gazebo_tf_snapshot",
        anonymous=True, disable_signals=True)
    Capture(args).run()


if __name__ == "__main__":
    main()

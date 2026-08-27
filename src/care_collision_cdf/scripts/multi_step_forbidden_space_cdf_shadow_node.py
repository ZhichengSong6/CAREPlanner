#!/usr/bin/env python3
"""C5.2 shadow diagnostic for multi-step forbidden-space CDF constraints.

Consumes low-confidence body-sweep samples exported by trajectory_risk_node.
Each PointCloud2 row already contains both workspace point p and the exact
7-DoF q_k that generated that predicted sweep sample, so there is no
cross-topic trajectory synchronization ambiguity.

The node:
  * deduplicates repeated samples within the same (timestep, confidence voxel),
  * evaluates all remaining (p, q_k) pairs in one batched CDF service call,
  * reports per-step and whole-horizon distance/gradient statistics,
  * never modifies the MPC, candidate trajectory, or VBC decision.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import defaultdict

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String

from care_collision_cdf.srv import QueryCollisionCDFPairs, QueryCollisionCDFPairsRequest


FIELDS = (
    "x", "y", "z",
    "q0", "q1", "q2", "q3", "q4", "q5", "q6",
    "confidence", "current_visibility", "radius",
    "eval_timestep", "original_timestep",
)


class MultiStepForbiddenSpaceCDFShadow:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/care_planner/trajectory_risk/low_confidence_sweep_pairs",
        )
        self.service_name = rospy.get_param(
            "~pair_service",
            "/care_planner/collision_cdf/query_pairs",
        )
        self.summary_topic = rospy.get_param(
            "~summary_topic",
            "/care_planner/collision_cdf/multi_step_forbidden_space_summary",
        )
        self.output_jsonl = os.path.abspath(
            os.path.expanduser(
                rospy.get_param(
                    "~output_jsonl",
                    "/tmp/c5_2_multi_step_forbidden_space_cdf.jsonl",
                )
            )
        )
        self.rate_hz = float(rospy.get_param("~rate", 5.0))
        self.stale_s = float(rospy.get_param("~stale_s", 0.5))
        self.dedup_resolution = float(
            rospy.get_param("~dedup_resolution", 0.05)
        )
        self.max_pairs = int(rospy.get_param("~max_pairs", 8000))

        if self.rate_hz <= 0.0:
            raise ValueError("~rate must be positive")
        if self.dedup_resolution <= 0.0:
            raise ValueError("~dedup_resolution must be positive")
        if self.max_pairs <= 0:
            raise ValueError("~max_pairs must be positive")

        out_dir = os.path.dirname(self.output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Start each experiment with a clean diagnostic stream.
        with open(self.output_jsonl, "w", encoding="utf-8"):
            pass

        self._lock = threading.Lock()
        self._latest_cloud = None
        self._latest_receive_time = rospy.Time(0)
        self._last_processed_stamp = None
        self._record_index = 0

        self._service = rospy.ServiceProxy(
            self.service_name, QueryCollisionCDFPairs, persistent=True
        )
        self._summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=10
        )
        self._sub = rospy.Subscriber(
            self.input_topic, PointCloud2, self._cloud_cb, queue_size=1
        )
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz), self._timer_cb
        )

        rospy.logwarn(
            "[C5.2 shadow] READY input=%s service=%s rate=%.2fHz "
            "dedup=%.3fm output=%s",
            self.input_topic,
            self.service_name,
            self.rate_hz,
            self.dedup_resolution,
            self.output_jsonl,
        )

    def _cloud_cb(self, msg):
        with self._lock:
            self._latest_cloud = msg
            self._latest_receive_time = rospy.Time.now()

    def _snapshot(self):
        with self._lock:
            return self._latest_cloud, self._latest_receive_time

    def _cell_key(self, x, y, z, original_timestep):
        r = self.dedup_resolution
        return (
            int(original_timestep),
            int(round(float(x) / r)),
            int(round(float(y) / r)),
            int(round(float(z) / r)),
        )

    def _decode_and_dedup(self, cloud):
        raw_rows = list(
            point_cloud2.read_points(
                cloud,
                field_names=FIELDS,
                skip_nans=True,
            )
        )

        # Keep the lowest-confidence representative if multiple body samples
        # land in the same confidence-map cell at the same horizon step.
        selected = {}
        for row in raw_rows:
            (
                x, y, z,
                q0, q1, q2, q3, q4, q5, q6,
                confidence, current_visibility, radius,
                eval_timestep, original_timestep,
            ) = row

            values = np.asarray(
                [x, y, z, q0, q1, q2, q3, q4, q5, q6, confidence, radius],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                continue

            item = {
                "point": [float(x), float(y), float(z)],
                "q": [
                    float(q0), float(q1), float(q2), float(q3),
                    float(q4), float(q5), float(q6),
                ],
                "confidence": float(confidence),
                "current_visibility": float(current_visibility),
                "radius": float(radius),
                "eval_timestep": int(eval_timestep),
                "original_timestep": int(original_timestep),
            }
            key = self._cell_key(x, y, z, original_timestep)
            old = selected.get(key)
            if old is None or item["confidence"] < old["confidence"]:
                selected[key] = item

        items = list(selected.values())
        items.sort(
            key=lambda a: (
                a["original_timestep"],
                a["eval_timestep"],
                a["confidence"],
                a["point"][0],
                a["point"][1],
                a["point"][2],
            )
        )
        return len(raw_rows), items

    def _build_request(self, items):
        req = QueryCollisionCDFPairsRequest()
        req.num_pairs = len(items)
        req.points = []
        q_flat = []
        for item in items:
            p = Point()
            p.x, p.y, p.z = item["point"]
            req.points.append(p)
            q_flat.extend(item["q"])
        req.q_flat = q_flat
        return req

    @staticmethod
    def _quantiles(values):
        if not values:
            return {
                "min": None,
                "p05": None,
                "median": None,
                "mean": None,
                "max": None,
            }
        a = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(a)),
            "p05": float(np.quantile(a, 0.05)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "max": float(np.max(a)),
        }

    def _make_record(
        self, cloud, raw_count, items, distance, gradient, inference_ms, rtt_ms
    ):
        per_step = defaultdict(list)
        for idx, item in enumerate(items):
            per_step[item["original_timestep"]].append(idx)

        step_rows = []
        for original_timestep in sorted(per_step):
            indices = per_step[original_timestep]
            d = [float(distance[i]) for i in indices]
            conf = [float(items[i]["confidence"]) for i in indices]
            grad_norm = [
                float(np.linalg.norm(gradient[i])) for i in indices
            ]
            min_local = int(indices[int(np.argmin(np.asarray(d)))])
            step_rows.append(
                {
                    "original_timestep": int(original_timestep),
                    "eval_timestep": int(items[min_local]["eval_timestep"]),
                    "pair_count": len(indices),
                    "distance": self._quantiles(d),
                    "confidence": self._quantiles(conf),
                    "gradient_norm": self._quantiles(grad_norm),
                    "min_pair_point": items[min_local]["point"],
                    "min_pair_radius": items[min_local]["radius"],
                    "min_pair_confidence": items[min_local]["confidence"],
                    "min_pair_current_visibility": items[min_local][
                        "current_visibility"
                    ],
                }
            )

        all_distance = [float(x) for x in distance]
        all_grad_norm = [
            float(np.linalg.norm(gradient[i])) for i in range(len(items))
        ]
        min_idx = int(np.argmin(np.asarray(all_distance))) if items else -1
        counts = [row["pair_count"] for row in step_rows]

        return {
            "record_index": self._record_index,
            "wall_time": time.time(),
            "cloud_stamp": float(cloud.header.stamp.to_sec()),
            "frame_id": cloud.header.frame_id,
            "raw_pair_count": int(raw_count),
            "deduplicated_pair_count": len(items),
            "dedup_removed_count": int(raw_count - len(items)),
            "active_step_count": len(step_rows),
            "pairs_per_step": {
                "min": int(min(counts)) if counts else 0,
                "median": float(np.median(counts)) if counts else 0.0,
                "mean": float(np.mean(counts)) if counts else 0.0,
                "max": int(max(counts)) if counts else 0,
            },
            "distance": self._quantiles(all_distance),
            "gradient_norm": self._quantiles(all_grad_norm),
            "inference_ms": float(inference_ms),
            "service_roundtrip_ms": float(rtt_ms),
            "global_min_pair": (
                {
                    **items[min_idx],
                    "distance": all_distance[min_idx],
                    "gradient": gradient[min_idx].tolist(),
                    "gradient_norm": all_grad_norm[min_idx],
                }
                if min_idx >= 0
                else None
            ),
            "per_step": step_rows,
        }

    def _publish_summary(self, record):
        d = record["distance"]
        pair = record["global_min_pair"]
        if pair is None:
            text = (
                "C5_2_SHADOW pairs=0 active_steps=0 "
                "inference_ms=0.000"
            )
        else:
            text = (
                "C5_2_SHADOW "
                f"pairs={record['deduplicated_pair_count']} "
                f"raw_pairs={record['raw_pair_count']} "
                f"active_steps={record['active_step_count']} "
                f"d_min={d['min']:.6f} "
                f"d_p05={d['p05']:.6f} "
                f"d_median={d['median']:.6f} "
                f"min_step={pair['original_timestep']} "
                f"min_conf={pair['confidence']:.3f} "
                f"inference_ms={record['inference_ms']:.3f} "
                f"rtt_ms={record['service_roundtrip_ms']:.3f}"
            )
        self._summary_pub.publish(String(data=text))

    def _timer_cb(self, _event):
        cloud, receive_time = self._snapshot()
        if cloud is None:
            rospy.logwarn_throttle(
                2.0, "[C5.2 shadow] waiting for %s", self.input_topic
            )
            return

        age = (rospy.Time.now() - receive_time).to_sec()
        if age > self.stale_s:
            rospy.logwarn_throttle(
                2.0, "[C5.2 shadow] source cloud stale: %.3fs", age
            )
            return

        stamp_key = (
            int(cloud.header.stamp.secs),
            int(cloud.header.stamp.nsecs),
            int(cloud.width),
        )
        if stamp_key == self._last_processed_stamp:
            return
        self._last_processed_stamp = stamp_key

        raw_count, items = self._decode_and_dedup(cloud)
        if not items:
            record = {
                "record_index": self._record_index,
                "wall_time": time.time(),
                "cloud_stamp": float(cloud.header.stamp.to_sec()),
                "frame_id": cloud.header.frame_id,
                "raw_pair_count": int(raw_count),
                "deduplicated_pair_count": 0,
                "dedup_removed_count": int(raw_count),
                "active_step_count": 0,
                "pairs_per_step": {
                    "min": 0, "median": 0.0, "mean": 0.0, "max": 0
                },
                "distance": self._quantiles([]),
                "gradient_norm": self._quantiles([]),
                "inference_ms": 0.0,
                "service_roundtrip_ms": 0.0,
                "global_min_pair": None,
                "per_step": [],
            }
            self._append_record(record)
            self._publish_summary(record)
            self._record_index += 1
            return

        if len(items) > self.max_pairs:
            rospy.logerr_throttle(
                1.0,
                "[C5.2 shadow] pair batch %d exceeds max_pairs=%d",
                len(items),
                self.max_pairs,
            )
            return

        try:
            self._service.wait_for_service(timeout=0.20)
            req = self._build_request(items)
            t0 = time.perf_counter()
            res = self._service(req)
            rtt_ms = 1000.0 * (time.perf_counter() - t0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(
                1.0, "[C5.2 shadow] CDF pair service failed: %s", exc
            )
            # Persistent service may be invalid after a server restart.
            self._service = rospy.ServiceProxy(
                self.service_name, QueryCollisionCDFPairs, persistent=True
            )
            return

        if not res.success:
            rospy.logwarn_throttle(
                1.0, "[C5.2 shadow] CDF service rejected batch: %s", res.message
            )
            return
        if len(res.distance) != len(items):
            rospy.logerr_throttle(
                1.0,
                "[C5.2 shadow] distance length mismatch: %d vs %d",
                len(res.distance),
                len(items),
            )
            return
        if len(res.gradient_flat) != len(items) * 7:
            rospy.logerr_throttle(
                1.0,
                "[C5.2 shadow] gradient length mismatch: %d vs %d",
                len(res.gradient_flat),
                len(items) * 7,
            )
            return

        distance = np.asarray(res.distance, dtype=np.float64)
        gradient = np.asarray(
            res.gradient_flat, dtype=np.float64
        ).reshape(len(items), 7)

        record = self._make_record(
            cloud,
            raw_count,
            items,
            distance,
            gradient,
            res.inference_ms,
            rtt_ms,
        )
        self._append_record(record)
        self._publish_summary(record)

        pair = record["global_min_pair"]
        rospy.loginfo_throttle(
            1.0,
            "[C5.2 shadow] pairs=%d steps=%d d_min=%.4f "
            "min_step=%d inference=%.2fms rtt=%.2fms",
            record["deduplicated_pair_count"],
            record["active_step_count"],
            record["distance"]["min"],
            pair["original_timestep"],
            record["inference_ms"],
            record["service_roundtrip_ms"],
        )
        self._record_index += 1

    def _append_record(self, record):
        with open(self.output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()


def main():
    rospy.init_node("multi_step_forbidden_space_cdf_shadow")
    MultiStepForbiddenSpaceCDFShadow()
    rospy.spin()


if __name__ == "__main__":
    main()

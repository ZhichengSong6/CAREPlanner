#!/usr/bin/env python3
"""C5.2 multi-step forbidden-space signed-CDF shadow diagnostic.

This node is deliberately shadow-only: it never changes MPC or VBC decisions.

Inputs
------
1) /care_planner/trajectory_risk/body_sweep_anchors
   Predicted robot-body sample centers for the raw MPC trajectory.  Each row
   carries the exact q_k and body-sphere radius used by the existing trajectory
   risk model.  These centers are ONLY spatial query anchors; they are not sent
   to the collision CDF as obstacle points.

2) /care_planner/confidence_map/points
   The complete regular confidence grid.  Actual forbidden CDF points are voxel
   centers with confidence below the configured threshold.

For each predicted timestep k, nearby low-confidence voxel centers are selected
around the existing body-sphere approximation, deduplicated as (k, voxel), and
queried as one batched set of (p_voxel, q_k) pairs.

The signed CDF convention expected by C5 is:
  d > 0 : workspace point outside robot (safe side)
  d = 0 : robot-point contact
  d < 0 : workspace point inside robot (violation side)

A sphere-model clearance is also logged as a diagnostic proxy.  The exact VBC
verifier remains the safety authority.
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

from care_collision_cdf.srv import (
    QueryCollisionCDFPairs,
    QueryCollisionCDFPairsRequest,
)


ANCHOR_FIELDS = (
    "x", "y", "z",
    "q0", "q1", "q2", "q3", "q4", "q5", "q6",
    "confidence", "current_visibility", "radius",
    "eval_timestep", "original_timestep",
)

MAP_FIELDS = (
    "x", "y", "z", "confidence", "current_visibility",
)


class MultiStepForbiddenSpaceCDFShadow:
    def __init__(self):
        self.anchor_topic = rospy.get_param(
            "~anchor_topic",
            "/care_planner/trajectory_risk/body_sweep_anchors",
        )
        self.map_topic = rospy.get_param(
            "~map_topic",
            "/care_planner/confidence_map/points",
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
        self.anchor_stale_s = float(rospy.get_param("~anchor_stale_s", 0.5))
        self.map_stale_s = float(rospy.get_param("~map_stale_s", 0.5))
        self.map_resolution = float(rospy.get_param("~map_resolution", 0.05))
        self.confidence_threshold = float(
            rospy.get_param("~confidence_threshold", 0.50)
        )
        self.proximity_margin = float(
            rospy.get_param("~proximity_margin", 0.075)
        )
        self.max_pairs_per_step = int(
            rospy.get_param("~max_pairs_per_step", 350)
        )
        self.max_pairs = int(rospy.get_param("~max_pairs", 8000))
        self.zero_band = float(rospy.get_param("~signed_zero_band", 0.05))
        self.proxy_sign_margin_m = float(
            rospy.get_param("~proxy_sign_margin_m", 0.01)
        )

        if self.rate_hz <= 0.0:
            raise ValueError("~rate must be positive")
        if self.map_resolution <= 0.0:
            raise ValueError("~map_resolution must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("~confidence_threshold must be in [0,1]")
        if self.proximity_margin < 0.0:
            raise ValueError("~proximity_margin must be non-negative")
        if self.max_pairs_per_step <= 0 or self.max_pairs <= 0:
            raise ValueError("pair limits must be positive")
        if self.zero_band <= 0.0:
            raise ValueError("~signed_zero_band must be positive")

        out_dir = os.path.dirname(self.output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_jsonl, "w", encoding="utf-8"):
            pass

        self._lock = threading.Lock()
        self._latest_anchor_cloud = None
        self._latest_anchor_receive_time = rospy.Time(0)
        self._latest_map_cloud = None
        self._latest_map_receive_time = rospy.Time(0)
        self._last_processed_anchor_stamp = None
        self._record_index = 0

        self._service = rospy.ServiceProxy(
            self.service_name, QueryCollisionCDFPairs, persistent=True
        )
        self._summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=10
        )
        self._anchor_sub = rospy.Subscriber(
            self.anchor_topic, PointCloud2, self._anchor_cb, queue_size=1
        )
        self._map_sub = rospy.Subscriber(
            self.map_topic, PointCloud2, self._map_cb, queue_size=1
        )
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz), self._timer_cb
        )

        rospy.logwarn(
            "[C5.2 shadow] READY anchors=%s map=%s service=%s rate=%.2fHz "
            "conf<%.3f resolution=%.3fm proximity_margin=%.3fm "
            "max_pairs_per_step=%d output=%s",
            self.anchor_topic,
            self.map_topic,
            self.service_name,
            self.rate_hz,
            self.confidence_threshold,
            self.map_resolution,
            self.proximity_margin,
            self.max_pairs_per_step,
            self.output_jsonl,
        )

    def _anchor_cb(self, msg):
        with self._lock:
            self._latest_anchor_cloud = msg
            self._latest_anchor_receive_time = rospy.Time.now()

    def _map_cb(self, msg):
        with self._lock:
            self._latest_map_cloud = msg
            self._latest_map_receive_time = rospy.Time.now()

    def _snapshot(self):
        with self._lock:
            return (
                self._latest_anchor_cloud,
                self._latest_anchor_receive_time,
                self._latest_map_cloud,
                self._latest_map_receive_time,
            )

    def _grid_key(self, xyz):
        r = self.map_resolution
        return (
            int(round(float(xyz[0]) / r)),
            int(round(float(xyz[1]) / r)),
            int(round(float(xyz[2]) / r)),
        )

    def _decode_anchors(self, cloud):
        items = []
        for row in point_cloud2.read_points(
            cloud, field_names=ANCHOR_FIELDS, skip_nans=True
        ):
            (
                x, y, z,
                q0, q1, q2, q3, q4, q5, q6,
                confidence, current_visibility, radius,
                eval_timestep, original_timestep,
            ) = row
            values = np.asarray(
                [x, y, z, q0, q1, q2, q3, q4, q5, q6, radius],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                continue
            if float(radius) <= 0.0:
                continue
            items.append(
                {
                    "center": np.asarray([x, y, z], dtype=np.float64),
                    "q": np.asarray(
                        [q0, q1, q2, q3, q4, q5, q6],
                        dtype=np.float64,
                    ),
                    "anchor_confidence": float(confidence),
                    "anchor_current_visibility": float(current_visibility),
                    "radius": float(radius),
                    "eval_timestep": int(eval_timestep),
                    "original_timestep": int(original_timestep),
                }
            )
        return items

    def _decode_low_confidence_map(self, cloud):
        low = {}
        total = 0
        for row in point_cloud2.read_points(
            cloud, field_names=MAP_FIELDS, skip_nans=True
        ):
            x, y, z, confidence, current_visibility = row
            total += 1
            values = np.asarray(
                [x, y, z, confidence, current_visibility], dtype=np.float64
            )
            if not np.all(np.isfinite(values)):
                continue
            if float(confidence) >= self.confidence_threshold:
                continue
            xyz = np.asarray([x, y, z], dtype=np.float64)
            low[self._grid_key(xyz)] = {
                "point": xyz,
                "confidence": float(confidence),
                "current_visibility": float(current_visibility),
            }
        return total, low

    def _build_forbidden_pairs(self, anchors, low_map):
        anchors_by_step = defaultdict(list)
        q_by_step = {}
        eval_by_step = {}
        q_spread_by_step = defaultdict(float)

        for anchor in anchors:
            step = int(anchor["original_timestep"])
            if step < 0:
                continue
            anchors_by_step[step].append(anchor)
            if step not in q_by_step:
                q_by_step[step] = anchor["q"].copy()
                eval_by_step[step] = int(anchor["eval_timestep"])
            else:
                q_spread_by_step[step] = max(
                    q_spread_by_step[step],
                    float(np.max(np.abs(anchor["q"] - q_by_step[step]))),
                )

        raw_count_by_step = {}
        retained_count_by_step = {}
        pairs = []

        r = self.map_resolution
        for step in sorted(anchors_by_step):
            candidates = {}
            for anchor in anchors_by_step[step]:
                center = anchor["center"]
                radius = float(anchor["radius"])
                search_radius = radius + self.proximity_margin
                n = int(math.ceil(search_radius / r))
                ck = self._grid_key(center)

                for dx in range(-n, n + 1):
                    for dy in range(-n, n + 1):
                        for dz in range(-n, n + 1):
                            key = (ck[0] + dx, ck[1] + dy, ck[2] + dz)
                            voxel = low_map.get(key)
                            if voxel is None:
                                continue
                            delta = voxel["point"] - center
                            center_distance = float(np.linalg.norm(delta))
                            approx_clearance = center_distance - radius
                            if approx_clearance > self.proximity_margin:
                                continue

                            old = candidates.get(key)
                            if (
                                old is None
                                or approx_clearance
                                < old["approx_body_clearance_m"]
                            ):
                                candidates[key] = {
                                    "point": voxel["point"].copy(),
                                    "confidence": voxel["confidence"],
                                    "current_visibility": voxel[
                                        "current_visibility"
                                    ],
                                    "approx_body_clearance_m": float(
                                        approx_clearance
                                    ),
                                    "nearest_anchor_radius": radius,
                                    "nearest_anchor_center": center.copy(),
                                }

            raw_count_by_step[step] = len(candidates)
            ordered = sorted(
                candidates.values(),
                key=lambda item: (
                    item["approx_body_clearance_m"],
                    item["confidence"],
                    float(item["point"][0]),
                    float(item["point"][1]),
                    float(item["point"][2]),
                ),
            )
            if len(ordered) > self.max_pairs_per_step:
                ordered = ordered[: self.max_pairs_per_step]
            retained_count_by_step[step] = len(ordered)

            q = q_by_step[step]
            for item in ordered:
                pairs.append(
                    {
                        "point": item["point"].tolist(),
                        "q": q.tolist(),
                        "confidence": float(item["confidence"]),
                        "current_visibility": float(
                            item["current_visibility"]
                        ),
                        "approx_body_clearance_m": float(
                            item["approx_body_clearance_m"]
                        ),
                        "nearest_anchor_radius": float(
                            item["nearest_anchor_radius"]
                        ),
                        "nearest_anchor_center": item[
                            "nearest_anchor_center"
                        ].tolist(),
                        "eval_timestep": int(eval_by_step[step]),
                        "original_timestep": int(step),
                    }
                )

        if len(pairs) > self.max_pairs:
            # The per-step cap should normally keep us below max_pairs.  If a
            # non-standard horizon exceeds it, retain the spatially closest
            # pairs globally rather than failing the CDF service.
            pairs = sorted(
                pairs,
                key=lambda item: item["approx_body_clearance_m"],
            )[: self.max_pairs]

        metadata = {
            "active_anchor_step_count": len(anchors_by_step),
            "raw_local_pair_count": int(sum(raw_count_by_step.values())),
            "retained_pair_count_before_global_cap": int(
                sum(retained_count_by_step.values())
            ),
            "q_spread_inf_max": (
                float(max(q_spread_by_step.values()))
                if q_spread_by_step else 0.0
            ),
            "raw_count_by_step": raw_count_by_step,
            "retained_count_by_step": retained_count_by_step,
        }
        return pairs, metadata

    def _build_request(self, pairs):
        req = QueryCollisionCDFPairsRequest()
        req.num_pairs = len(pairs)
        req.points = []
        q_flat = []
        for item in pairs:
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
                "p95": None,
                "max": None,
            }
        a = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(a)),
            "p05": float(np.quantile(a, 0.05)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "p95": float(np.quantile(a, 0.95)),
            "max": float(np.max(a)),
        }

    def _signed_counts(self, d):
        arr = np.asarray(d, dtype=np.float64)
        n = int(arr.size)
        neg = int(np.sum(arr < -self.zero_band))
        zero = int(np.sum(np.abs(arr) <= self.zero_band))
        pos = int(np.sum(arr > self.zero_band))
        return {
            "negative": neg,
            "near_zero": zero,
            "positive": pos,
            "negative_rate": float(neg) / n if n else 0.0,
            "near_zero_rate": float(zero) / n if n else 0.0,
            "positive_rate": float(pos) / n if n else 0.0,
        }

    def _proxy_sign_stats(self, pairs, distance):
        d = np.asarray(distance, dtype=np.float64)
        clearance = np.asarray(
            [p["approx_body_clearance_m"] for p in pairs],
            dtype=np.float64,
        )
        inside = clearance < -self.proxy_sign_margin_m
        outside = clearance > self.proxy_sign_margin_m
        negative = d < -self.zero_band
        positive = d > self.zero_band

        return {
            "proxy_sign_margin_m": self.proxy_sign_margin_m,
            "proxy_inside_count": int(np.sum(inside)),
            "proxy_outside_count": int(np.sum(outside)),
            "proxy_ambiguous_count": int(
                len(pairs) - np.sum(inside) - np.sum(outside)
            ),
            "negative_given_proxy_inside_rate": (
                float(np.mean(negative[inside])) if np.any(inside) else None
            ),
            "positive_given_proxy_outside_rate": (
                float(np.mean(positive[outside])) if np.any(outside) else None
            ),
        }

    def _make_record(
        self,
        anchor_cloud,
        map_cloud,
        anchors,
        map_total_count,
        low_map_count,
        pairs,
        pair_meta,
        distance,
        gradient,
        inference_ms,
        rtt_ms,
    ):
        per_step = defaultdict(list)
        for idx, item in enumerate(pairs):
            per_step[item["original_timestep"]].append(idx)

        step_rows = []
        for step in sorted(per_step):
            indices = per_step[step]
            d = [float(distance[i]) for i in indices]
            gnorm = [
                float(np.linalg.norm(gradient[i])) for i in indices
            ]
            conf = [float(pairs[i]["confidence"]) for i in indices]
            approx = [
                float(pairs[i]["approx_body_clearance_m"]) for i in indices
            ]
            min_local = indices[int(np.argmin(np.asarray(d)))]
            step_rows.append(
                {
                    "original_timestep": int(step),
                    "eval_timestep": int(
                        pairs[min_local]["eval_timestep"]
                    ),
                    "raw_local_pair_count": int(
                        pair_meta["raw_count_by_step"].get(step, 0)
                    ),
                    "pair_count": len(indices),
                    "distance": self._quantiles(d),
                    "signed_counts": self._signed_counts(d),
                    "confidence": self._quantiles(conf),
                    "gradient_norm": self._quantiles(gnorm),
                    "approx_body_clearance_m": self._quantiles(approx),
                    "proxy_sign": self._proxy_sign_stats(
                        [pairs[i] for i in indices], d
                    ),
                    "min_cdf_pair": {
                        **pairs[min_local],
                        "distance": float(distance[min_local]),
                        "gradient": gradient[min_local].tolist(),
                        "gradient_norm": float(
                            np.linalg.norm(gradient[min_local])
                        ),
                    },
                }
            )

        all_distance = [float(v) for v in distance]
        all_gnorm = [
            float(np.linalg.norm(gradient[i])) for i in range(len(pairs))
        ]
        all_approx = [
            float(p["approx_body_clearance_m"]) for p in pairs
        ]
        min_idx = (
            int(np.argmin(np.asarray(all_distance))) if pairs else -1
        )
        deepest_idx = (
            int(np.argmin(np.asarray(all_approx))) if pairs else -1
        )

        return {
            "record_index": self._record_index,
            "wall_time": time.time(),
            "anchor_cloud_stamp": float(anchor_cloud.header.stamp.to_sec()),
            "map_cloud_stamp": float(map_cloud.header.stamp.to_sec()),
            "frame_id": map_cloud.header.frame_id,
            "anchor_count": len(anchors),
            "map_voxel_count": int(map_total_count),
            "low_confidence_voxel_count": int(low_map_count),
            "active_anchor_step_count": int(
                pair_meta["active_anchor_step_count"]
            ),
            "raw_local_pair_count": int(
                pair_meta["raw_local_pair_count"]
            ),
            "retained_pair_count_before_global_cap": int(
                pair_meta["retained_pair_count_before_global_cap"]
            ),
            "retained_pair_count": len(pairs),
            # Backward-compatible key used by the C5.2 packer.
            "deduplicated_pair_count": len(pairs),
            "active_step_count": len(step_rows),
            "q_spread_inf_max": float(pair_meta["q_spread_inf_max"]),
            "distance": self._quantiles(all_distance),
            "signed_zero_band": self.zero_band,
            "signed_counts": self._signed_counts(all_distance),
            "gradient_norm": self._quantiles(all_gnorm),
            "approx_body_clearance_m": self._quantiles(all_approx),
            "proxy_sign": self._proxy_sign_stats(pairs, all_distance),
            "inference_ms": float(inference_ms),
            "service_roundtrip_ms": float(rtt_ms),
            "global_min_pair": (
                {
                    **pairs[min_idx],
                    "distance": all_distance[min_idx],
                    "gradient": gradient[min_idx].tolist(),
                    "gradient_norm": all_gnorm[min_idx],
                }
                if min_idx >= 0 else None
            ),
            "deepest_proxy_overlap_pair": (
                {
                    **pairs[deepest_idx],
                    "distance": all_distance[deepest_idx],
                    "gradient": gradient[deepest_idx].tolist(),
                    "gradient_norm": all_gnorm[deepest_idx],
                }
                if deepest_idx >= 0 else None
            ),
            "per_step": step_rows,
        }

    def _publish_summary(self, record):
        pair = record.get("global_min_pair")
        if pair is None:
            text = (
                "C5_2_FORBIDDEN_SHADOW pairs=0 "
                f"anchors={record.get('anchor_count', 0)}"
            )
        else:
            d = record["distance"]
            signed = record["signed_counts"]
            proxy = record["proxy_sign"]
            inside_rate = proxy["negative_given_proxy_inside_rate"]
            inside_text = (
                "nan" if inside_rate is None else f"{inside_rate:.3f}"
            )
            text = (
                "C5_2_FORBIDDEN_SHADOW "
                f"pairs={record['retained_pair_count']} "
                f"raw_pairs={record['raw_local_pair_count']} "
                f"anchors={record['anchor_count']} "
                f"low_voxels={record['low_confidence_voxel_count']} "
                f"steps={record['active_step_count']} "
                f"d_min={d['min']:.6f} "
                f"d_p05={d['p05']:.6f} "
                f"neg_rate={signed['negative_rate']:.3f} "
                f"proxy_inside_neg_rate={inside_text} "
                f"min_step={pair['original_timestep']} "
                f"inference_ms={record['inference_ms']:.3f} "
                f"rtt_ms={record['service_roundtrip_ms']:.3f}"
            )
        self._summary_pub.publish(String(data=text))

    def _append_record(self, record):
        with open(self.output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()

    def _timer_cb(self, _event):
        (
            anchor_cloud,
            anchor_receive_time,
            map_cloud,
            map_receive_time,
        ) = self._snapshot()

        if anchor_cloud is None:
            rospy.logwarn_throttle(
                2.0, "[C5.2 shadow] waiting for anchors: %s",
                self.anchor_topic,
            )
            return
        if map_cloud is None:
            rospy.logwarn_throttle(
                2.0, "[C5.2 shadow] waiting for map: %s", self.map_topic
            )
            return

        now = rospy.Time.now()
        if (now - anchor_receive_time).to_sec() > self.anchor_stale_s:
            rospy.logwarn_throttle(
                2.0, "[C5.2 shadow] anchor cloud stale"
            )
            return
        if (now - map_receive_time).to_sec() > self.map_stale_s:
            rospy.logwarn_throttle(
                2.0, "[C5.2 shadow] confidence map cloud stale"
            )
            return

        stamp_key = (
            int(anchor_cloud.header.stamp.secs),
            int(anchor_cloud.header.stamp.nsecs),
            int(anchor_cloud.width),
        )
        if stamp_key == self._last_processed_anchor_stamp:
            return
        self._last_processed_anchor_stamp = stamp_key

        anchors = self._decode_anchors(anchor_cloud)
        map_total_count, low_map = self._decode_low_confidence_map(map_cloud)
        pairs, pair_meta = self._build_forbidden_pairs(anchors, low_map)

        if not pairs:
            record = {
                "record_index": self._record_index,
                "wall_time": time.time(),
                "anchor_cloud_stamp": float(
                    anchor_cloud.header.stamp.to_sec()
                ),
                "map_cloud_stamp": float(map_cloud.header.stamp.to_sec()),
                "frame_id": map_cloud.header.frame_id,
                "anchor_count": len(anchors),
                "map_voxel_count": int(map_total_count),
                "low_confidence_voxel_count": len(low_map),
                "active_anchor_step_count": int(
                    pair_meta["active_anchor_step_count"]
                ),
                "raw_local_pair_count": int(
                    pair_meta["raw_local_pair_count"]
                ),
                "retained_pair_count": 0,
                "deduplicated_pair_count": 0,
                "active_step_count": 0,
                "distance": self._quantiles([]),
                "signed_counts": self._signed_counts([]),
                "proxy_sign": self._proxy_sign_stats([], []),
                "inference_ms": 0.0,
                "service_roundtrip_ms": 0.0,
                "global_min_pair": None,
                "deepest_proxy_overlap_pair": None,
                "per_step": [],
            }
            self._append_record(record)
            self._publish_summary(record)
            self._record_index += 1
            return

        if len(pairs) > self.max_pairs:
            rospy.logerr_throttle(
                1.0,
                "[C5.2 shadow] retained pair batch %d exceeds max_pairs=%d",
                len(pairs),
                self.max_pairs,
            )
            return

        try:
            self._service.wait_for_service(timeout=0.20)
            req = self._build_request(pairs)
            t0 = time.perf_counter()
            res = self._service(req)
            rtt_ms = 1000.0 * (time.perf_counter() - t0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(
                1.0, "[C5.2 shadow] CDF pair service failed: %s", exc
            )
            self._service = rospy.ServiceProxy(
                self.service_name,
                QueryCollisionCDFPairs,
                persistent=True,
            )
            return

        if not res.success:
            rospy.logwarn_throttle(
                1.0, "[C5.2 shadow] CDF service rejected batch: %s",
                res.message,
            )
            return
        if len(res.distance) != len(pairs):
            rospy.logerr(
                "[C5.2 shadow] distance length mismatch: %d vs %d",
                len(res.distance), len(pairs),
            )
            return
        if len(res.gradient_flat) != len(pairs) * 7:
            rospy.logerr(
                "[C5.2 shadow] gradient length mismatch: %d vs %d",
                len(res.gradient_flat), len(pairs) * 7,
            )
            return

        distance = np.asarray(res.distance, dtype=np.float64)
        gradient = np.asarray(
            res.gradient_flat, dtype=np.float64
        ).reshape(len(pairs), 7)

        record = self._make_record(
            anchor_cloud,
            map_cloud,
            anchors,
            map_total_count,
            len(low_map),
            pairs,
            pair_meta,
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
            "[C5.2 shadow] pairs=%d raw=%d anchors=%d low_voxels=%d "
            "steps=%d d_min=%.4f neg_rate=%.3f min_step=%d "
            "inference=%.2fms rtt=%.2fms",
            record["retained_pair_count"],
            record["raw_local_pair_count"],
            record["anchor_count"],
            record["low_confidence_voxel_count"],
            record["active_step_count"],
            record["distance"]["min"],
            record["signed_counts"]["negative_rate"],
            pair["original_timestep"],
            record["inference_ms"],
            record["service_roundtrip_ms"],
        )
        self._record_index += 1


def main():
    rospy.init_node("multi_step_forbidden_space_cdf_shadow")
    MultiStepForbiddenSpaceCDFShadow()
    rospy.spin()


if __name__ == "__main__":
    main()

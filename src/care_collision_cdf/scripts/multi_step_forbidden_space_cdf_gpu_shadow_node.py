#!/usr/bin/env python3
"""C5.2g online end-to-end GPU shadow benchmark.

This node runs in the ordinary ROS Python environment. It never imports torch.
A persistent CUDA worker in the viscdf environment performs signed-CDF inference
through a Unix-domain socket.

Measured planner-side path per fresh raw-MPC prediction:
  anchor PointCloud2 decode
  + local low-confidence voxel selection
  + binary request packing
  + Unix-socket round trip
  + GPU H2D / signed-CDF forward+autograd / D2H

The node is shadow-only and never changes MPC, VBC, commit, or execution.
"""

from __future__ import annotations

import json
import math
import os
import socket
import struct
import threading
import time
from collections import defaultdict

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


REQ_MAGIC = b"CQ01"
RESP_MAGIC = b"CR01"
REQ_HEADER = struct.Struct("<4sI")
RESP_HEADER = struct.Struct("<4sI4d")

ANCHOR_FIELDS = (
    "x", "y", "z",
    "q0", "q1", "q2", "q3", "q4", "q5", "q6",
    "confidence", "current_visibility", "radius",
    "eval_timestep", "original_timestep",
)
MAP_FIELDS = (
    "x", "y", "z", "confidence", "current_visibility",
)


def recv_exact(sock, nbytes):
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("GPU worker closed socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def quantiles(values):
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


class GPUWorkerClient:
    def __init__(self, socket_path, timeout_s):
        self.socket_path = os.path.abspath(os.path.expanduser(socket_path))
        self.timeout_s = float(timeout_s)
        self.sock = None

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def connect(self):
        self.close()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect(self.socket_path)
        self.sock = sock

    def query(self, pairs):
        if not pairs:
            raise ValueError("pairs must be non-empty")

        pack_t0 = time.perf_counter()
        rows = np.empty((len(pairs), 10), dtype="<f4")
        for i, item in enumerate(pairs):
            rows[i, 0:3] = item["point"]
            rows[i, 3:10] = item["q"]
        payload = rows.tobytes(order="C")
        header = REQ_HEADER.pack(REQ_MAGIC, len(pairs))
        pack_ms = (time.perf_counter() - pack_t0) * 1000.0

        if self.sock is None:
            self.connect()

        ipc_t0 = time.perf_counter()
        try:
            self.sock.sendall(header)
            self.sock.sendall(payload)
            response_header = recv_exact(self.sock, RESP_HEADER.size)
            (
                magic,
                pair_count,
                h2d_ms,
                inference_ms,
                d2h_ms,
                worker_total_ms,
            ) = RESP_HEADER.unpack(response_header)
            if magic != RESP_MAGIC:
                raise ValueError("bad GPU worker response magic")
            if int(pair_count) != len(pairs):
                raise ValueError(
                    "GPU worker pair count mismatch: {} vs {}".format(
                        pair_count, len(pairs)
                    )
                )

            distance_bytes = recv_exact(
                self.sock, int(pair_count) * 4
            )
            gradient_bytes = recv_exact(
                self.sock, int(pair_count) * 7 * 4
            )
        except Exception:
            self.close()
            raise

        ipc_ms = (time.perf_counter() - ipc_t0) * 1000.0

        distance = np.frombuffer(
            distance_bytes, dtype="<f4"
        ).astype(np.float64, copy=True)
        gradient = np.frombuffer(
            gradient_bytes, dtype="<f4"
        ).reshape(int(pair_count), 7).astype(np.float64, copy=True)

        return {
            "distance": distance,
            "gradient": gradient,
            "request_pack_ms": pack_ms,
            "ipc_roundtrip_ms": ipc_ms,
            "worker_h2d_ms": float(h2d_ms),
            "worker_inference_ms": float(inference_ms),
            "worker_d2h_ms": float(d2h_ms),
            "worker_total_ms": float(worker_total_ms),
        }


class OnlineGPUShadow:
    def __init__(self):
        self.anchor_topic = rospy.get_param(
            "~anchor_topic",
            "/care_planner/trajectory_risk/body_sweep_anchors",
        )
        self.map_topic = rospy.get_param(
            "~map_topic",
            "/care_planner/confidence_map/points",
        )
        self.summary_topic = rospy.get_param(
            "~summary_topic",
            "/care_planner/collision_cdf/gpu_online_summary",
        )
        self.output_jsonl = os.path.abspath(
            os.path.expanduser(
                rospy.get_param(
                    "~output_jsonl",
                    "/tmp/c5_2g_gpu_online.jsonl",
                )
            )
        )
        self.socket_path = rospy.get_param(
            "~gpu_socket",
            "/tmp/care_collision_cdf_gpu.sock",
        )

        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.anchor_stale_s = float(
            rospy.get_param("~anchor_stale_s", 0.25)
        )
        self.map_stale_s = float(rospy.get_param("~map_stale_s", 0.50))
        self.worker_timeout_s = float(
            rospy.get_param("~worker_timeout_s", 0.20)
        )
        self.map_resolution = float(
            rospy.get_param("~map_resolution", 0.05)
        )
        self.confidence_threshold = float(
            rospy.get_param("~confidence_threshold", 0.50)
        )
        self.proximity_margin = float(
            rospy.get_param("~proximity_margin", 0.075)
        )
        self.max_pairs_per_step = int(
            rospy.get_param("~max_pairs_per_step", 250)
        )
        self.max_pairs = int(rospy.get_param("~max_pairs", 8000))
        self.zero_band = float(
            rospy.get_param("~signed_zero_band", 0.05)
        )

        if self.rate_hz <= 0.0:
            raise ValueError("~rate must be positive")
        if self.map_resolution <= 0.0:
            raise ValueError("~map_resolution must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("invalid confidence threshold")
        if self.max_pairs_per_step <= 0 or self.max_pairs <= 0:
            raise ValueError("pair limits must be positive")

        out_dir = os.path.dirname(self.output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_jsonl, "w", encoding="utf-8"):
            pass

        self.client = GPUWorkerClient(
            self.socket_path, self.worker_timeout_s
        )

        self.lock = threading.Lock()
        self.latest_anchor = None
        self.latest_anchor_received = rospy.Time(0)

        self.low_map = None
        self.map_total_count = 0
        self.low_map_count = 0
        self.latest_map_stamp = rospy.Time(0)
        self.latest_map_received = rospy.Time(0)
        self.latest_map_index_ms = math.nan

        self.last_processed_anchor_stamp = None
        self.record_index = 0

        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=10
        )
        self.anchor_sub = rospy.Subscriber(
            self.anchor_topic,
            PointCloud2,
            self._anchor_cb,
            queue_size=1,
        )
        self.map_sub = rospy.Subscriber(
            self.map_topic,
            PointCloud2,
            self._map_cb,
            queue_size=1,
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz), self._timer_cb
        )

        rospy.logwarn(
            "[C5.2g] READY rate=%.1fHz anchors=%s map=%s socket=%s "
            "conf<%.2f margin=%.3fm max_pairs_per_step=%d",
            self.rate_hz,
            self.anchor_topic,
            self.map_topic,
            self.socket_path,
            self.confidence_threshold,
            self.proximity_margin,
            self.max_pairs_per_step,
        )

    def _grid_key(self, xyz):
        r = self.map_resolution
        return (
            int(round(float(xyz[0]) / r)),
            int(round(float(xyz[1]) / r)),
            int(round(float(xyz[2]) / r)),
        )

    def _anchor_cb(self, msg):
        with self.lock:
            self.latest_anchor = msg
            self.latest_anchor_received = rospy.Time.now()

    def _map_cb(self, msg):
        t0 = time.perf_counter()
        low = {}
        total = 0

        for row in point_cloud2.read_points(
            msg, field_names=MAP_FIELDS, skip_nans=True
        ):
            x, y, z, confidence, current_visibility = row
            total += 1
            values = (x, y, z, confidence, current_visibility)
            if not all(math.isfinite(float(v)) for v in values):
                continue
            if float(confidence) >= self.confidence_threshold:
                continue
            point = np.asarray([x, y, z], dtype=np.float64)
            low[self._grid_key(point)] = {
                "point": point,
                "confidence": float(confidence),
                "current_visibility": float(current_visibility),
            }

        index_ms = (time.perf_counter() - t0) * 1000.0
        with self.lock:
            self.low_map = low
            self.map_total_count = int(total)
            self.low_map_count = len(low)
            self.latest_map_stamp = msg.header.stamp
            self.latest_map_received = rospy.Time.now()
            self.latest_map_index_ms = float(index_ms)

    def _snapshot(self):
        with self.lock:
            return (
                self.latest_anchor,
                self.latest_anchor_received,
                self.low_map,
                self.map_total_count,
                self.low_map_count,
                self.latest_map_stamp,
                self.latest_map_received,
                self.latest_map_index_ms,
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
            vals = (
                x, y, z, q0, q1, q2, q3, q4, q5, q6, radius
            )
            if not all(math.isfinite(float(v)) for v in vals):
                continue
            if float(radius) <= 0.0:
                continue
            items.append(
                {
                    "center": np.asarray(
                        [x, y, z], dtype=np.float64
                    ),
                    "q": np.asarray(
                        [q0, q1, q2, q3, q4, q5, q6],
                        dtype=np.float64,
                    ),
                    "radius": float(radius),
                    "eval_timestep": int(eval_timestep),
                    "original_timestep": int(original_timestep),
                }
            )
        return items

    def _build_pairs(self, anchors, low_map):
        anchors_by_step = defaultdict(list)
        q_by_step = {}
        eval_by_step = {}

        for anchor in anchors:
            step = int(anchor["original_timestep"])
            if step < 0:
                continue
            anchors_by_step[step].append(anchor)
            if step not in q_by_step:
                q_by_step[step] = anchor["q"]
                eval_by_step[step] = anchor["eval_timestep"]

        raw_by_step = {}
        retained_by_step = {}
        pairs = []
        r = self.map_resolution

        for step in sorted(anchors_by_step):
            candidates = {}
            for anchor in anchors_by_step[step]:
                center = anchor["center"]
                radius = anchor["radius"]
                search_radius = radius + self.proximity_margin
                n = int(math.ceil(search_radius / r))
                ck = self._grid_key(center)

                for dx in range(-n, n + 1):
                    for dy in range(-n, n + 1):
                        for dz in range(-n, n + 1):
                            key = (
                                ck[0] + dx,
                                ck[1] + dy,
                                ck[2] + dz,
                            )
                            voxel = low_map.get(key)
                            if voxel is None:
                                continue
                            clearance = (
                                float(
                                    np.linalg.norm(
                                        voxel["point"] - center
                                    )
                                )
                                - radius
                            )
                            if clearance > self.proximity_margin:
                                continue
                            old = candidates.get(key)
                            if (
                                old is None
                                or clearance
                                < old["approx_body_clearance_m"]
                            ):
                                candidates[key] = {
                                    "point": voxel["point"],
                                    "confidence": voxel["confidence"],
                                    "current_visibility": voxel[
                                        "current_visibility"
                                    ],
                                    "approx_body_clearance_m": clearance,
                                }

            raw_by_step[step] = len(candidates)
            ordered = sorted(
                candidates.values(),
                key=lambda x: (
                    x["approx_body_clearance_m"],
                    x["confidence"],
                ),
            )
            if len(ordered) > self.max_pairs_per_step:
                ordered = ordered[: self.max_pairs_per_step]
            retained_by_step[step] = len(ordered)

            for item in ordered:
                pairs.append(
                    {
                        "point": item["point"],
                        "q": q_by_step[step],
                        "confidence": item["confidence"],
                        "current_visibility": item[
                            "current_visibility"
                        ],
                        "approx_body_clearance_m": item[
                            "approx_body_clearance_m"
                        ],
                        "eval_timestep": int(eval_by_step[step]),
                        "original_timestep": int(step),
                    }
                )

        if len(pairs) > self.max_pairs:
            pairs = sorted(
                pairs,
                key=lambda x: x["approx_body_clearance_m"],
            )[: self.max_pairs]

        return pairs, {
            "active_anchor_step_count": len(anchors_by_step),
            "raw_local_pair_count": int(sum(raw_by_step.values())),
            "retained_before_global_cap": int(
                sum(retained_by_step.values())
            ),
            "raw_by_step": raw_by_step,
            "retained_by_step": retained_by_step,
        }

    def _signed_counts(self, distance):
        d = np.asarray(distance, dtype=np.float64)
        n = int(d.size)
        neg = int(np.sum(d < -self.zero_band))
        zero = int(np.sum(np.abs(d) <= self.zero_band))
        pos = int(np.sum(d > self.zero_band))
        return {
            "negative": neg,
            "near_zero": zero,
            "positive": pos,
            "negative_rate": float(neg) / n if n else 0.0,
            "near_zero_rate": float(zero) / n if n else 0.0,
            "positive_rate": float(pos) / n if n else 0.0,
        }

    def _append(self, record):
        with open(self.output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _publish_summary(self, record):
        d = record.get("distance", {})
        timing = record.get("timing_ms", {})
        msg = String()
        msg.data = (
            "C5_2G_GPU_ONLINE "
            "pairs={} steps={} d_min={} neg_rate={:.4f} "
            "anchor_decode_ms={:.3f} selection_ms={:.3f} "
            "pack_ms={:.3f} ipc_ms={:.3f} "
            "gpu_h2d_ms={:.3f} gpu_infer_ms={:.3f} "
            "gpu_d2h_ms={:.3f} pipeline_ms={:.3f} "
            "map_index_ms={:.3f}"
        ).format(
            record["retained_pair_count"],
            record["active_step_count"],
            "nan" if d.get("min") is None else "{:.5f}".format(d["min"]),
            record["signed_counts"]["negative_rate"],
            timing["anchor_decode_ms"],
            timing["pair_selection_ms"],
            timing["request_pack_ms"],
            timing["ipc_roundtrip_ms"],
            timing["worker_h2d_ms"],
            timing["worker_inference_ms"],
            timing["worker_d2h_ms"],
            timing["online_pipeline_ms"],
            timing["map_index_build_ms"],
        )
        self.summary_pub.publish(msg)

    def _timer_cb(self, _event):
        pipeline_t0 = time.perf_counter()
        (
            anchor_cloud,
            anchor_received,
            low_map,
            map_total_count,
            low_map_count,
            map_stamp,
            map_received,
            map_index_ms,
        ) = self._snapshot()

        if anchor_cloud is None or low_map is None:
            return

        now = rospy.Time.now()
        if (now - anchor_received).to_sec() > self.anchor_stale_s:
            return
        if (now - map_received).to_sec() > self.map_stale_s:
            return

        stamp_key = (
            int(anchor_cloud.header.stamp.secs),
            int(anchor_cloud.header.stamp.nsecs),
            int(anchor_cloud.width),
        )
        if stamp_key == self.last_processed_anchor_stamp:
            return
        self.last_processed_anchor_stamp = stamp_key

        t0 = time.perf_counter()
        anchors = self._decode_anchors(anchor_cloud)
        t1 = time.perf_counter()
        pairs, pair_meta = self._build_pairs(anchors, low_map)
        t2 = time.perf_counter()

        if not pairs:
            return

        if len(pairs) > self.max_pairs:
            rospy.logerr(
                "[C5.2g] pair batch %d exceeds max_pairs=%d",
                len(pairs),
                self.max_pairs,
            )
            return

        try:
            result = self.client.query(pairs)
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0, "[C5.2g] GPU worker query failed: %s", exc
            )
            try:
                self.client.connect()
            except Exception:
                pass
            return

        response_t = time.perf_counter()
        distance = result["distance"]
        gradient = result["gradient"]
        signed = self._signed_counts(distance)

        per_step_min = {}
        per_step_count = defaultdict(int)
        for i, pair in enumerate(pairs):
            step = int(pair["original_timestep"])
            per_step_count[step] += 1
            value = float(distance[i])
            if step not in per_step_min or value < per_step_min[step]:
                per_step_min[step] = value

        min_idx = int(np.argmin(distance))
        min_pair = pairs[min_idx]

        record = {
            "record_index": self.record_index,
            "wall_time": time.time(),
            "anchor_cloud_stamp": float(
                anchor_cloud.header.stamp.to_sec()
            ),
            "map_cloud_stamp": float(map_stamp.to_sec()),
            "anchor_count": len(anchors),
            "map_voxel_count": int(map_total_count),
            "low_confidence_voxel_count": int(low_map_count),
            "active_anchor_step_count": int(
                pair_meta["active_anchor_step_count"]
            ),
            "raw_local_pair_count": int(
                pair_meta["raw_local_pair_count"]
            ),
            "retained_pair_count": len(pairs),
            "active_step_count": len(per_step_count),
            "distance": quantiles(distance.tolist()),
            "signed_counts": signed,
            "gradient_norm": quantiles(
                np.linalg.norm(gradient, axis=1).tolist()
            ),
            "per_step_min_distance": {
                str(k): float(v)
                for k, v in sorted(per_step_min.items())
            },
            "per_step_pair_count": {
                str(k): int(v)
                for k, v in sorted(per_step_count.items())
            },
            "global_min_pair": {
                "point": np.asarray(
                    min_pair["point"], dtype=float
                ).tolist(),
                "q": np.asarray(
                    min_pair["q"], dtype=float
                ).tolist(),
                "confidence": float(min_pair["confidence"]),
                "approx_body_clearance_m": float(
                    min_pair["approx_body_clearance_m"]
                ),
                "original_timestep": int(
                    min_pair["original_timestep"]
                ),
                "distance": float(distance[min_idx]),
                "gradient": gradient[min_idx].tolist(),
            },
            "timing_ms": {
                "anchor_decode_ms": (t1 - t0) * 1000.0,
                "pair_selection_ms": (t2 - t1) * 1000.0,
                "request_pack_ms": result["request_pack_ms"],
                "ipc_roundtrip_ms": result["ipc_roundtrip_ms"],
                "worker_h2d_ms": result["worker_h2d_ms"],
                "worker_inference_ms": result[
                    "worker_inference_ms"
                ],
                "worker_d2h_ms": result["worker_d2h_ms"],
                "worker_total_ms": result["worker_total_ms"],
                "map_index_build_ms": float(map_index_ms),
                "online_pipeline_ms": (
                    response_t - pipeline_t0
                ) * 1000.0,
            },
        }

        self._append(record)
        self._publish_summary(record)

        rospy.loginfo_throttle(
            1.0,
            "[C5.2g] pairs=%d pipeline=%.2fms selection=%.2fms "
            "ipc=%.2fms gpu=%.2fms dmin=%.4f",
            len(pairs),
            record["timing_ms"]["online_pipeline_ms"],
            record["timing_ms"]["pair_selection_ms"],
            record["timing_ms"]["ipc_roundtrip_ms"],
            record["timing_ms"]["worker_inference_ms"],
            record["distance"]["min"],
        )
        self.record_index += 1


def main():
    rospy.init_node("multi_step_forbidden_space_cdf_gpu_shadow")
    OnlineGPUShadow()
    rospy.spin()


if __name__ == "__main__":
    main()

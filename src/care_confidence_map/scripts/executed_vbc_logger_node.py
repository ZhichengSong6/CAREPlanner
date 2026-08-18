#!/usr/bin/env python3
"""Executed visibility-before-contact diagnostic logger.

For one frozen target x*, measure from the actually executed robot motion:
  d_body(t) = min_i ||c_i(q_measured(t)) - x*|| - r_i
The first d_body <= sweep_extra_margin is executed sweep. The first
current_visibility >= visibility_threshold is executed sensor-seen.
The executed VBC margin is t_sweep_exec - t_see_exec.

This node is observational only. It also writes a standalone per-cycle CSV and
summary JSON under the configured output root so controlled A/B result bundles
contain the executed-VBC metric without modifying the main MPC logger.
"""

from __future__ import annotations

import csv
import json
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rospy
import tf2_geometry_msgs
import tf2_ros
import yaml
from care_confidence_map.srv import QueryConfidence, QueryConfidenceRequest
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, String


def _sanitize(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return out or "trial"


def _lambda_tag(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


class ExecutedVBCLogger:
    FIELDS = [
        "ros_time", "t_from_target_s", "target_x", "target_y", "target_z",
        "min_clearance_m", "min_clearance_all_m", "closest_link",
        "closest_sample", "sweep_latched", "sweep_delay_s", "seen_latched",
        "see_delay_s", "executed_vbc_margin_s", "confidence",
        "current_visibility", "inside_map",
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.target_topic = str(rospy.get_param(
            "~target_topic", "/care_planner/active_sensing/target_point"))
        self.body_samples_file = Path(
            rospy.get_param("~body_samples_file", "")).expanduser()
        self.query_service = str(rospy.get_param(
            "~confidence_query_service", "/care_planner/confidence_map/query"))
        self.rate = float(rospy.get_param("~rate", 50.0))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.01))
        self.query_timeout = float(rospy.get_param("~query_timeout", 0.01))
        self.visibility_threshold = float(rospy.get_param("~visibility_threshold", 0.5))
        self.sweep_extra_margin = float(rospy.get_param("~sweep_extra_margin", 0.0))
        self.target_change_tolerance = float(rospy.get_param("~target_change_tolerance", 1e-4))
        self.ignored_links = set(str(v) for v in rospy.get_param(
            "~ignored_links", ["base_link", "link1"]))
        self.trial_label = str(rospy.get_param("~trial_label", "trial"))
        self.lambda_vis = float(rospy.get_param("~lambda_vis", 0.0))
        self.output_root = Path(rospy.get_param(
            "~output_root", "outputs/phase_b2_v2_ab")).expanduser().resolve()

        if self.rate <= 0.0 or self.query_timeout <= 0.0 or self.tf_timeout < 0.0:
            raise ValueError("invalid rate/timeout")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError("~visibility_threshold must be in [0,1]")
        if not self.body_samples_file.is_file():
            raise ValueError(f"~body_samples_file does not exist: {self.body_samples_file}")

        self.samples_by_frame = self._load_samples(self.body_samples_file)
        if not self.samples_by_frame:
            raise ValueError("no executable risk body samples loaded")

        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"executed_vbc_{_sanitize(self.trial_label)}_lambda_{_lambda_tag(self.lambda_vis)}_{stamp}"
        self.csv_path = self.output_root / f"{stem}.csv"
        self.json_path = self.output_root / f"{stem}.json"
        self._csv_file = self.csv_path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDS)
        self._writer.writeheader()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.query_client = rospy.ServiceProxy(self.query_service, QueryConfidence, persistent=False)

        self._target: Optional[PointStamped] = None
        self._target_xyz: Optional[Tuple[float, float, float]] = None
        self._target_time: Optional[float] = None
        self._sweep_time: Optional[float] = None
        self._see_time: Optional[float] = None
        self._min_clearance_all = math.inf
        self._latest_clearance = math.nan
        self._latest_closest_link = "none"
        self._latest_closest_sample = -1
        self._latest_confidence = math.nan
        self._latest_current_visibility = math.nan
        self._latest_inside_map = 0
        self._sequence = 0

        self.summary_pub = rospy.Publisher(
            "/care_planner/trajectory_risk/executed_vbc_summary", String, queue_size=1)
        self.clearance_pub = rospy.Publisher(
            "/care_planner/trajectory_risk/executed_vbc_min_clearance_m", Float32, queue_size=1)
        self.margin_pub = rospy.Publisher(
            "/care_planner/trajectory_risk/executed_vbc_margin_s", Float32, queue_size=1)
        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / self.rate), self._timer_callback)
        rospy.on_shutdown(self._on_shutdown)

        rospy.logwarn(
            "[executed_vbc] logger enabled: target=%s csv=%s json=%s",
            self.target_topic, self.csv_path, self.json_path)

    def _load_samples(self, path: Path) -> Dict[str, List[Tuple[str, int, Tuple[float, float, float], float]]]:
        data = yaml.safe_load(path.read_text())
        out: Dict[str, List[Tuple[str, int, Tuple[float, float, float], float]]] = {}
        for link in data.get("body_sampling", {}).get("links", []):
            link_name = str(link.get("link_name", ""))
            frame = str(link.get("frame", link_name))
            if not bool(link.get("include_for_risk", True)) or link_name in self.ignored_links:
                continue
            for idx, sample in enumerate(link.get("samples", [])):
                center = sample.get("center")
                radius = float(sample.get("radius", 0.0))
                if center is None or len(center) != 3 or radius <= 0.0:
                    continue
                out.setdefault(frame, []).append((
                    link_name, idx,
                    (float(center[0]), float(center[1]), float(center[2])),
                    radius))
        return out

    @staticmethod
    def _xyz(msg: PointStamped) -> Tuple[float, float, float]:
        return float(msg.point.x), float(msg.point.y), float(msg.point.z)

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = self._xyz(msg)
        if not all(math.isfinite(v) for v in xyz):
            return
        now = rospy.Time.now().to_sec()
        with self._lock:
            changed = self._target_xyz is None or self._distance(xyz, self._target_xyz) > self.target_change_tolerance
            if changed:
                self._target_time = now
                self._sweep_time = None
                self._see_time = None
                self._min_clearance_all = math.inf
                self._latest_clearance = math.nan
                rospy.logwarn("[executed_vbc] new target epoch: [%.6f, %.6f, %.6f]", *xyz)
            self._target = msg
            self._target_xyz = xyz

    def _target_in_base(self, target: PointStamped) -> Optional[PointStamped]:
        frame = target.header.frame_id.lstrip("/") or self.base_frame.lstrip("/")
        if frame == self.base_frame.lstrip("/"):
            out = PointStamped()
            out.header.stamp = rospy.Time(0)
            out.header.frame_id = self.base_frame
            out.point = target.point
            return out
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame, target.header.frame_id, rospy.Time(0), rospy.Duration(self.tf_timeout))
            return tf2_geometry_msgs.do_transform_point(target, tf_msg)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "[executed_vbc] target transform unavailable: %s", exc)
            return None

    def _compute_min_clearance(self, target_base: PointStamped) -> Optional[Tuple[float, str, int]]:
        target_xyz = self._xyz(target_base)
        best, best_link, best_sample = math.inf, "none", -1
        for frame, samples in self.samples_by_frame.items():
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame, frame, rospy.Time(0), rospy.Duration(self.tf_timeout))
            except Exception as exc:
                rospy.logwarn_throttle(1.0, "[executed_vbc] missing TF %s -> %s: %s", self.base_frame, frame, exc)
                return None
            for link_name, sample_index, center, radius in samples:
                p = PointStamped()
                p.header.stamp = rospy.Time(0)
                p.header.frame_id = frame
                p.point.x, p.point.y, p.point.z = center
                center_base = self._xyz(tf2_geometry_msgs.do_transform_point(p, tf_msg))
                clearance = self._distance(center_base, target_xyz) - radius
                if clearance < best:
                    best, best_link, best_sample = clearance, link_name, sample_index
        return None if not math.isfinite(best) else (best, best_link, best_sample)

    def _query_visibility(self, target_base: PointStamped):
        try:
            rospy.wait_for_service(self.query_service, timeout=self.query_timeout)
            req = QueryConfidenceRequest()
            req.points = [target_base.point]
            res = self.query_client(req)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(1.0, "[executed_vbc] confidence query unavailable: %s", exc)
            return None
        if len(res.confidence) != 1 or len(res.current_visibility) != 1 or len(res.inside_map) != 1:
            return None
        return float(res.confidence[0]), float(res.current_visibility[0]), int(res.inside_map[0])

    def _summary_payload(self) -> Dict[str, object]:
        target_time = self._target_time
        sweep_delay = None if self._sweep_time is None or target_time is None else self._sweep_time - target_time
        see_delay = None if self._see_time is None or target_time is None else self._see_time - target_time
        margin = None if self._sweep_time is None or self._see_time is None else self._sweep_time - self._see_time
        return {
            "trial_label": self.trial_label,
            "lambda_vis": self.lambda_vis,
            "target_xyz": None if self._target_xyz is None else list(self._target_xyz),
            "sweep_delay_from_target_s": sweep_delay,
            "see_delay_from_target_s": see_delay,
            "executed_vbc_margin_s": margin,
            "seen_before_sweep": None if margin is None else bool(margin >= 0.0),
            "min_clearance_all_m": None if not math.isfinite(self._min_clearance_all) else self._min_clearance_all,
            "latest_clearance_m": None if not math.isfinite(self._latest_clearance) else self._latest_clearance,
            "latest_closest_link": self._latest_closest_link,
            "latest_closest_sample": self._latest_closest_sample,
            "latest_confidence": None if not math.isfinite(self._latest_confidence) else self._latest_confidence,
            "latest_current_visibility": None if not math.isfinite(self._latest_current_visibility) else self._latest_current_visibility,
            "latest_inside_map": self._latest_inside_map,
            "num_rows": self._sequence,
            "csv": str(self.csv_path),
            "summary_json": str(self.json_path),
        }

    def _write_json(self) -> None:
        payload = self._summary_payload()
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.json_path)

    def _timer_callback(self, _event) -> None:
        with self._lock:
            target, target_time = self._target, self._target_time
        if target is None or target_time is None:
            return
        target_base = self._target_in_base(target)
        if target_base is None:
            return
        clearance_info = self._compute_min_clearance(target_base)
        vis_info = self._query_visibility(target_base)
        if clearance_info is None or vis_info is None:
            return

        now = rospy.Time.now().to_sec()
        clearance, closest_link, closest_sample = clearance_info
        confidence, current_visibility, inside_map = vis_info
        with self._lock:
            self._sequence += 1
            self._latest_clearance = clearance
            self._latest_closest_link = closest_link
            self._latest_closest_sample = closest_sample
            self._latest_confidence = confidence
            self._latest_current_visibility = current_visibility
            self._latest_inside_map = inside_map
            self._min_clearance_all = min(self._min_clearance_all, clearance)

            if self._sweep_time is None and clearance <= self.sweep_extra_margin:
                self._sweep_time = now
                rospy.logwarn(
                    "[executed_vbc] EXECUTED SWEEP event: delay=%.6f s clearance=%.6f m link=%s sample=%d",
                    now - target_time, clearance, closest_link, closest_sample)
            if self._see_time is None and inside_map != 0 and current_visibility >= self.visibility_threshold:
                self._see_time = now
                rospy.logwarn(
                    "[executed_vbc] EXECUTED SENSOR-SEEN event: delay=%.6f s confidence=%.6f",
                    now - target_time, confidence)

            sweep_delay = math.nan if self._sweep_time is None else self._sweep_time - target_time
            see_delay = math.nan if self._see_time is None else self._see_time - target_time
            margin = math.nan if self._sweep_time is None or self._see_time is None else self._sweep_time - self._see_time

            self._writer.writerow({
                "ros_time": now,
                "t_from_target_s": now - target_time,
                "target_x": target_base.point.x,
                "target_y": target_base.point.y,
                "target_z": target_base.point.z,
                "min_clearance_m": clearance,
                "min_clearance_all_m": self._min_clearance_all,
                "closest_link": closest_link,
                "closest_sample": closest_sample,
                "sweep_latched": int(self._sweep_time is not None),
                "sweep_delay_s": "" if not math.isfinite(sweep_delay) else sweep_delay,
                "seen_latched": int(self._see_time is not None),
                "see_delay_s": "" if not math.isfinite(see_delay) else see_delay,
                "executed_vbc_margin_s": "" if not math.isfinite(margin) else margin,
                "confidence": confidence,
                "current_visibility": current_visibility,
                "inside_map": inside_map,
            })
            if self._sequence % 10 == 0 or math.isfinite(margin):
                self._write_json()

            msg = String()
            msg.data = (
                f"seq={self._sequence} elapsed={now-target_time:.9f} "
                f"min_clearance={clearance:.9f} min_clearance_all={self._min_clearance_all:.9f} "
                f"closest_link={closest_link} closest_sample={closest_sample} "
                f"sweep_latched={int(self._sweep_time is not None)} "
                f"sweep_delay={'nan' if not math.isfinite(sweep_delay) else f'{sweep_delay:.9f}'} "
                f"seen_latched={int(self._see_time is not None)} "
                f"see_delay={'nan' if not math.isfinite(see_delay) else f'{see_delay:.9f}'} "
                f"exec_margin={'nan' if not math.isfinite(margin) else f'{margin:.9f}'} "
                f"confidence={confidence:.9f} current_visibility={current_visibility:.9f} inside_map={inside_map}")
            self.summary_pub.publish(msg)

            clear_msg = Float32(); clear_msg.data = float(clearance); self.clearance_pub.publish(clear_msg)
            if math.isfinite(margin):
                margin_msg = Float32(); margin_msg.data = float(margin); self.margin_pub.publish(margin_msg)

    def _on_shutdown(self) -> None:
        try:
            with self._lock:
                self._write_json()
                self._csv_file.flush()
                self._csv_file.close()
                rospy.logwarn("[executed_vbc] finalized -> %s", self.json_path)
        except Exception as exc:
            try:
                rospy.logerr("[executed_vbc] finalize failed: %s", exc)
            except Exception:
                pass


def main() -> None:
    rospy.init_node("executed_vbc_logger")
    ExecutedVBCLogger()
    rospy.spin()


if __name__ == "__main__":
    main()

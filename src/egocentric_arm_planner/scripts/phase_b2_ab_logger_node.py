#!/usr/bin/env python3
"""Per-cycle logger for CAREPlanner Phase B.2-v2 controlled A/B trials.

The C++ MPC, the B.2-v2 acquisition adapter, and the NCDF observer already
publish machine-readable String summaries every control/observer cycle.  Their
console output is throttled, however, so short acquisition windows can be missed
when inspecting terminal logs manually.

This node subscribes to the unthrottled summary topics and writes one CSV row
for every MPC summary message.  It also tracks event times and writes an
incrementally updated summary.json so Ctrl-C after a trial still leaves a compact
set of metrics for comparison.

No control messages are published by this node.
"""

from __future__ import annotations

import csv
import json
import math
import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
_RANGE_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")


def _parse_tokens(text: str) -> Dict[str, str]:
    return {key: value for key, value in _TOKEN_RE.findall(text or "")}


def _as_float(value: Optional[str], default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text.endswith("ms"):
        text = text[:-2]
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _as_int(value: Optional[str], default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_range(value: Optional[str]) -> Tuple[float, float]:
    if value is None:
        return math.nan, math.nan
    match = _RANGE_RE.match(str(value).strip())
    if not match:
        return math.nan, math.nan
    return _as_float(match.group(1)), _as_float(match.group(2))


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_delay(event_time: Optional[float], target_time: Optional[float]) -> Optional[float]:
    if event_time is None or target_time is None:
        return None
    return float(event_time - target_time)


def _sanitize_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return text or "trial"


def _lambda_tag(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


class PhaseB2ABLogger:
    CSV_FIELDS = [
        "ros_time",
        "t_from_target_s",
        "target_x",
        "target_y",
        "target_z",
        "mpc_seq",
        "mpc_status",
        "mpc_vis_status",
        "lambda_vis",
        "solve_ms",
        "tracking_inf",
        "command_inf",
        "pred_dev_inf",
        "vis_age",
        "vis_grad_max",
        "base_grad_inf",
        "vis_linear_inf",
        "ref_horizon",
        "adapter_age_s",
        "adapter_seq",
        "acquisition_active",
        "target_seen",
        "inside_map",
        "confidence",
        "current_visibility",
        "adapter_reason",
        "raw_grad_norm_min",
        "raw_grad_norm_max",
        "normalized_grad_norm_min",
        "normalized_grad_norm_max",
        "observer_age_s",
        "observer_seq",
        "f_min",
        "f_max",
        "g_min",
        "g_max",
        "learned_visible_frac",
        "true_visible_frac",
        "sign_agree",
        "observer_grad_norm_min",
        "observer_grad_norm_max",
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.trial_label = str(rospy.get_param("~trial_label", ""))
        self.lambda_vis = float(rospy.get_param("~lambda_vis", 0.0))
        self.output_root = Path(
            rospy.get_param("~output_root", "outputs/phase_b2_v2_ab")
        ).expanduser().resolve()

        self.target_topic = str(
            rospy.get_param(
                "~target_topic", "/care_planner/active_sensing/target_point"
            )
        )
        self.mpc_summary_topic = str(
            rospy.get_param("~mpc_summary_topic", "/velocity_qp_mpc_node/summary")
        )
        self.adapter_summary_topic = str(
            rospy.get_param(
                "~adapter_summary_topic",
                "/phase_b2_visibility_acquisition_adapter/summary",
            )
        )
        self.observer_summary_topic = str(
            rospy.get_param(
                "~observer_summary_topic", "/ncdf_horizon_observer/summary"
            )
        )

        if not math.isfinite(self.lambda_vis) or self.lambda_vis < 0.0:
            raise ValueError("~lambda_vis must be finite and non-negative")

        if not self.trial_label:
            self.trial_label = "baseline" if self.lambda_vis <= 0.0 else "b2_v2"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{_sanitize_label(self.trial_label)}_lambda_{_lambda_tag(self.lambda_vis)}_{stamp}"
        )
        self.run_dir = self.output_root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.run_dir / "per_cycle.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.metadata_path = self.run_dir / "metadata.json"

        self._csv_file = self.csv_path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._writer.writeheader()

        self._target_xyz: Optional[Tuple[float, float, float]] = None
        self._target_time: Optional[float] = None
        self._latest_adapter: Dict[str, str] = {}
        self._latest_adapter_time: Optional[float] = None
        self._latest_observer: Dict[str, str] = {}
        self._latest_observer_time: Optional[float] = None

        self._event_times: Dict[str, Optional[float]] = {
            "acquisition_active_first": None,
            "confidence_seen_first": None,
            "current_visibility_first": None,
            "learned_visible_any_first": None,
            "learned_visible_half_first": None,
            "learned_visible_full_first": None,
            "true_visible_any_first": None,
            "true_visible_half_first": None,
            "true_visible_full_first": None,
            "mpc_vis_used_first": None,
        }

        self._num_rows = 0
        self._vis_status_counts = Counter()
        self._max_tracking_all = 0.0
        self._max_tracking_acquiring = 0.0
        self._max_tracking_post_seen = 0.0
        self._max_pred_dev_all = 0.0
        self._max_pred_dev_acquiring = 0.0
        self._max_pred_dev_post_seen = 0.0
        self._max_vis_linear = 0.0
        self._max_vis_grad = 0.0
        self._max_raw_grad = 0.0
        self._max_normalized_grad = 0.0
        self._min_normalized_grad = math.inf
        self._max_confidence = 0.0
        self._last_tracking = math.nan
        self._last_pred_dev = math.nan
        self._last_mpc_vis_status = "none"

        metadata = {
            "trial_label": self.trial_label,
            "lambda_vis": self.lambda_vis,
            "created_local": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(self.run_dir),
            "topics": {
                "target": self.target_topic,
                "mpc_summary": self.mpc_summary_topic,
                "adapter_summary": self.adapter_summary_topic,
                "observer_summary": self.observer_summary_topic,
            },
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_callback, queue_size=10
        )
        self.adapter_sub = rospy.Subscriber(
            self.adapter_summary_topic, String, self._adapter_callback, queue_size=100
        )
        self.observer_sub = rospy.Subscriber(
            self.observer_summary_topic, String, self._observer_callback, queue_size=100
        )
        self.mpc_sub = rospy.Subscriber(
            self.mpc_summary_topic, String, self._mpc_callback, queue_size=200
        )

        rospy.on_shutdown(self._on_shutdown)
        rospy.logwarn(
            "[phase_b2_ab_logger] RECORDING per-cycle controlled A/B data: %s",
            self.run_dir,
        )

    def _set_event_once(self, name: str, now: float) -> None:
        if self._event_times.get(name) is None:
            self._event_times[name] = now

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
        if not all(math.isfinite(v) for v in xyz):
            return
        now = rospy.Time.now().to_sec()
        with self._lock:
            if self._target_time is None:
                self._target_time = now
                self._target_xyz = xyz
                rospy.logwarn(
                    "[phase_b2_ab_logger] target epoch started: [%.6f, %.6f, %.6f]",
                    *xyz,
                )

    def _adapter_callback(self, msg: String) -> None:
        now = rospy.Time.now().to_sec()
        tokens = _parse_tokens(msg.data)
        with self._lock:
            self._latest_adapter = tokens
            self._latest_adapter_time = now

            active = _as_int(tokens.get("active"), 0) == 1
            seen = _as_int(tokens.get("seen"), 0) == 1
            confidence = _as_float(tokens.get("confidence"))
            current_visibility = _as_float(tokens.get("current_visibility"))
            raw_min, raw_max = _as_range(tokens.get("raw_grad_norm"))
            norm_min, norm_max = _as_range(tokens.get("normalized_grad_norm"))

            if active:
                self._set_event_once("acquisition_active_first", now)
            if seen:
                self._set_event_once("confidence_seen_first", now)
            if _finite(current_visibility) and current_visibility >= 0.5:
                self._set_event_once("current_visibility_first", now)
            if _finite(confidence):
                self._max_confidence = max(self._max_confidence, confidence)
            if _finite(raw_max):
                self._max_raw_grad = max(self._max_raw_grad, raw_max)
            if _finite(norm_max):
                self._max_normalized_grad = max(self._max_normalized_grad, norm_max)
            if _finite(norm_min):
                self._min_normalized_grad = min(self._min_normalized_grad, norm_min)

    def _observer_callback(self, msg: String) -> None:
        now = rospy.Time.now().to_sec()
        tokens = _parse_tokens(msg.data)
        with self._lock:
            self._latest_observer = tokens
            self._latest_observer_time = now

            learned = _as_float(tokens.get("learned_visible_frac"))
            true = _as_float(tokens.get("true_visible_frac"))
            if _finite(learned):
                if learned > 0.0:
                    self._set_event_once("learned_visible_any_first", now)
                if learned >= 0.5:
                    self._set_event_once("learned_visible_half_first", now)
                if learned >= 1.0 - 1e-6:
                    self._set_event_once("learned_visible_full_first", now)
            if _finite(true):
                if true > 0.0:
                    self._set_event_once("true_visible_any_first", now)
                if true >= 0.5:
                    self._set_event_once("true_visible_half_first", now)
                if true >= 1.0 - 1e-6:
                    self._set_event_once("true_visible_full_first", now)

    def _mpc_callback(self, msg: String) -> None:
        now = rospy.Time.now().to_sec()
        mpc = _parse_tokens(msg.data)

        with self._lock:
            adapter = dict(self._latest_adapter)
            observer = dict(self._latest_observer)
            adapter_time = self._latest_adapter_time
            observer_time = self._latest_observer_time
            target_time = self._target_time
            target_xyz = self._target_xyz

            vis_status = mpc.get("vis", "unknown")
            tracking = _as_float(mpc.get("tracking_inf"))
            pred_dev = _as_float(mpc.get("pred_dev_inf"))
            vis_linear = _as_float(mpc.get("vis_linear_inf"))
            vis_grad = _as_float(mpc.get("vis_grad_max"))
            acquiring = _as_int(adapter.get("active"), 0) == 1
            seen = _as_int(adapter.get("seen"), 0) == 1

            self._num_rows += 1
            self._vis_status_counts[vis_status] += 1
            self._last_mpc_vis_status = vis_status
            if vis_status == "used":
                self._set_event_once("mpc_vis_used_first", now)

            if _finite(tracking):
                self._last_tracking = tracking
                self._max_tracking_all = max(self._max_tracking_all, tracking)
                if acquiring:
                    self._max_tracking_acquiring = max(
                        self._max_tracking_acquiring, tracking
                    )
                if seen:
                    self._max_tracking_post_seen = max(
                        self._max_tracking_post_seen, tracking
                    )
            if _finite(pred_dev):
                self._last_pred_dev = pred_dev
                self._max_pred_dev_all = max(self._max_pred_dev_all, pred_dev)
                if acquiring:
                    self._max_pred_dev_acquiring = max(
                        self._max_pred_dev_acquiring, pred_dev
                    )
                if seen:
                    self._max_pred_dev_post_seen = max(
                        self._max_pred_dev_post_seen, pred_dev
                    )
            if _finite(vis_linear):
                self._max_vis_linear = max(self._max_vis_linear, vis_linear)
            if _finite(vis_grad):
                self._max_vis_grad = max(self._max_vis_grad, vis_grad)

            raw_min, raw_max = _as_range(adapter.get("raw_grad_norm"))
            norm_min, norm_max = _as_range(adapter.get("normalized_grad_norm"))
            f_min, f_max = _as_range(observer.get("f"))
            g_min, g_max = _as_range(observer.get("g"))
            obs_grad_min, obs_grad_max = _as_range(observer.get("grad_norm"))

            row = {
                "ros_time": now,
                "t_from_target_s": "" if target_time is None else now - target_time,
                "target_x": "" if target_xyz is None else target_xyz[0],
                "target_y": "" if target_xyz is None else target_xyz[1],
                "target_z": "" if target_xyz is None else target_xyz[2],
                "mpc_seq": _as_int(mpc.get("seq")),
                "mpc_status": mpc.get("status", ""),
                "mpc_vis_status": vis_status,
                "lambda_vis": _as_float(mpc.get("lambda_vis"), self.lambda_vis),
                "solve_ms": _as_float(mpc.get("solve")),
                "tracking_inf": tracking,
                "command_inf": _as_float(mpc.get("command_inf")),
                "pred_dev_inf": pred_dev,
                "vis_age": _as_float(mpc.get("vis_age")),
                "vis_grad_max": vis_grad,
                "base_grad_inf": _as_float(mpc.get("base_grad_inf")),
                "vis_linear_inf": vis_linear,
                "ref_horizon": _as_float(mpc.get("ref_horizon")),
                "adapter_age_s": "" if adapter_time is None else now - adapter_time,
                "adapter_seq": _as_int(adapter.get("seq")),
                "acquisition_active": int(acquiring),
                "target_seen": int(seen),
                "inside_map": _as_int(adapter.get("inside_map")),
                "confidence": _as_float(adapter.get("confidence")),
                "current_visibility": _as_float(adapter.get("current_visibility")),
                "adapter_reason": adapter.get("reason", ""),
                "raw_grad_norm_min": raw_min,
                "raw_grad_norm_max": raw_max,
                "normalized_grad_norm_min": norm_min,
                "normalized_grad_norm_max": norm_max,
                "observer_age_s": "" if observer_time is None else now - observer_time,
                "observer_seq": _as_int(observer.get("seq")),
                "f_min": f_min,
                "f_max": f_max,
                "g_min": g_min,
                "g_max": g_max,
                "learned_visible_frac": _as_float(observer.get("learned_visible_frac")),
                "true_visible_frac": _as_float(observer.get("true_visible_frac")),
                "sign_agree": _as_float(observer.get("sign_agree")),
                "observer_grad_norm_min": obs_grad_min,
                "observer_grad_norm_max": obs_grad_max,
            }
            self._writer.writerow(row)

            if self._num_rows % 10 == 0:
                self._write_summary_locked(final=False)

    def _summary_dict_locked(self, final: bool) -> Dict[str, object]:
        target_time = self._target_time
        events = {
            key.replace("_first", "_delay_from_target_s"): _safe_delay(value, target_time)
            for key, value in self._event_times.items()
        }
        acquisition_start = self._event_times["acquisition_active_first"]
        seen_time = self._event_times["confidence_seen_first"]
        acquisition_to_seen = None
        if acquisition_start is not None and seen_time is not None:
            acquisition_to_seen = seen_time - acquisition_start

        return {
            "finalized": bool(final),
            "trial_label": self.trial_label,
            "lambda_vis": self.lambda_vis,
            "run_dir": str(self.run_dir),
            "target_xyz": None if self._target_xyz is None else list(self._target_xyz),
            "num_mpc_rows": self._num_rows,
            "mpc_vis_status_counts": dict(self._vis_status_counts),
            "events": events,
            "acquisition_active_to_seen_s": acquisition_to_seen,
            "metrics": {
                "max_tracking_inf_all": self._max_tracking_all,
                "max_tracking_inf_during_acquisition": self._max_tracking_acquiring,
                "max_tracking_inf_post_seen": self._max_tracking_post_seen,
                "final_tracking_inf": self._last_tracking,
                "max_pred_dev_inf_all": self._max_pred_dev_all,
                "max_pred_dev_inf_during_acquisition": self._max_pred_dev_acquiring,
                "max_pred_dev_inf_post_seen": self._max_pred_dev_post_seen,
                "final_pred_dev_inf": self._last_pred_dev,
                "max_vis_linear_inf": self._max_vis_linear,
                "max_vis_grad_max": self._max_vis_grad,
                "max_raw_grad_norm": self._max_raw_grad,
                "min_normalized_grad_norm": None
                if not math.isfinite(self._min_normalized_grad)
                else self._min_normalized_grad,
                "max_normalized_grad_norm": self._max_normalized_grad,
                "max_confidence": self._max_confidence,
                "last_mpc_vis_status": self._last_mpc_vis_status,
            },
            "files": {
                "per_cycle_csv": str(self.csv_path),
                "metadata_json": str(self.metadata_path),
                "summary_json": str(self.summary_path),
            },
        }

    def _write_summary_locked(self, final: bool) -> None:
        payload = self._summary_dict_locked(final=final)
        tmp_path = self.summary_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self.summary_path)

    def _on_shutdown(self) -> None:
        try:
            with self._lock:
                self._write_summary_locked(final=True)
                self._csv_file.flush()
                self._csv_file.close()
                rospy.logwarn(
                    "[phase_b2_ab_logger] finalized %d MPC rows -> %s",
                    self._num_rows,
                    self.run_dir,
                )
        except Exception as exc:  # pragma: no cover - shutdown safety
            try:
                rospy.logerr("[phase_b2_ab_logger] finalize failed: %s", exc)
            except Exception:
                pass


def main() -> None:
    rospy.init_node("phase_b2_ab_logger")
    PhaseB2ABLogger()
    rospy.spin()


if __name__ == "__main__":
    main()

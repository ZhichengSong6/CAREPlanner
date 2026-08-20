#!/usr/bin/env python3
"""Per-cycle logger for CAREPlanner VBC waypoint/recovery experiments."""

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


def _tokens(text: str) -> Dict[str, str]:
    return {k: v for k, v in _TOKEN_RE.findall(text or "")}


def _float(value, default=math.nan):
    try:
        text = str(value)
        if text.endswith("ms"):
            text = text[:-2]
        return float(text)
    except Exception:
        return default


def _int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def _sanitize(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return out or "trial"


class VbcWaypointLogger:
    FIELDS = [
        "ros_time", "t_from_target_s", "target_x", "target_y", "target_z",
        "mpc_seq", "mpc_status", "control_mode", "recovery_active",
        "recovery_hold", "solve_ms", "tracking_inf", "command_inf",
        "pred_dev_inf", "vbc_wp_status", "waypoint_weight", "waypoint_age",
        "deadline_remaining", "waypoint_k", "waypoint_grid_time",
        "waypoint_nominal_error_inf", "waypoint_pred_error_inf",
        "waypoint_linear_inf", "waypoint_hessian_inf", "base_grad_inf",
        "ref_horizon", "generator_active", "generator_seen",
        "generator_ready", "generator_confidence", "generator_current_visibility",
        "generator_deadline_remaining", "generator_reason",
        "predictive_enabled", "predictive_active", "predictive_triggered",
        "predictive_trigger_count_total", "predictive_status",
        "predictive_miss_streak", "predictive_required_streak",
        "predictive_error_threshold_inf", "predictive_require_stall",
        "predictive_pred_error_inf", "predictive_pred_improvement_inf",
        "predictive_physical_deadline_remaining",
        "predictive_effective_deadline_remaining", "predictive_trigger_lead_s",
        "predictive_last_trigger_lead_s", "predictive_last_trigger_error_inf",
        "predictive_waypoint_k",
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.trial_label = str(rospy.get_param("~trial_label", "vbc_waypoint"))
        self.waypoint_weight = float(rospy.get_param("~waypoint_weight", 100.0))
        self.output_root = Path(rospy.get_param(
            "~output_root", "outputs/phase_b2_vbc_waypoint")).expanduser().resolve()
        self.target_topic = str(rospy.get_param(
            "~target_topic", "/care_planner/active_sensing/target_point"))
        self.mpc_summary_topic = str(rospy.get_param(
            "~mpc_summary_topic", "/velocity_qp_mpc_waypoint_node/summary"))
        self.generator_summary_topic = str(rospy.get_param(
            "~generator_summary_topic",
            "/care_planner/active_sensing/visibility_waypoint_summary"))
        self.predictive_summary_topic = str(rospy.get_param(
            "~predictive_summary_topic",
            "/care_planner/execution/predictive_recovery_summary"))

        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{_sanitize(self.trial_label)}_weight_{self.waypoint_weight:.1f}_{stamp}"
        self.run_dir = self.output_root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.run_dir / "per_cycle.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.metadata_path = self.run_dir / "metadata.json"
        self._csv_file = self.csv_path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDS)
        self._writer.writeheader()

        self._target_xyz: Optional[Tuple[float, float, float]] = None
        self._target_time: Optional[float] = None
        self._latest_generator: Dict[str, str] = {}
        self._latest_predictive: Dict[str, str] = {}
        self._num_rows = 0
        self._status_counts = Counter()
        self._control_mode_counts = Counter()
        self._max_tracking = 0.0
        self._max_pred_dev = 0.0
        self._max_command = 0.0
        self._max_solve_ms = 0.0
        self._sum_solve_ms = 0.0
        self._first_waypoint_used: Optional[float] = None
        self._first_recovery: Optional[float] = None
        self._first_recovery_hold: Optional[float] = None
        self._first_waypoint_inactive_after_used: Optional[float] = None
        self._min_waypoint_pred_error = math.inf
        self._last_waypoint_pred_error = math.nan
        self._max_predictive_trigger_count = 0
        self._last_predictive_trigger_lead = math.nan
        self._last_predictive_trigger_error = math.nan
        self._first_predictive_trigger_observed: Optional[float] = None

        self.metadata_path.write_text(json.dumps({
            "trial_label": self.trial_label,
            "waypoint_weight": self.waypoint_weight,
            "target_topic": self.target_topic,
            "mpc_summary_topic": self.mpc_summary_topic,
            "generator_summary_topic": self.generator_summary_topic,
            "predictive_summary_topic": self.predictive_summary_topic,
            "csv": str(self.csv_path),
            "summary": str(self.summary_path),
        }, indent=2, sort_keys=True))

        self.target_sub = rospy.Subscriber(
            self.target_topic, PointStamped, self._target_cb, queue_size=10)
        self.generator_sub = rospy.Subscriber(
            self.generator_summary_topic, String, self._generator_cb, queue_size=20)
        self.predictive_sub = rospy.Subscriber(
            self.predictive_summary_topic, String, self._predictive_cb, queue_size=50)
        self.mpc_sub = rospy.Subscriber(
            self.mpc_summary_topic, String, self._mpc_cb, queue_size=100)
        rospy.on_shutdown(self._shutdown)

        rospy.logwarn("[vbc_waypoint_logger] logging -> %s", self.run_dir)

    def _target_cb(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
        now = rospy.Time.now().to_sec()
        with self._lock:
            if self._target_xyz is None:
                self._target_xyz = xyz
                self._target_time = now

    def _generator_cb(self, msg: String) -> None:
        with self._lock:
            self._latest_generator = _tokens(msg.data if msg else "")

    def _predictive_cb(self, msg: String) -> None:
        now = rospy.Time.now().to_sec()
        with self._lock:
            p = _tokens(msg.data if msg else "")
            self._latest_predictive = p
            count = max(0, _int(p.get("trigger_count_total"), 0))
            if count > self._max_predictive_trigger_count:
                self._max_predictive_trigger_count = count
                if self._first_predictive_trigger_observed is None:
                    self._first_predictive_trigger_observed = now
            lead = _float(p.get("last_trigger_lead_s"))
            err = _float(p.get("last_trigger_error_inf"))
            if math.isfinite(lead):
                self._last_predictive_trigger_lead = lead
            if math.isfinite(err):
                self._last_predictive_trigger_error = err

    def _mpc_cb(self, msg: String) -> None:
        if msg is None:
            return
        now = rospy.Time.now().to_sec()
        m = _tokens(msg.data)
        with self._lock:
            g = dict(self._latest_generator)
            p = dict(self._latest_predictive)
            xyz = self._target_xyz
            target_time = self._target_time

            status = m.get("vbc_wp", "unknown")
            control_mode = m.get("control_mode", "unknown")
            solve = _float(m.get("solve"))
            tracking = _float(m.get("tracking_inf"))
            command = _float(m.get("command_inf"))
            pred_dev = _float(m.get("pred_dev_inf"))
            pred_error = _float(m.get("waypoint_pred_error_inf"))

            self._num_rows += 1
            self._status_counts[status] += 1
            self._control_mode_counts[control_mode] += 1
            if math.isfinite(solve):
                self._max_solve_ms = max(self._max_solve_ms, solve)
                self._sum_solve_ms += solve
            if math.isfinite(tracking):
                self._max_tracking = max(self._max_tracking, tracking)
            if math.isfinite(command):
                self._max_command = max(self._max_command, command)
            if math.isfinite(pred_dev):
                self._max_pred_dev = max(self._max_pred_dev, pred_dev)
            if math.isfinite(pred_error):
                self._min_waypoint_pred_error = min(self._min_waypoint_pred_error, pred_error)
                self._last_waypoint_pred_error = pred_error

            if status in ("used", "overdue_used") and self._first_waypoint_used is None:
                self._first_waypoint_used = now
            if control_mode == "recovery" and self._first_recovery is None:
                self._first_recovery = now
            if control_mode == "recovery_hold" and self._first_recovery_hold is None:
                self._first_recovery_hold = now
            if (status == "inactive" and self._first_waypoint_used is not None
                    and self._first_waypoint_inactive_after_used is None):
                self._first_waypoint_inactive_after_used = now

            row = {
                "ros_time": now,
                "t_from_target_s": "" if target_time is None else now - target_time,
                "target_x": "" if xyz is None else xyz[0],
                "target_y": "" if xyz is None else xyz[1],
                "target_z": "" if xyz is None else xyz[2],
                "mpc_seq": _int(m.get("seq")),
                "mpc_status": m.get("status", ""),
                "control_mode": control_mode,
                "recovery_active": _int(m.get("recovery_active")),
                "recovery_hold": _int(m.get("recovery_hold")),
                "solve_ms": solve,
                "tracking_inf": tracking,
                "command_inf": command,
                "pred_dev_inf": pred_dev,
                "vbc_wp_status": status,
                "waypoint_weight": _float(m.get("waypoint_weight")),
                "waypoint_age": _float(m.get("waypoint_age")),
                "deadline_remaining": _float(m.get("deadline_remaining")),
                "waypoint_k": _int(m.get("waypoint_k")),
                "waypoint_grid_time": _float(m.get("waypoint_grid_time")),
                "waypoint_nominal_error_inf": _float(m.get("waypoint_nominal_error_inf")),
                "waypoint_pred_error_inf": pred_error,
                "waypoint_linear_inf": _float(m.get("waypoint_linear_inf")),
                "waypoint_hessian_inf": _float(m.get("waypoint_hessian_inf")),
                "base_grad_inf": _float(m.get("base_grad_inf")),
                "ref_horizon": _float(m.get("ref_horizon")),
                "generator_active": _int(g.get("active")),
                "generator_seen": _int(g.get("seen")),
                "generator_ready": _int(g.get("ready")),
                "generator_confidence": _float(g.get("confidence")),
                "generator_current_visibility": _float(g.get("current_visibility")),
                "generator_deadline_remaining": _float(g.get("deadline_remaining")),
                "generator_reason": g.get("reason", ""),
                "predictive_enabled": _int(p.get("enabled")),
                "predictive_active": _int(p.get("active")),
                "predictive_triggered": _int(p.get("triggered")),
                "predictive_trigger_count_total": _int(p.get("trigger_count_total"), 0),
                "predictive_status": p.get("status", ""),
                "predictive_miss_streak": _int(p.get("miss_streak")),
                "predictive_required_streak": _int(p.get("required_streak")),
                "predictive_error_threshold_inf": _float(p.get("error_threshold_inf")),
                "predictive_require_stall": _int(p.get("require_stall")),
                "predictive_pred_error_inf": _float(p.get("pred_error_inf")),
                "predictive_pred_improvement_inf": _float(p.get("pred_improvement_inf")),
                "predictive_physical_deadline_remaining": _float(p.get("physical_deadline_remaining")),
                "predictive_effective_deadline_remaining": _float(p.get("effective_deadline_remaining")),
                "predictive_trigger_lead_s": _float(p.get("trigger_lead_s")),
                "predictive_last_trigger_lead_s": _float(p.get("last_trigger_lead_s")),
                "predictive_last_trigger_error_inf": _float(p.get("last_trigger_error_inf")),
                "predictive_waypoint_k": _int(p.get("waypoint_k")),
            }
            self._writer.writerow(row)
            if self._num_rows % 5 == 0:
                self._write_summary()

    def _delay(self, stamp: Optional[float]):
        if stamp is None or self._target_time is None:
            return None
        return stamp - self._target_time

    def _write_summary(self) -> None:
        payload = {
            "trial_label": self.trial_label,
            "waypoint_weight": self.waypoint_weight,
            "target_xyz": None if self._target_xyz is None else list(self._target_xyz),
            "num_rows": self._num_rows,
            "waypoint_status_counts": dict(self._status_counts),
            "control_mode_counts": dict(self._control_mode_counts),
            "first_waypoint_used_delay_from_target_s": self._delay(self._first_waypoint_used),
            "first_recovery_delay_from_target_s": self._delay(self._first_recovery),
            "first_recovery_hold_delay_from_target_s": self._delay(self._first_recovery_hold),
            "first_inactive_after_used_delay_from_target_s": self._delay(
                self._first_waypoint_inactive_after_used),
            "predictive_trigger_count_total": self._max_predictive_trigger_count,
            "first_predictive_trigger_observed_delay_from_target_s": self._delay(
                self._first_predictive_trigger_observed),
            "last_predictive_trigger_lead_s": None
                if not math.isfinite(self._last_predictive_trigger_lead)
                else self._last_predictive_trigger_lead,
            "last_predictive_trigger_error_inf": None
                if not math.isfinite(self._last_predictive_trigger_error)
                else self._last_predictive_trigger_error,
            "max_tracking_inf": self._max_tracking,
            "max_pred_dev_inf": self._max_pred_dev,
            "max_command_inf": self._max_command,
            "solve_ms_mean": None if self._num_rows == 0 else self._sum_solve_ms / self._num_rows,
            "solve_ms_max": self._max_solve_ms,
            "min_waypoint_pred_error_inf": None
                if not math.isfinite(self._min_waypoint_pred_error)
                else self._min_waypoint_pred_error,
            "last_waypoint_pred_error_inf": None
                if not math.isfinite(self._last_waypoint_pred_error)
                else self._last_waypoint_pred_error,
            "csv": str(self.csv_path),
        }
        tmp = self.summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.summary_path)

    def _shutdown(self) -> None:
        try:
            with self._lock:
                self._write_summary()
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass


def main() -> None:
    rospy.init_node("phase_b2_vbc_waypoint_logger")
    VbcWaypointLogger()
    rospy.spin()


if __name__ == "__main__":
    main()

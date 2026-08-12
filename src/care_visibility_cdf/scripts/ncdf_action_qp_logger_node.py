#!/usr/bin/env python3
"""Independent logger for Stage-II NCDF action-QP dry-run results.

This node never computes or modifies actions. It only subscribes to the debug
outputs of /ncdf_action_qp_dryrun plus the current target/joint state, records
unique cases to CSV, and prints running Stage-II statistics.
"""

from __future__ import annotations

import copy
import csv
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray, String


PACKAGE_DIR = Path(rospkg.RosPack().get_path("care_visibility_cdf")).resolve()
DEFAULT_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]


class Stage2ActionQPLogger:
    def __init__(self) -> None:
        self.joint_names = [
            str(v) for v in rospy.get_param("~joint_names", DEFAULT_JOINT_NAMES)
        ]
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("~joint_names must contain exactly 7 unique joints")

        self.target_topic = str(
            rospy.get_param("~target_topic", "/care_planner/active_sensing/target_point")
        )
        self.joint_state_topic = str(
            rospy.get_param("~joint_state_topic", "/care_arm/joint_states")
        )
        self.stage2_ns = str(rospy.get_param("~stage2_ns", "/ncdf_action_qp_dryrun"))
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.25))

        self.unique_target_threshold_m = float(
            rospy.get_param("~unique_target_threshold_m", 0.01)
        )
        self.unique_q_threshold_rad = float(
            rospy.get_param("~unique_q_threshold_rad", 0.02)
        )
        self.unique_action_threshold_rad_s = float(
            rospy.get_param("~unique_action_threshold_rad_s", 0.05)
        )

        stamp = time.strftime("%Y%m%d_%H%M%S")
        default_path = PACKAGE_DIR / "outputs" / f"ncdf_stage2_action_qp_{stamp}.csv"
        self.log_path = Path(
            rospy.get_param("~log_path", str(default_path))
        ).expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._target: Optional[PointStamped] = None
        self._target_time: Optional[rospy.Time] = None
        self._joint_state: Optional[JointState] = None
        self._joint_time: Optional[rospy.Time] = None
        self._active = False
        self._active_time: Optional[rospy.Time] = None

        self._gradient: Optional[np.ndarray] = None
        self._u_nom: Optional[np.ndarray] = None
        self._u_corr: Optional[np.ndarray] = None
        self._du: Optional[np.ndarray] = None
        self._q_nom_next: Optional[np.ndarray] = None
        self._q_corr_next: Optional[np.ndarray] = None

        self._rate_nom: Optional[float] = None
        self._rate_corr: Optional[float] = None
        self._drate: Optional[float] = None
        self._delta_f: Optional[float] = None
        self._delta_g: Optional[float] = None
        self._qp_ms: Optional[float] = None
        self._ncdf_ms: Optional[float] = None
        self._oracle_ms: Optional[float] = None

        self._last_x: Optional[np.ndarray] = None
        self._last_q: Optional[np.ndarray] = None
        self._last_u_nom: Optional[np.ndarray] = None
        self._case_id = 0
        self._results: List[Dict[str, float]] = []

        self._file = self.log_path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._file.flush()

        rospy.Subscriber(self.target_topic, PointStamped, self._target_cb, queue_size=1)
        rospy.Subscriber(self.joint_state_topic, JointState, self._joint_cb, queue_size=1)
        rospy.Subscriber(self._topic("active"), Bool, self._active_cb, queue_size=1)
        rospy.Subscriber(self._topic("gradient_q"), Float64MultiArray, self._grad_cb, queue_size=1)
        rospy.Subscriber(self._topic("nominal_action"), Float64MultiArray, self._u_nom_cb, queue_size=1)
        rospy.Subscriber(self._topic("corrected_action"), Float64MultiArray, self._u_corr_cb, queue_size=1)
        rospy.Subscriber(self._topic("delta_action"), Float64MultiArray, self._du_cb, queue_size=1)
        rospy.Subscriber(self._topic("predicted_nominal_joint_state"), JointState, self._q_nom_cb, queue_size=1)
        rospy.Subscriber(self._topic("predicted_corrected_joint_state"), JointState, self._q_corr_cb, queue_size=1)
        rospy.Subscriber(self._topic("visibility_rate_nominal"), Float32, self._rate_nom_cb, queue_size=1)
        rospy.Subscriber(self._topic("visibility_rate_corrected"), Float32, self._rate_corr_cb, queue_size=1)
        rospy.Subscriber(self._topic("delta_visibility_rate"), Float32, self._drate_cb, queue_size=1)
        rospy.Subscriber(self._topic("delta_f"), Float32, self._df_cb, queue_size=1)
        rospy.Subscriber(self._topic("delta_g"), Float32, self._dg_cb, queue_size=1)
        rospy.Subscriber(self._topic("qp_time_ms"), Float32, self._qp_cb, queue_size=1)
        rospy.Subscriber(self._topic("ncdf_time_ms"), Float32, self._ncdf_cb, queue_size=1)
        rospy.Subscriber(self._topic("oracle_time_ms"), Float32, self._oracle_cb, queue_size=1)
        rospy.Subscriber(self._topic("summary"), String, self._summary_cb, queue_size=1)

        rospy.on_shutdown(self._shutdown)
        rospy.logwarn("[stage2_logger] LOGGER ONLY: does not modify or publish robot actions.")
        rospy.loginfo("[stage2_logger] Stage-II namespace: %s", self.stage2_ns)
        rospy.loginfo(
            "[stage2_logger] unique if dx>=%.3fm OR dq>=%.3frad OR du_nom>=%.3frad/s",
            self.unique_target_threshold_m,
            self.unique_q_threshold_rad,
            self.unique_action_threshold_rad_s,
        )
        rospy.loginfo("[stage2_logger] CSV: %s", self.log_path)

    def _topic(self, suffix: str) -> str:
        return self.stage2_ns.rstrip("/") + "/" + suffix

    def _fieldnames(self) -> List[str]:
        fields = [
            "case_id", "ros_time", "wall_time_unix",
            "target_x", "target_y", "target_z",
            "grad_norm",
            "rate_nom", "rate_corr", "delta_rate",
            "f_nom_next", "f_corr_next", "delta_f",
            "g_nom_next", "g_corr_next", "delta_g",
            "learned_up", "oracle_up", "f_up_g_down", "neg_to_pos",
            "active_bounds", "accel_ok", "accel_ratio", "nom_accel_ratio",
            "nom_vel_clip", "qpert_l2",
            "ncdf_time_ms", "qp_time_ms", "oracle_time_ms",
        ]
        for prefix in ("q_current", "dq_current", "q_nom_next", "q_corr_next", "grad", "u_nom", "du", "u_corr"):
            fields.extend([f"{prefix}_{name}" for name in self.joint_names])
        return fields

    @staticmethod
    def _vec7(msg: Float64MultiArray) -> Optional[np.ndarray]:
        arr = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if arr.shape != (7,) or not np.all(np.isfinite(arr)):
            return None
        return arr

    def _joint_vectors(self, msg: JointState) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        idx = {name: i for i, name in enumerate(msg.name)}
        if any(name not in idx for name in self.joint_names):
            return None
        q = np.zeros(7, dtype=np.float64)
        dq = np.zeros(7, dtype=np.float64)
        for i, name in enumerate(self.joint_names):
            j = idx[name]
            if j >= len(msg.position):
                return None
            q[i] = float(msg.position[j])
            dq[i] = float(msg.velocity[j]) if j < len(msg.velocity) else 0.0
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            return None
        return q, dq

    def _joint_q(self, msg: JointState) -> Optional[np.ndarray]:
        result = self._joint_vectors(msg)
        return None if result is None else result[0]

    def _target_cb(self, msg): self._target, self._target_time = copy.deepcopy(msg), rospy.Time.now()
    def _joint_cb(self, msg): self._joint_state, self._joint_time = copy.deepcopy(msg), rospy.Time.now()
    def _active_cb(self, msg): self._active, self._active_time = bool(msg.data), rospy.Time.now()
    def _grad_cb(self, msg): self._gradient = self._vec7(msg)
    def _u_nom_cb(self, msg): self._u_nom = self._vec7(msg)
    def _u_corr_cb(self, msg): self._u_corr = self._vec7(msg)
    def _du_cb(self, msg): self._du = self._vec7(msg)
    def _q_nom_cb(self, msg): self._q_nom_next = self._joint_q(msg)
    def _q_corr_cb(self, msg): self._q_corr_next = self._joint_q(msg)
    def _rate_nom_cb(self, msg): self._rate_nom = float(msg.data)
    def _rate_corr_cb(self, msg): self._rate_corr = float(msg.data)
    def _drate_cb(self, msg): self._drate = float(msg.data)
    def _df_cb(self, msg): self._delta_f = float(msg.data)
    def _dg_cb(self, msg): self._delta_g = float(msg.data)
    def _qp_cb(self, msg): self._qp_ms = float(msg.data)
    def _ncdf_cb(self, msg): self._ncdf_ms = float(msg.data)
    def _oracle_cb(self, msg): self._oracle_ms = float(msg.data)

    @staticmethod
    def _extract_float(text: str, key: str) -> Optional[float]:
        m = re.search(r"(?:^|\s)" + re.escape(key) + r"=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
        return None if m is None else float(m.group(1))

    @staticmethod
    def _extract_pair(text: str, key: str) -> Tuple[Optional[float], Optional[float]]:
        pat = (
            r"(?:^|\s)" + re.escape(key) +
            r"=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"->([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        )
        m = re.search(pat, text)
        if m is None:
            return None, None
        return float(m.group(1)), float(m.group(2))

    def _fresh(self, now: rospy.Time, stamp: Optional[rospy.Time]) -> bool:
        if stamp is None:
            return False
        return 0.0 <= (now - stamp).to_sec() <= self.input_timeout

    def _is_unique(self, x: np.ndarray, q: np.ndarray, u_nom: np.ndarray):
        if self._last_x is None:
            return True, math.inf, math.inf, math.inf
        dx = float(np.linalg.norm(x - self._last_x))
        dq = float(np.linalg.norm(q - self._last_q))
        du = float(np.linalg.norm(u_nom - self._last_u_nom))
        unique = (
            dx >= self.unique_target_threshold_m
            or dq >= self.unique_q_threshold_rad
            or du >= self.unique_action_threshold_rad_s
        )
        return unique, dx, dq, du

    def _summary_cb(self, msg: String) -> None:
        now = rospy.Time.now()
        if not self._active or not self._fresh(now, self._active_time):
            return
        if self._target is None or not self._fresh(now, self._target_time):
            return
        if self._joint_state is None or not self._fresh(now, self._joint_time):
            return

        required = [
            self._gradient, self._u_nom, self._u_corr, self._du,
            self._q_nom_next, self._q_corr_next,
        ]
        if any(v is None for v in required):
            return
        if any(v is None for v in [self._rate_nom, self._rate_corr, self._drate, self._delta_f, self._delta_g]):
            return

        measured = self._joint_vectors(self._joint_state)
        if measured is None:
            return
        q_current, dq_current = measured
        x = np.asarray([
            self._target.point.x, self._target.point.y, self._target.point.z
        ], dtype=np.float64)

        unique, dx, dq, du_nom_change = self._is_unique(x, q_current, self._u_nom)
        if not unique:
            return

        text = msg.data
        f_nom, f_corr = self._extract_pair(text, "f_next")
        g_nom, g_corr = self._extract_pair(text, "g_next")
        grad_norm = self._extract_float(text, "grad_norm")
        active_bounds = self._extract_float(text, "active_bounds")
        accel_ok = self._extract_float(text, "accel_ok")
        accel_ratio = self._extract_float(text, "accel_ratio")
        nom_accel_ratio = self._extract_float(text, "nom_accel_ratio")
        nom_vel_clip = self._extract_float(text, "nom_vel_clip")
        qpert_l2 = self._extract_float(text, "qpert_l2")

        learned_up = int(self._delta_f > 1e-8)
        oracle_up = int(self._delta_g > 1e-8)
        f_up_g_down = int(self._delta_f > 1e-8 and self._delta_g < -1e-8)
        neg_to_pos = int(self._rate_nom < 0.0 and self._rate_corr > 0.0)

        self._case_id += 1
        row: Dict[str, object] = {
            "case_id": self._case_id,
            "ros_time": float(now.to_sec()),
            "wall_time_unix": float(time.time()),
            "target_x": float(x[0]), "target_y": float(x[1]), "target_z": float(x[2]),
            "grad_norm": grad_norm,
            "rate_nom": self._rate_nom, "rate_corr": self._rate_corr,
            "delta_rate": self._drate,
            "f_nom_next": f_nom, "f_corr_next": f_corr, "delta_f": self._delta_f,
            "g_nom_next": g_nom, "g_corr_next": g_corr, "delta_g": self._delta_g,
            "learned_up": learned_up, "oracle_up": oracle_up,
            "f_up_g_down": f_up_g_down, "neg_to_pos": neg_to_pos,
            "active_bounds": active_bounds, "accel_ok": accel_ok,
            "accel_ratio": accel_ratio, "nom_accel_ratio": nom_accel_ratio,
            "nom_vel_clip": nom_vel_clip, "qpert_l2": qpert_l2,
            "ncdf_time_ms": self._ncdf_ms, "qp_time_ms": self._qp_ms,
            "oracle_time_ms": self._oracle_ms,
        }

        vectors = {
            "q_current": q_current, "dq_current": dq_current,
            "q_nom_next": self._q_nom_next, "q_corr_next": self._q_corr_next,
            "grad": self._gradient, "u_nom": self._u_nom,
            "du": self._du, "u_corr": self._u_corr,
        }
        for prefix, vec in vectors.items():
            for i, name in enumerate(self.joint_names):
                row[f"{prefix}_{name}"] = float(vec[i])

        self._writer.writerow(row)
        self._file.flush()

        self._last_x = x.copy()
        self._last_q = q_current.copy()
        self._last_u_nom = self._u_nom.copy()
        self._results.append({
            "drate": float(self._drate), "df": float(self._delta_f),
            "dg": float(self._delta_g), "f_up_g_down": float(f_up_g_down),
            "neg_to_pos": float(neg_to_pos),
            "neg_nom": float(self._rate_nom < 0.0),
            "accel_ok": float(accel_ok if accel_ok is not None else math.nan),
        })

        if math.isinf(dx):
            unique_text = "first case"
        else:
            unique_text = f"dx={dx:.3f}m dq={dq:.3f}rad du_nom={du_nom_change:.3f}rad/s"
        rospy.logwarn(
            "[stage2_logger] saved case %d (%s): drate=%+.5f df=%+.5f dg=%+.5f neg_to_pos=%d",
            self._case_id, unique_text, self._drate, self._delta_f, self._delta_g, neg_to_pos,
        )
        self._print_summary()

    def _print_summary(self) -> None:
        if not self._results:
            return
        dr = np.asarray([r["drate"] for r in self._results])
        df = np.asarray([r["df"] for r in self._results])
        dg = np.asarray([r["dg"] for r in self._results])
        disagree = np.asarray([r["f_up_g_down"] for r in self._results]) > 0.5
        neg_mask = np.asarray([r["neg_nom"] for r in self._results]) > 0.5
        flip = np.asarray([r["neg_to_pos"] for r in self._results]) > 0.5
        accel = np.asarray([r["accel_ok"] for r in self._results], dtype=np.float64)

        flip_rate = float(np.mean(flip[neg_mask])) if np.any(neg_mask) else math.nan
        accel_valid = np.isfinite(accel)
        accel_rate = float(np.mean(accel[accel_valid] > 0.5)) if np.any(accel_valid) else math.nan
        rospy.logwarn(
            "[stage2_logger] N=%d drate_up=%.1f%% learned_up=%.1f%% oracle_up=%.1f%% "
            "f_up_g_down=%.1f%% neg_to_pos=%s accel_ok=%s mean_drate=%+.5f mean_dg=%+.5f",
            len(self._results),
            100.0 * float(np.mean(dr > 1e-8)),
            100.0 * float(np.mean(df > 1e-8)),
            100.0 * float(np.mean(dg > 1e-8)),
            100.0 * float(np.mean(disagree)),
            "n/a" if math.isnan(flip_rate) else f"{100.0*flip_rate:.1f}%",
            "n/a" if math.isnan(accel_rate) else f"{100.0*accel_rate:.1f}%",
            float(np.mean(dr)), float(np.mean(dg)),
        )

    def _shutdown(self) -> None:
        try:
            if self._results:
                rospy.logwarn("[stage2_logger] ===== final Stage-II summary =====")
                self._print_summary()
                rospy.logwarn("[stage2_logger] CSV saved to: %s", self.log_path)
            else:
                rospy.logwarn("[stage2_logger] no unique cases recorded")
        finally:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass


def main() -> None:
    rospy.init_node("ncdf_action_qp_logger")
    Stage2ActionQPLogger()
    rospy.spin()


if __name__ == "__main__":
    main()

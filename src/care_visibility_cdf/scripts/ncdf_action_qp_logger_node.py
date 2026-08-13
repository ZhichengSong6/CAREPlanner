#!/usr/bin/env python3
"""Atomic Stage-II logger for CAREPlanner NCDF action-QP.

This node subscribes ONLY to /ncdf_action_qp_dryrun/summary. Every CSV row is
therefore parsed from one computation cycle and cannot mix asynchronous debug
topics from adjacent 20-Hz cycles. It never publishes robot commands.
"""

from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospkg
import rospy
from std_msgs.msg import String

PACKAGE_DIR = Path(rospkg.RosPack().get_path("care_visibility_cdf")).resolve()
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


class Stage2ActionQPLogger:
    def __init__(self) -> None:
        self.summary_topic = str(
            rospy.get_param("~summary_topic", "/ncdf_action_qp_dryrun/summary")
        )
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
        default_path = PACKAGE_DIR / "outputs" / f"ncdf_stage2_action_qp_atomic_{stamp}.csv"
        self.log_path = Path(
            rospy.get_param("~log_path", str(default_path))
        ).expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._last_x: Optional[np.ndarray] = None
        self._last_q: Optional[np.ndarray] = None
        self._last_u: Optional[np.ndarray] = None
        self._last_seq: Optional[int] = None
        self._case_id = 0
        self._results: List[Dict[str, float]] = []

        self._file = self.log_path.open("w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._file.flush()

        rospy.Subscriber(self.summary_topic, String, self._summary_cb, queue_size=20)
        rospy.on_shutdown(self._shutdown)

        rospy.logwarn("[stage2_logger] ATOMIC SUMMARY LOGGER ONLY; no action publishing.")
        rospy.loginfo("[stage2_logger] summary=%s", self.summary_topic)
        rospy.loginfo(
            "[stage2_logger] unique if dx>=%.3fm OR dq>=%.3frad OR du_nom>=%.3frad/s",
            self.unique_target_threshold_m,
            self.unique_q_threshold_rad,
            self.unique_action_threshold_rad_s,
        )
        rospy.loginfo("[stage2_logger] CSV: %s", self.log_path)

    def _fieldnames(self) -> List[str]:
        fields = [
            "case_id", "seq", "ros_time", "wall_time_unix", "dt", "execute",
            "target_x", "target_y", "target_z", "grad_norm",
            "rate_nom", "rate_corr", "delta_rate",
            "f_nom_next", "f_corr_next", "delta_f",
            "g_nom_next", "g_corr_next", "delta_g",
            "learned_up", "oracle_up", "f_up_g_down", "neg_to_pos",
            "active_bounds", "accel_ok", "accel_ratio", "nom_accel_ratio",
            "nom_vel_clip", "qpert_l2", "ncdf_time_ms", "qp_time_ms",
            "oracle_time_ms", "rate_identity_error", "action_identity_error",
            "qrollout_identity_error",
        ]
        for prefix in (
            "q_current", "dq_current", "q_nom_next", "q_corr_next",
            "grad", "u_nom", "du", "u_corr",
        ):
            fields += [f"{prefix}_{j}" for j in JOINT_NAMES]
        return fields

    @staticmethod
    def _float(text: str, key: str) -> Optional[float]:
        m = re.search(r"(?:^|\s)" + re.escape(key) + r"=(" + NUM + r")", text)
        return None if m is None else float(m.group(1))

    @staticmethod
    def _int(text: str, key: str) -> Optional[int]:
        v = Stage2ActionQPLogger._float(text, key)
        return None if v is None else int(round(v))

    @staticmethod
    def _pair(text: str, key: str) -> Tuple[Optional[float], Optional[float]]:
        m = re.search(
            r"(?:^|\s)" + re.escape(key) + r"=(" + NUM + r")->(" + NUM + r")",
            text,
        )
        if m is None:
            return None, None
        return float(m.group(1)), float(m.group(2))

    @staticmethod
    def _vec(text: str, key: str, n: int) -> Optional[np.ndarray]:
        m = re.search(r"(?:^|\s)" + re.escape(key) + r"=\[([^\]]+)\]", text)
        if m is None:
            return None
        try:
            arr = np.asarray([float(v) for v in m.group(1).split(",")], dtype=np.float64)
        except ValueError:
            return None
        if arr.shape != (n,) or not np.all(np.isfinite(arr)):
            return None
        return arr

    @staticmethod
    def _ms(text: str, key: str) -> Optional[float]:
        m = re.search(r"(?:^|\s)" + re.escape(key) + r"=(" + NUM + r")ms", text)
        return None if m is None else float(m.group(1))

    def _parse(self, text: str) -> Optional[Dict[str, object]]:
        seq = self._int(text, "seq")
        dt = self._float(text, "dt")
        execute = self._int(text, "execute")
        x = self._vec(text, "x", 3)
        q = self._vec(text, "q_cur", 7)
        dq = self._vec(text, "dq_meas", 7)
        qn = self._vec(text, "q_nom_next", 7)
        qc = self._vec(text, "q_corr_next", 7)
        grad = self._vec(text, "grad", 7)
        un = self._vec(text, "u_nom", 7)
        du = self._vec(text, "du", 7)
        uc = self._vec(text, "u_corr", 7)
        rate_nom, rate_corr = self._pair(text, "rate")
        f_nom, f_corr = self._pair(text, "f_next")
        g_nom, g_corr = self._pair(text, "g_next")

        required = [seq, dt, execute, x, q, dq, qn, qc, grad, un, du, uc,
                    rate_nom, rate_corr, f_nom, f_corr, g_nom, g_corr]
        if any(v is None for v in required):
            rospy.logwarn_throttle(1.0, "[stage2_logger] incomplete atomic summary; skipping")
            return None

        out: Dict[str, object] = {
            "seq": seq, "dt": dt, "execute": execute,
            "x": x, "q": q, "dq": dq, "qn": qn, "qc": qc,
            "grad": grad, "un": un, "du": du, "uc": uc,
            "grad_norm": self._float(text, "grad_norm"),
            "rate_nom": rate_nom, "rate_corr": rate_corr,
            "drate": self._float(text, "drate"),
            "f_nom": f_nom, "f_corr": f_corr, "df": self._float(text, "df"),
            "g_nom": g_nom, "g_corr": g_corr, "dg": self._float(text, "dg"),
            "learned_up": self._int(text, "learned_up"),
            "oracle_up": self._int(text, "oracle_up"),
            "f_up_g_down": self._int(text, "f_up_g_down"),
            "active_bounds": self._int(text, "active_bounds"),
            "accel_ok": self._int(text, "accel_ok"),
            "accel_ratio": self._float(text, "accel_ratio"),
            "nom_accel_ratio": self._float(text, "nom_accel_ratio"),
            "nom_vel_clip": self._int(text, "nom_vel_clip"),
            "qpert_l2": self._float(text, "qpert_l2"),
            "ncdf_ms": self._ms(text, "ncdf"),
            "qp_ms": self._ms(text, "qp"),
            "oracle_ms": self._ms(text, "oracle"),
        }
        scalar_required = [out[k] for k in (
            "grad_norm", "drate", "df", "dg", "learned_up", "oracle_up",
            "f_up_g_down", "active_bounds", "accel_ok", "accel_ratio",
            "nom_accel_ratio", "nom_vel_clip", "qpert_l2", "ncdf_ms",
            "qp_ms", "oracle_ms",
        )]
        if any(v is None for v in scalar_required):
            rospy.logwarn_throttle(1.0, "[stage2_logger] incomplete atomic scalars; skipping")
            return None
        return out

    def _is_unique(self, x: np.ndarray, q: np.ndarray, u: np.ndarray):
        if self._last_x is None:
            return True, math.inf, math.inf, math.inf
        dx = float(np.linalg.norm(x - self._last_x))
        dq = float(np.linalg.norm(q - self._last_q))
        du = float(np.linalg.norm(u - self._last_u))
        return (
            dx >= self.unique_target_threshold_m
            or dq >= self.unique_q_threshold_rad
            or du >= self.unique_action_threshold_rad_s,
            dx, dq, du,
        )

    def _summary_cb(self, msg: String) -> None:
        p = self._parse(msg.data)
        if p is None:
            return
        seq = int(p["seq"])
        if self._last_seq is not None and seq <= self._last_seq:
            return
        self._last_seq = seq

        x = p["x"]; q = p["q"]; un = p["un"]
        unique, dx, dq, du_nom_change = self._is_unique(x, q, un)
        if not unique:
            return

        dt = float(p["dt"])
        rate_identity_error = abs((float(p["rate_corr"]) - float(p["rate_nom"])) - float(p["drate"]))
        action_identity_error = float(np.linalg.norm((p["uc"] - p["un"]) - p["du"]))
        qrollout_identity_error = float(np.linalg.norm((p["qc"] - p["qn"]) - dt * p["du"]))

        self._case_id += 1
        neg_to_pos = int(float(p["rate_nom"]) < 0.0 and float(p["rate_corr"]) > 0.0)
        row: Dict[str, object] = {
            "case_id": self._case_id, "seq": seq,
            "ros_time": float(rospy.Time.now().to_sec()),
            "wall_time_unix": float(time.time()), "dt": dt,
            "execute": int(p["execute"]),
            "target_x": float(x[0]), "target_y": float(x[1]), "target_z": float(x[2]),
            "grad_norm": p["grad_norm"],
            "rate_nom": p["rate_nom"], "rate_corr": p["rate_corr"], "delta_rate": p["drate"],
            "f_nom_next": p["f_nom"], "f_corr_next": p["f_corr"], "delta_f": p["df"],
            "g_nom_next": p["g_nom"], "g_corr_next": p["g_corr"], "delta_g": p["dg"],
            "learned_up": p["learned_up"], "oracle_up": p["oracle_up"],
            "f_up_g_down": p["f_up_g_down"], "neg_to_pos": neg_to_pos,
            "active_bounds": p["active_bounds"], "accel_ok": p["accel_ok"],
            "accel_ratio": p["accel_ratio"], "nom_accel_ratio": p["nom_accel_ratio"],
            "nom_vel_clip": p["nom_vel_clip"], "qpert_l2": p["qpert_l2"],
            "ncdf_time_ms": p["ncdf_ms"], "qp_time_ms": p["qp_ms"],
            "oracle_time_ms": p["oracle_ms"],
            "rate_identity_error": rate_identity_error,
            "action_identity_error": action_identity_error,
            "qrollout_identity_error": qrollout_identity_error,
        }
        vectors = {
            "q_current": p["q"], "dq_current": p["dq"],
            "q_nom_next": p["qn"], "q_corr_next": p["qc"],
            "grad": p["grad"], "u_nom": p["un"], "du": p["du"], "u_corr": p["uc"],
        }
        for prefix, vec in vectors.items():
            for i, name in enumerate(JOINT_NAMES):
                row[f"{prefix}_{name}"] = float(vec[i])

        self._writer.writerow(row)
        self._file.flush()
        self._last_x = x.copy(); self._last_q = q.copy(); self._last_u = un.copy()
        self._results.append({
            "drate": float(p["drate"]), "df": float(p["df"]), "dg": float(p["dg"]),
            "neg": float(float(p["rate_nom"]) < 0.0), "flip": float(neg_to_pos),
            "disagree": float(int(p["f_up_g_down"])), "accel": float(int(p["accel_ok"])),
            "rate_err": rate_identity_error, "action_err": action_identity_error,
            "q_err": qrollout_identity_error,
        })

        unique_text = "first case" if math.isinf(dx) else (
            f"dx={dx:.3f}m dq={dq:.3f}rad du_nom={du_nom_change:.3f}rad/s"
        )
        rospy.logwarn(
            "[stage2_logger] saved case %d seq=%d (%s): drate=%+.5f df=%+.5f dg=%+.5f",
            self._case_id, seq, unique_text, p["drate"], p["df"], p["dg"],
        )
        self._print_summary()

    def _print_summary(self) -> None:
        if not self._results:
            return
        dr = np.asarray([r["drate"] for r in self._results])
        df = np.asarray([r["df"] for r in self._results])
        dg = np.asarray([r["dg"] for r in self._results])
        neg = np.asarray([r["neg"] for r in self._results]) > 0.5
        flip = np.asarray([r["flip"] for r in self._results]) > 0.5
        disagree = np.asarray([r["disagree"] for r in self._results]) > 0.5
        accel = np.asarray([r["accel"] for r in self._results]) > 0.5
        flip_rate = float(np.mean(flip[neg])) if np.any(neg) else math.nan
        max_rate_err = max(r["rate_err"] for r in self._results)
        max_action_err = max(r["action_err"] for r in self._results)
        max_q_err = max(r["q_err"] for r in self._results)
        rospy.logwarn(
            "[stage2_logger] N=%d drate_up=%.1f%% learned_up=%.1f%% oracle_up=%.1f%% "
            "f_up_g_down=%.1f%% neg_to_pos=%s accel_ok=%.1f%% sync_err=[%.1e,%.1e,%.1e]",
            len(self._results), 100*np.mean(dr>1e-8), 100*np.mean(df>1e-8),
            100*np.mean(dg>1e-8), 100*np.mean(disagree),
            "n/a" if math.isnan(flip_rate) else f"{100*flip_rate:.1f}%",
            100*np.mean(accel), max_rate_err, max_action_err, max_q_err,
        )

    def _shutdown(self) -> None:
        try:
            if self._results:
                rospy.logwarn("[stage2_logger] ===== final atomic Stage-II summary =====")
                self._print_summary()
                rospy.logwarn("[stage2_logger] CSV saved to: %s", self.log_path)
            else:
                rospy.logwarn("[stage2_logger] no unique cases recorded")
        finally:
            try:
                self._file.flush(); self._file.close()
            except Exception:
                pass


def main() -> None:
    rospy.init_node("ncdf_action_qp_logger")
    Stage2ActionQPLogger()
    rospy.spin()


if __name__ == "__main__":
    main()

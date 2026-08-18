#!/usr/bin/env python3
"""Persist unthrottled Phase B.2 adapter steering summaries to CSV."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

import rospy
from std_msgs.msg import String


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
_RANGE_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")


def _tokens(text: str) -> Dict[str, str]:
    return {k: v for k, v in _TOKEN_RE.findall(text or "")}


def _range(text: str):
    match = _RANGE_RE.match(str(text or ""))
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def _sanitize(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return out or "trial"


def _lambda_tag(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


class SteeringLogger:
    FIELDS = [
        "ros_time", "adapter_seq", "active", "seen", "inside_map",
        "confidence", "current_visibility", "steering_mode", "stage",
        "project_count", "ascent_count", "f_min", "f_max",
        "raw_grad_norm_min", "raw_grad_norm_max",
        "normalized_grad_norm_min", "normalized_grad_norm_max",
        "residual_norm_min", "residual_norm_max", "reason",
    ]

    def __init__(self) -> None:
        self.topic = str(rospy.get_param(
            "~adapter_summary_topic",
            "/phase_b2_visibility_acquisition_adapter/summary"))
        self.output_root = Path(rospy.get_param(
            "~output_root", "outputs/phase_b2_v2_ab")).expanduser().resolve()
        self.trial_label = str(rospy.get_param("~trial_label", "trial"))
        self.lambda_vis = float(rospy.get_param("~lambda_vis", 0.0))
        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_root / (
            f"steering_{_sanitize(self.trial_label)}_lambda_"
            f"{_lambda_tag(self.lambda_vis)}_{stamp}.csv")
        self.file = self.path.open("w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.sub = rospy.Subscriber(self.topic, String, self._callback, queue_size=200)
        rospy.on_shutdown(self._shutdown)
        rospy.logwarn("[phase_b2_steering_logger] recording -> %s", self.path)

    def _callback(self, msg: String) -> None:
        t = _tokens(msg.data)
        f_min, f_max = _range(t.get("f", ""))
        raw_min, raw_max = _range(t.get("raw_grad_norm", ""))
        norm_min, norm_max = _range(t.get("normalized_grad_norm", ""))
        res_min, res_max = _range(t.get("residual_norm", ""))
        self.writer.writerow({
            "ros_time": rospy.Time.now().to_sec(),
            "adapter_seq": t.get("seq", ""),
            "active": t.get("active", ""),
            "seen": t.get("seen", ""),
            "inside_map": t.get("inside_map", ""),
            "confidence": t.get("confidence", ""),
            "current_visibility": t.get("current_visibility", ""),
            "steering_mode": t.get("steering_mode", ""),
            "stage": t.get("stage", ""),
            "project_count": t.get("project_count", ""),
            "ascent_count": t.get("ascent_count", ""),
            "f_min": f_min,
            "f_max": f_max,
            "raw_grad_norm_min": raw_min,
            "raw_grad_norm_max": raw_max,
            "normalized_grad_norm_min": norm_min,
            "normalized_grad_norm_max": norm_max,
            "residual_norm_min": res_min,
            "residual_norm_max": res_max,
            "reason": t.get("reason", ""),
        })

    def _shutdown(self) -> None:
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


def main() -> None:
    rospy.init_node("phase_b2_steering_logger")
    SteeringLogger()
    rospy.spin()


if __name__ == "__main__":
    main()

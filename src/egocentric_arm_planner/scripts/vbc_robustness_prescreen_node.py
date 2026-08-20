#!/usr/bin/env python3
"""Phase-C1 VBC robustness pre-screen for CAREPlanner.

The robot stays at one measured q0.  This node publishes a deterministic set of
EE goals, records the one-shot nominal trajectory for each reachable goal, lets
the existing VBC selector choose the most urgent unseen swept point, and sends
that target/sweep time to the existing explicit VisCDF waypoint generator.

Default goals use seeded Latin-hypercube XYZ sampling with one fixed orientation.
A JSON goal list can be supplied through ~goals_file for manual experiments.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]
TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def tokens(text: str) -> Dict[str, str]:
    return {k: v for k, v in TOKEN_RE.findall(text or "")}


def safe_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def joint_state_q(msg: JointState) -> np.ndarray:
    index = {name: i for i, name in enumerate(msg.name)}
    missing = [name for name in JOINT_NAMES if name not in index]
    if missing:
        raise ValueError("JointState missing joints: " + ",".join(missing))
    q = np.asarray([msg.position[index[n]] for n in JOINT_NAMES], dtype=np.float64)
    if q.size != 7 or not np.all(np.isfinite(q)):
        raise ValueError("invalid JointState q")
    return q


def trajectory_q_at(msg: JointTrajectory, t: float) -> np.ndarray:
    if not msg.points:
        raise ValueError("empty trajectory")
    index = {name: i for i, name in enumerate(msg.joint_names)}
    missing = [name for name in JOINT_NAMES if name not in index]
    if missing:
        raise ValueError("trajectory missing joints: " + ",".join(missing))
    mapping = [index[n] for n in JOINT_NAMES]
    times = np.asarray([p.time_from_start.to_sec() for p in msg.points], dtype=np.float64)
    q = np.asarray([[p.positions[j] for j in mapping] for p in msg.points], dtype=np.float64)
    if t <= times[0]:
        return q[0].copy()
    if t >= times[-1]:
        return q[-1].copy()
    hi = int(np.searchsorted(times, t, side="right"))
    lo = hi - 1
    dt = float(times[hi] - times[lo])
    if dt <= 1e-12:
        return q[hi].copy()
    alpha = float((t - times[lo]) / dt)
    return (1.0 - alpha) * q[lo] + alpha * q[hi]


def serialize_trajectory(msg: JointTrajectory) -> Dict[str, object]:
    return {
        "header_stamp_ros_s": float(msg.header.stamp.to_sec()),
        "frame_id": str(msg.header.frame_id),
        "joint_names": list(msg.joint_names),
        "points": [
            {
                "time_from_start_s": float(p.time_from_start.to_sec()),
                "positions": [float(v) for v in p.positions],
            }
            for p in msg.points
        ],
    }


class Prescreen:
    def __init__(self) -> None:
        self.lock = threading.Lock()

        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.num_goals = int(rospy.get_param("~num_goals", 20))
        self.seed = int(rospy.get_param("~random_seed", 20260820))
        self.select_num = int(rospy.get_param("~select_num_cases", 12))
        self.safety_margin = float(rospy.get_param("~safety_margin_s", 0.30))
        self.goals_file = str(rospy.get_param("~goals_file", "")).strip()

        self.bounds = {
            "x": (float(rospy.get_param("~goal_bounds/x_min", 0.22)),
                  float(rospy.get_param("~goal_bounds/x_max", 0.46))),
            "y": (float(rospy.get_param("~goal_bounds/y_min", -0.26)),
                  float(rospy.get_param("~goal_bounds/y_max", 0.26))),
            "z": (float(rospy.get_param("~goal_bounds/z_min", 0.40)),
                  float(rospy.get_param("~goal_bounds/z_max", 0.72))),
        }
        self.orientation = np.asarray([
            float(rospy.get_param("~goal_orientation/qx", 0.0)),
            float(rospy.get_param("~goal_orientation/qy", 0.0)),
            float(rospy.get_param("~goal_orientation/qz", 0.70710678)),
            float(rospy.get_param("~goal_orientation/qw", 0.70710678)),
        ], dtype=np.float64)
        qnorm = float(np.linalg.norm(self.orientation))
        if not math.isfinite(qnorm) or qnorm < 1e-9:
            raise ValueError("invalid fixed goal orientation")
        self.orientation /= qnorm

        self.plan_timeout = float(rospy.get_param("~plan_timeout_s", 3.0))
        self.vbc_timeout = float(rospy.get_param("~vbc_timeout_s", 3.0))
        self.projector_timeout = float(rospy.get_param("~projector_timeout_s", 12.0))
        self.prior_timeout = float(rospy.get_param("~initial_prior_timeout_s", 20.0))
        self.inter_goal_delay = float(rospy.get_param("~inter_goal_delay_s", 0.20))

        self.output_path = Path(rospy.get_param(
            "~output_path", "outputs/phase_c1_vbc_prescreen/vbc_robustness_prescreen.json"
        )).expanduser().resolve()
        self.trace_dir = Path(rospy.get_param(
            "~waypoint_trace_dir", str(self.output_path.parent / "waypoint_traces")
        )).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.goal_topic = str(rospy.get_param("~goal_topic", "/care_planner/ee_target_pose"))
        self.traj_topic = str(rospy.get_param("~task_trajectory_topic", "/care_planner/task_trajectory"))
        self.vbc_topic = str(rospy.get_param("~vbc_summary_topic", "/care_planner/trajectory_risk/vbc_summary"))
        self.candidate_topic = str(rospy.get_param("~candidate_topic", "/care_planner/active_sensing/target_candidate"))
        self.sweep_topic = str(rospy.get_param("~selected_sweep_topic", "/care_planner/trajectory_risk/vbc_selected_sweep_time_s"))
        self.see_topic = str(rospy.get_param("~selected_see_topic", "/care_planner/trajectory_risk/vbc_selected_see_time_s"))
        self.margin_topic = str(rospy.get_param("~selected_margin_topic", "/care_planner/trajectory_risk/vbc_selected_margin_s"))
        self.frozen_target_topic = str(rospy.get_param("~frozen_target_topic", "/care_planner/active_sensing/target_point"))
        self.frozen_sweep_topic = str(rospy.get_param("~frozen_sweep_topic", "/care_planner/active_sensing/frozen_sweep_time_s"))
        self.generator_topic = str(rospy.get_param("~generator_summary_topic", "/care_planner/active_sensing/visibility_waypoint_summary"))
        self.qvis_topic = str(rospy.get_param("~q_vis_topic", "/care_planner/active_sensing/visibility_waypoint_q"))
        self.qzero_topic = str(rospy.get_param("~q_zero_topic", "/care_planner/active_sensing/visibility_zero_q"))
        self.prior_topic = str(rospy.get_param("~initial_prior_ready_topic", "/care_planner/confidence_map/initial_prior_ready"))
        self.joint_topic = str(rospy.get_param("~joint_states_topic", "/care_arm/joint_states"))

        if self.num_goals <= 0 or self.select_num <= 0 or self.safety_margin < 0.0:
            raise ValueError("invalid Phase-C1 count/margin parameters")
        for lo, hi in self.bounds.values():
            if not lo < hi:
                raise ValueError("invalid goal bounds")

        self.seq: Dict[str, int] = {}
        self.joint_state: Optional[JointState] = None
        self.trajectory: Optional[JointTrajectory] = None
        self.vbc_summary = ""
        self.candidate: Optional[PointStamped] = None
        self.sweep = math.nan
        self.see = math.nan
        self.margin = math.nan
        self.generator_summary = ""
        self.qvis: Optional[np.ndarray] = None
        self.qzero: Optional[np.ndarray] = None
        self.prior_ready = False

        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        self.frozen_target_pub = rospy.Publisher(self.frozen_target_topic, PointStamped, queue_size=1)
        self.frozen_sweep_pub = rospy.Publisher(self.frozen_sweep_topic, Float32, queue_size=1)

        rospy.Subscriber(self.joint_topic, JointState, self.cb_joint, queue_size=1)
        rospy.Subscriber(self.traj_topic, JointTrajectory, self.cb_traj, queue_size=1)
        rospy.Subscriber(self.vbc_topic, String, self.cb_vbc, queue_size=5)
        rospy.Subscriber(self.candidate_topic, PointStamped, self.cb_candidate, queue_size=5)
        rospy.Subscriber(self.sweep_topic, Float32, self.cb_sweep, queue_size=5)
        rospy.Subscriber(self.see_topic, Float32, self.cb_see, queue_size=5)
        rospy.Subscriber(self.margin_topic, Float32, self.cb_margin, queue_size=5)
        rospy.Subscriber(self.generator_topic, String, self.cb_generator, queue_size=5)
        rospy.Subscriber(self.qvis_topic, Float64MultiArray, self.cb_qvis, queue_size=5)
        rospy.Subscriber(self.qzero_topic, Float64MultiArray, self.cb_qzero, queue_size=5)
        rospy.Subscriber(self.prior_topic, Bool, self.cb_prior, queue_size=1)

        rospy.logwarn("[phase_c1_prescreen] armed: goals=%d seed=%d output=%s",
                      self.num_goals, self.seed, self.output_path)

    def mark(self, key: str) -> None:
        self.seq[key] = self.seq.get(key, 0) + 1

    def cb_joint(self, msg):
        with self.lock:
            self.joint_state = copy.deepcopy(msg); self.mark("joint")

    def cb_traj(self, msg):
        with self.lock:
            self.trajectory = copy.deepcopy(msg); self.mark("traj")

    def cb_vbc(self, msg):
        with self.lock:
            self.vbc_summary = str(msg.data); self.mark("vbc")

    def cb_candidate(self, msg):
        with self.lock:
            self.candidate = copy.deepcopy(msg); self.mark("candidate")

    def cb_sweep(self, msg):
        with self.lock:
            self.sweep = float(msg.data); self.mark("sweep")

    def cb_see(self, msg):
        with self.lock:
            self.see = float(msg.data); self.mark("see")

    def cb_margin(self, msg):
        with self.lock:
            self.margin = float(msg.data); self.mark("margin")

    def cb_generator(self, msg):
        with self.lock:
            self.generator_summary = str(msg.data); self.mark("gen")

    def cb_qvis(self, msg):
        with self.lock:
            self.qvis = np.asarray(msg.data, dtype=np.float64); self.mark("qvis")

    def cb_qzero(self, msg):
        with self.lock:
            self.qzero = np.asarray(msg.data, dtype=np.float64); self.mark("qzero")

    def cb_prior(self, msg):
        if bool(msg.data):
            with self.lock:
                self.prior_ready = True; self.mark("prior")

    def seqv(self, key: str) -> int:
        with self.lock:
            return int(self.seq.get(key, 0))

    def wait(self, predicate, timeout_s: float, sleep_s: float = 0.02) -> bool:
        end = time.monotonic() + timeout_s
        while not rospy.is_shutdown() and time.monotonic() < end:
            if predicate():
                return True
            time.sleep(sleep_s)
        return False

    def wait_connections(self) -> None:
        self.wait(lambda: self.goal_pub.get_num_connections() > 0, 10.0)
        self.wait(lambda: self.frozen_target_pub.get_num_connections() > 0, 10.0)
        self.wait(lambda: self.frozen_sweep_pub.get_num_connections() > 0, 10.0)

    def make_goals(self) -> List[Dict[str, object]]:
        if self.goals_file:
            path = Path(self.goals_file).expanduser().resolve()
            raw = json.loads(path.read_text())
            items = raw if isinstance(raw, list) else raw.get("goals", [])
            goals = []
            for i, item in enumerate(items[: self.num_goals]):
                p = item.get("position", [item.get("x"), item.get("y"), item.get("z")])
                q = item.get("orientation", self.orientation.tolist())
                p = [float(v) for v in p]
                q = np.asarray([float(v) for v in q], dtype=np.float64)
                if len(p) != 3 or q.size != 4 or not np.all(np.isfinite(p + q.tolist())):
                    raise ValueError("invalid manual goal at index %d" % i)
                q /= max(float(np.linalg.norm(q)), 1e-12)
                goals.append({"goal_id": f"goal_{i:03d}", "position": p,
                              "orientation": q.tolist(), "source": "manual_json"})
            return goals

        rng = np.random.default_rng(self.seed)
        n = self.num_goals
        coords = []
        for axis in ("x", "y", "z"):
            lo, hi = self.bounds[axis]
            u = (np.arange(n, dtype=np.float64) + rng.random(n)) / float(n)
            u = u[rng.permutation(n)]
            coords.append(lo + (hi - lo) * u)
        return [
            {
                "goal_id": f"goal_{i:03d}",
                "position": [float(coords[0][i]), float(coords[1][i]), float(coords[2][i])],
                "orientation": self.orientation.tolist(),
                "source": "seeded_latin_hypercube",
            }
            for i in range(n)
        ]

    def goal_msg(self, goal: Dict[str, object]) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now(); msg.header.frame_id = self.base_frame
        p = goal["position"]; q = goal["orientation"]
        msg.pose.position.x = p[0]; msg.pose.position.y = p[1]; msg.pose.position.z = p[2]
        msg.pose.orientation.x = q[0]; msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]; msg.pose.orientation.w = q[3]
        return msg

    def latest_trace(self, wall_epoch: float) -> Optional[Dict[str, object]]:
        files = sorted(self.trace_dir.glob("vbc_visibility_waypoint_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            try:
                if path.stat().st_mtime + 0.05 < wall_epoch:
                    continue
                data = json.loads(path.read_text())
                data["trace_file"] = str(path)
                return data
            except Exception:
                pass
        return None

    @staticmethod
    def choose_cases(cases: List[Dict[str, object]], n_select: int) -> List[str]:
        if not cases:
            return []
        ordered = sorted(cases, key=lambda c: float(c["distance_qvis_from_nominal_l2"]))
        n_select = min(n_select, len(ordered))
        bins = np.array_split(np.asarray(ordered, dtype=object), 3)
        labels = ["easy", "medium", "hard"]
        target_counts = [n_select // 3] * 3
        for i in range(n_select % 3):
            target_counts[i] += 1
        selected = []
        for items_np, label, take in zip(bins, labels, target_counts):
            items = list(items_np)
            for c in items:
                c["difficulty_bin"] = label
            take = min(take, len(items))
            if take:
                idx = np.linspace(0, len(items) - 1, take).round().astype(int)
                selected.extend(items[int(i)] for i in idx)
        if len(selected) < n_select:
            chosen = {id(c) for c in selected}
            selected.extend(c for c in ordered if id(c) not in chosen)
        return [str(c["case_id"]) for c in selected[:n_select]]

    def write(self, payload: Dict[str, object]) -> None:
        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=True))
        tmp.replace(self.output_path)

    def run(self) -> None:
        self.wait_connections()
        if not self.wait(lambda: self.seqv("joint") > 0, 10.0):
            raise RuntimeError("no JointState")
        if not self.wait(lambda: self.prior_ready, self.prior_timeout):
            raise RuntimeError("initial confidence prior never became ready")

        with self.lock:
            initial_js = copy.deepcopy(self.joint_state)
        initial_q = joint_state_q(initial_js)
        goals = self.make_goals()

        payload: Dict[str, object] = {
            "metadata": {
                "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "goal_generation": "manual_json" if self.goals_file else "seeded_latin_hypercube",
                "random_seed": self.seed,
                "requested_num_goals": self.num_goals,
                "actual_num_goals": len(goals),
                "select_num_cases": self.select_num,
                "safety_margin_s": self.safety_margin,
                "goal_bounds": {k: list(v) for k, v in self.bounds.items()},
                "fixed_orientation": self.orientation.tolist(),
                "initial_q": initial_q.tolist(),
                "goals_file": self.goals_file or None,
            },
            "attempts": [], "valid_cases": [], "selected_case_ids": [],
        }

        for i, goal in enumerate(goals):
            if rospy.is_shutdown():
                break
            rospy.logwarn("[phase_c1_prescreen] [%d/%d] %s p=%s",
                          i + 1, len(goals), goal["goal_id"],
                          np.round(goal["position"], 4).tolist())
            attempt = dict(goal); attempt["status"] = "started"

            with self.lock:
                current_js = copy.deepcopy(self.joint_state)
            if current_js is not None:
                q_now = joint_state_q(current_js)
                attempt["q0_drift_inf"] = float(np.max(np.abs(q_now - initial_q)))

            traj_base = self.seqv("traj")
            self.goal_pub.publish(self.goal_msg(goal))
            if not self.wait(lambda: self.seqv("traj") > traj_base, self.plan_timeout):
                attempt["status"] = "plan_failed_or_timeout"
                payload["attempts"].append(attempt); self.write(payload)
                time.sleep(self.inter_goal_delay); continue

            with self.lock:
                traj = copy.deepcopy(self.trajectory)
            attempt["trajectory_duration_s"] = float(traj.points[-1].time_from_start.to_sec())

            # Selector continuously re-evaluates its newest trajectory, so take
            # a summary strictly after the new nominal trajectory arrived.
            vbc_base = self.seqv("vbc")
            if not self.wait(lambda: self.seqv("vbc") > vbc_base, self.vbc_timeout):
                attempt["status"] = "vbc_timeout"
                payload["attempts"].append(attempt); self.write(payload)
                time.sleep(self.inter_goal_delay); continue
            with self.lock:
                vbc_summary = self.vbc_summary
            vt = tokens(vbc_summary)
            has_violation = vt.get("has_violation") == "1"
            attempt.update({
                "vbc_summary": vbc_summary,
                "has_violation": has_violation,
                "candidate_count": int(safe_float(vt.get("candidate_count"), 0)),
                "violation_count": int(safe_float(vt.get("violation_count"), 0)),
            })
            if not has_violation:
                attempt["status"] = "no_vbc_violation"
                payload["attempts"].append(attempt); self.write(payload)
                time.sleep(self.inter_goal_delay); continue

            # Require a complete fresh selected target tuple from the same case.
            pair_base = {k: self.seqv(k) for k in ("candidate", "sweep", "see", "margin")}
            if not self.wait(lambda: all(self.seqv(k) > pair_base[k] for k in pair_base),
                             self.vbc_timeout):
                attempt["status"] = "violation_pair_timeout"
                payload["attempts"].append(attempt); self.write(payload)
                time.sleep(self.inter_goal_delay); continue
            with self.lock:
                candidate = copy.deepcopy(self.candidate)
                sweep, see, margin = float(self.sweep), float(self.see), float(self.margin)
            target_xyz = [float(candidate.point.x), float(candidate.point.y), float(candidate.point.z)]
            attempt.update({
                "selected_target_xyz": target_xyz,
                "selected_link": vt.get("link"),
                "selected_sample_index": int(safe_float(vt.get("sample_index"), -1)),
                "nominal_sweep_time_s": sweep,
                "nominal_see_time_s": see,
                "nominal_vbc_margin_s": margin,
            })

            qzero_base, qvis_base, gen_base = self.seqv("qzero"), self.seqv("qvis"), self.seqv("gen")
            trace_epoch = time.time()
            frozen = copy.deepcopy(candidate)
            frozen.header.stamp = rospy.Time.now(); frozen.header.frame_id = self.base_frame
            smsg = Float32(); smsg.data = sweep
            self.frozen_target_pub.publish(frozen); self.frozen_sweep_pub.publish(smsg)

            def projection_finished() -> bool:
                if self.seqv("qzero") > qzero_base and self.seqv("qvis") > qvis_base:
                    return True
                if self.seqv("gen") > gen_base:
                    with self.lock:
                        gt = tokens(self.generator_summary)
                    return str(gt.get("reason", "")).startswith("generation_failed")
                return False

            if not self.wait(projection_finished, self.projector_timeout):
                attempt["status"] = "projector_timeout"
                payload["attempts"].append(attempt); self.write(payload)
                time.sleep(self.inter_goal_delay); continue
            with self.lock:
                gen_summary = self.generator_summary
                q_zero = None if self.qzero is None else self.qzero.copy()
                q_vis = None if self.qvis is None else self.qvis.copy()
            if (q_zero is None or q_vis is None or q_zero.size != 7 or q_vis.size != 7 or
                    not np.all(np.isfinite(q_zero)) or not np.all(np.isfinite(q_vis))):
                attempt["status"] = "projector_failed"
                attempt["generator_summary"] = gen_summary
                payload["attempts"].append(attempt); self.write(payload)
                time.sleep(self.inter_goal_delay); continue

            deadline = max(0.0, sweep - self.safety_margin)
            q_nom = trajectory_q_at(traj, deadline)
            delta = q_vis - q_nom
            case_id = f"case_{len(payload['valid_cases']):03d}"
            case: Dict[str, object] = {
                "case_id": case_id,
                "source_goal_id": goal["goal_id"],
                "goal_position": list(goal["position"]),
                "goal_orientation": list(goal["orientation"]),
                "initial_q": initial_q.tolist(),
                "selected_target_xyz": target_xyz,
                "selected_link": vt.get("link"),
                "selected_sample_index": int(safe_float(vt.get("sample_index"), -1)),
                "nominal_sweep_time_s": sweep,
                "nominal_see_time_s": see,
                "nominal_vbc_margin_s": margin,
                "safety_margin_s": self.safety_margin,
                "deadline_from_start_s": deadline,
                "q_nom_deadline": q_nom.tolist(),
                "q_zero": q_zero.tolist(),
                "q_vis": q_vis.tolist(),
                "distance_qvis_from_nominal_l2": float(np.linalg.norm(delta)),
                "distance_qvis_from_nominal_inf": float(np.max(np.abs(delta))),
                "vbc_summary": vbc_summary,
                "generator_summary": gen_summary,
                "nominal_trajectory": serialize_trajectory(traj),
            }
            trace = self.latest_trace(trace_epoch)
            if trace is not None:
                case["projector_trace_file"] = trace.get("trace_file")
                for key in (
                    "initial_f", "projection_zero_f", "final_f", "projection_root_source",
                    "distance_qzero_from_nominal", "distance_qvis_from_nominal",
                    "initial_oracle_diagnostic", "zero_oracle_diagnostic", "final_oracle_diagnostic",
                ):
                    if key in trace:
                        case[key] = trace[key]

            attempt["status"] = "valid_vbc_case"; attempt["case_id"] = case_id
            attempt["distance_qvis_from_nominal_l2"] = case["distance_qvis_from_nominal_l2"]
            payload["attempts"].append(attempt); payload["valid_cases"].append(case)
            payload["selected_case_ids"] = self.choose_cases(payload["valid_cases"], self.select_num)
            self.write(payload)
            rospy.logwarn("[phase_c1_prescreen] VALID %s sweep=%.3f margin=%s dq_l2=%.4f",
                          case_id, sweep,
                          "-inf" if not math.isfinite(margin) else f"{margin:+.3f}",
                          case["distance_qvis_from_nominal_l2"])
            time.sleep(self.inter_goal_delay)

        selected_ids = self.choose_cases(payload["valid_cases"], self.select_num)
        payload["selected_case_ids"] = selected_ids
        selected_set = set(selected_ids)
        payload["selected_cases"] = [c for c in payload["valid_cases"] if c["case_id"] in selected_set]
        counts: Dict[str, int] = {}
        for attempt in payload["attempts"]:
            status = str(attempt.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        payload["summary"] = {
            "num_attempted": len(payload["attempts"]),
            "num_valid_vbc_cases": len(payload["valid_cases"]),
            "num_selected_cases": len(selected_ids),
            "status_counts": counts,
        }
        self.write(payload)
        rospy.logwarn("[phase_c1_prescreen] DONE attempted=%d valid=%d selected=%d",
                      len(payload["attempts"]), len(payload["valid_cases"]), len(selected_ids))
        rospy.logwarn("[phase_c1_prescreen] selected=%s", selected_ids)


def main() -> None:
    rospy.init_node("vbc_robustness_prescreen")
    Prescreen().run()


if __name__ == "__main__":
    main()

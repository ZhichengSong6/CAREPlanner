#!/usr/bin/env python3
"""Case026 GEOMETRY-ONLY feasibility search using existing CARE FOV + LOS.

No neural-network inference, scalar projection, trained gradients, robot
control, or motion planning. The original conservative FOV/primitive LOS
predicate from PerSensorVisibilityRuntime._candidate_geometry is reused.

One passing (sensor, q) proves a feasible *candidate* under these two checks;
failure to find one within a finite search budget does NOT prove infeasibility.
No global collision, swept-VBC, dynamics, or reachability is certified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from urdf_parser_py.urdf import URDF

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
VIS = REPO / "src/care_visibility_cdf/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(VIS))

from test_phase_e_case026_targeted_per_sensor_fallback import (  # noqa: E402
    TARGET, MEASURED_SEED, KNOWN_BLOCKED_QVIS, Q_MIN, Q_MAX,
)
from per_sensor_visibility_runtime import (  # noqa: E402
    PerSensorVisibilityRuntime,
)
from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES,
)
from validate_visibility_oracle import (  # noqa: E402
    find_chain_joints, fk_transform, sensor_margin,
)
from check_visibility_self_occlusion import (  # noqa: E402
    load_collision_primitives, q_row_to_map,
)

DEFAULT_ROOT = Path(
    "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/"
    "outputs/mainline_b/h9_grid25_data_v1"
)


def require(ok, reason):
    if not ok:
        raise RuntimeError("[CASE026 GEOMETRY STOP] " + reason)


class GeometryOnlyCase026(PerSensorVisibilityRuntime):
    """Initialize only geometry fields; never load or instantiate a network."""

    def __init__(self, reference_urdf, self_filter_urdf):
        self.reference_robot = URDF.from_xml_file(str(reference_urdf))
        self.self_filter_robot = URDF.from_xml_file(str(self_filter_urdf))
        self.sensor_chains = [
            find_chain_joints(self.reference_robot, "base_link", frame)
            for frame in DEFAULT_SENSOR_FRAMES
        ]
        self.primitives = load_collision_primitives(self.self_filter_robot)
        require(bool(self.primitives), "Self-filter URDF has no primitives")
        self.collision_chains = [
            find_chain_joints(
                self.self_filter_robot, "base_link", primitive["link"]
            ) for primitive in self.primitives
        ]
        # Frozen Case026 analytic gate. Do not loosen any threshold.
        self.cons_hfov = 50.0
        self.cons_vfov = 66.0
        self.cons_z_min = 0.20
        self.cons_z_max = 0.70
        self.cons_delta = 0.01
        self.nom_hfov = 55.0
        self.nom_vfov = 72.0
        self.nom_z_min = 0.15
        self.nom_z_max = 0.75
        self.min_conservative_g = 0.0
        self.require_primitive_los = True

    def fov(self, q):
        q_map = q_row_to_map(DEFAULT_JOINT_NAMES, q)
        values = np.empty(8, dtype=np.float64)
        for s, chain in enumerate(self.sensor_chains):
            transform = fk_transform(chain, q_map)
            margin, _, _ = sensor_margin(
                TARGET, transform, self.cons_hfov, self.cons_vfov,
                self.cons_z_min, self.cons_z_max,
            )
            values[s] = margin - self.cons_delta
        return values

    def check(self, q, sensor):
        # Exact same conservative FOV + ray/primitive LOS implementation
        # already used by the neural Case026 diagnostic.
        return self._candidate_geometry(
            TARGET.reshape(1, 3), np.asarray(q, dtype=np.float64), sensor
        )


def maybe_h9_seeds(root):
    case = (
        root / "derived/grid25_v3_runtime_integration_v1/case026/grid25_v3_h9.json"
    )
    if not case.is_file():
        return []
    payload = json.loads(case.read_text())
    require(np.allclose(
        np.asarray(payload["target_xyz"], dtype=float), TARGET, atol=1e-8
    ), "Previous Case026 report does not use the frozen target")
    require(np.allclose(
        np.asarray(payload["measured_seed_q"], dtype=float), MEASURED_SEED,
        atol=1e-8,
    ), "Previous Case026 report does not use the frozen measured seed")
    result = []
    q_zero = payload.get("scalar", {}).get("q_zero")
    if q_zero is not None:
        result.append(("scalar_q_zero", np.asarray(q_zero, dtype=float)))
    for attempt in payload.get("per_sensor", {}).get("attempts", []):
        sid = int(attempt["sensor_id"])
        require(0 <= sid < 8, "Invalid sensor ID in prior Case026 report")
        for which in ("q_zero", "q_candidate"):
            value = attempt.get(which)
            if value is not None:
                result.append((f"h9_S{sid}_{which}", np.asarray(value, dtype=float)))
    return result


def generate_candidates(root, *, seed, local_samples, global_samples):
    seeds = [
        ("measured", np.asarray(MEASURED_SEED, dtype=float)),
        ("known_blocked_old_scalar", np.asarray(KNOWN_BLOCKED_QVIS, dtype=float)),
    ] + maybe_h9_seeds(root)
    for label, q in seeds:
        require(q.shape == (7,) and np.isfinite(q).all(),
                "Malformed fixed candidate "+label)
        if np.all(q >= Q_MIN-1e-8) and np.all(q <= Q_MAX+1e-8):
            yield label, q.copy()
    rng = np.random.default_rng(seed)
    centers = [q for _, q in seeds if np.all(q >= Q_MIN) and np.all(q <= Q_MAX)]
    require(bool(centers), "No valid search centers")
    widths = [0.05, 0.15, 0.4, 0.9]
    for i in range(local_samples):
        width = widths[i % len(widths)]
        center = centers[(i // len(widths)) % len(centers)]
        q = np.clip(center + rng.normal(0.0, width, 7), Q_MIN, Q_MAX)
        yield f"local_{i:05d}_sd{width}", q
    if global_samples:
        # Use Sobol covering the entire frozen joint-limit box, not the
        # same q0 library used to generate V3 supervision.
        qmc_engine = qmc.Sobol(d=7, scramble=True, seed=seed)
        sample = qmc_engine.random_base2(
            m=int(math.ceil(math.log2(global_samples)))
        )[:global_samples]
        for i, scaled in enumerate(sample):
            yield f"global_sobol_{i:05d}", Q_MIN+(Q_MAX-Q_MIN)*scaled


def run(args):
    require(0 <= args.seed < 2**32
            and 0 <= args.local_samples <= 100000
            and 0 <= args.global_samples <= 100000
            and 0 <= args.refine_per_sensor <= 64
            and 10 <= args.maxiter <= 1000,
            "Invalid bounded search settings")
    root = args.root.expanduser().resolve()
    output = args.output or (
        root / "derived/grid25_v3_runtime_integration_v1" /
        "case026_geometry_only.json"
    )
    output = output.expanduser().resolve()
    require(not output.exists(), "Output report already exists; refuse overwrite")
    reference = REPO/"src/arm_description/urdf/Arm.urdf"
    self_filter = REPO/"src/arm_description/urdf/Arm_with_self_filter_collision.urdf"
    require(reference.is_file() and self_filter.is_file(),
            "Reference or self-filter URDF missing")
    oracle = GeometryOnlyCase026(reference, self_filter)

    # Reproduce the precise known S4 failure before scanning: if this
    # check changes, refuse conclusions from a different geometry model.
    known = oracle.check(KNOWN_BLOCKED_QVIS, 4)
    require(
        bool(known["any_primitive_self_occluded"])
        and abs(float(known["min_conservative_g"])-0.018168) < 0.003,
        "Case026 known blocked S4 geometry does not reproduce",
    )
    print("[CASE026 GEOMETRY SANITY PASS] Known S4 q_vis self-occluded",flush=True)

    counts = dict(
        q_tested=0, fov_sensor_pairs_passed=0,
        candidates_with_any_fov=0, geometry_checks=0,
        primitive_los_blocked_after_fov=0,
        accepted=0, optimized_seeds=0,
    )
    best = [(-math.inf, None, None) for _ in range(8)]
    # Keep one best FOV candidate per distinct q for each sensor.
    top = [[] for _ in range(8)]
    found = None
    start = time.monotonic()

    def check_candidate(label, q):
        nonlocal found
        q = np.asarray(q, dtype=np.float64).reshape(7)
        require(np.isfinite(q).all()
                and np.all(q >= Q_MIN-1e-8)
                and np.all(q <= Q_MAX+1e-8),
                "Generated candidate violates joint limits")
        counts["q_tested"] += 1
        fov = oracle.fov(q)
        require(np.isfinite(fov).all(), "Nonfinite analytic FOV margin")
        if float(fov.max()) >= 0:
            counts["candidates_with_any_fov"] += 1
        for sensor in range(8):
            g = float(fov[sensor])
            if g > best[sensor][0]:
                best[sensor] = (g, label, q.copy())
            if len(top[sensor]) < max(1, args.refine_per_sensor):
                top[sensor].append((g, label, q.copy()))
                top[sensor].sort(key=lambda z: z[0], reverse=True)
            elif g > top[sensor][-1][0]:
                top[sensor][-1] = (g, label, q.copy())
                top[sensor].sort(key=lambda z: z[0], reverse=True)
            if g < 0:
                continue
            counts["fov_sensor_pairs_passed"] += 1
            actual = oracle.check(q, sensor)
            counts["geometry_checks"] += 1
            require(abs(actual["min_conservative_g"]-g) < 1e-8,
                    "FOV prefilter disagrees with original geometry gate")
            if actual["accepted"]:
                counts["accepted"] += 1
                found = dict(
                    source=label, sensor_id=sensor,
                    sensor_frame=str(DEFAULT_SENSOR_FRAMES[sensor]),
                    q=[float(v) for v in q], geometry=actual,
                    elapsed_seconds=time.monotonic()-start,
                )
                return True
            if actual["any_primitive_self_occluded"]:
                counts["primitive_los_blocked_after_fov"] += 1
        return False

    try:
        for label, candidate in generate_candidates(
            root, seed=args.seed,
            local_samples=args.local_samples,
            global_samples=args.global_samples,
        ):
            if check_candidate(label, candidate):
                break
            if counts["q_tested"] % 512 == 0:
                print(
                    "[CASE026 GEOMETRY PROGRESS]",
                    counts["q_tested"],
                    "FOV-pass=", counts["candidates_with_any_fov"],
                    "LOS-blocked=", counts["primitive_los_blocked_after_fov"],
                    "elapsed_s=", round(time.monotonic()-start, 1),
                    flush=True,
                )
        if found is None and args.refine_per_sensor:
            # Optimize analytic FOV margin only; do not differentiate through
            # primitive raycasting or create fake LOS surrogate labels.
            bounds = list(zip(Q_MIN.tolist(), Q_MAX.tolist()))
            for sensor in range(8):
                starts = top[sensor][:args.refine_per_sensor]
                for index, (_, origin, init) in enumerate(starts):
                    counts["optimized_seeds"] += 1
                    result = minimize(
                        lambda q,s=sensor: -oracle.fov(q)[s],
                        x0=init.copy(),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options=dict(maxiter=args.maxiter, ftol=1e-8),
                    )
                    if check_candidate(
                        f"refine_S{sensor}_{index}_{origin}",
                        np.clip(result.x, Q_MIN, Q_MAX),
                    ):
                        break
                if found is not None:
                    break
    except BaseException:
        # Preserve an explicit error instead of writing a false NO_SOLUTION.
        raise

    record = dict(
        format="careplanner_case026_geometry_only_feasibility_v1",
        status="COMPLETE",
        verdict=("FOUND_FOV_AND_PRIMITIVE_LOS_CANDIDATE" if found is not None
                 else "NOT_FOUND_WITHIN_FINITE_SEARCH_BUDGET"),
        target_xyz=[float(v) for v in TARGET],
        original_measured_q=[float(v) for v in MEASURED_SEED],
        original_blocked_s4=known,
        search=dict(
            seed=args.seed,
            local_samples=args.local_samples,
            global_samples=args.global_samples,
            refine_per_sensor=args.refine_per_sensor,
            maxiter=args.maxiter,
            joint_min=[float(v) for v in Q_MIN],
            joint_max=[float(v) for v in Q_MAX],
        ),
        counts=counts,
        best_fov_by_sensor=[
            dict(sensor_id=i,
                 margin=float(best[i][0]),
                 source=best[i][1],
                 q=([float(v) for v in best[i][2]]
                    if best[i][2] is not None else None))
            for i in range(8)
        ],
        selected=found,
        elapsed_seconds=time.monotonic()-start,
        diagnostic_only=True,
        limitations=[
            "No neural network used to generate, rank or score candidates",
            "FOV uses existing Case026 conservative 50/66deg z=0.20/0.70 delta=.01",
            "Primitive LOS uses existing self-filter URDF and zero-padding ray predicate",
            "No global/continuous collision or swept-VBC certification",
            "No trajectory, dynamic feasibility, execution safety or reachability proof",
            "Finite search failure is not a proof of infeasibility",
        ],
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+output.name+".",dir=output.parent)
    try:
        with os.fdopen(fd,"w") as file:
            json.dump(record,file,indent=2,sort_keys=True,allow_nan=False)
            file.write("\n")
            file.flush();os.fsync(file.fileno())
        os.replace(tmp,output)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print("[CASE026 GEOMETRY SEARCH COMPLETE]",record["verdict"],flush=True)
    print("[CASE026 GEOMETRY COUNTS]",json.dumps(counts),flush=True)
    if found is not None:
        print("[CASE026 FEASIBLE CANDIDATE]",json.dumps(
            dict(sensor_id=found["sensor_id"],q=found["q"],
                 margin=found["geometry"]["min_conservative_g"]),
            indent=2),flush=True)
    print("[CASE026 GEOMETRY REPORT]",output,flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--output",type=Path)
    p.add_argument("--seed",type=int,default=260926)
    p.add_argument("--local-samples",type=int,default=2048)
    p.add_argument("--global-samples",type=int,default=4096)
    p.add_argument("--refine-per-sensor",type=int,default=4)
    p.add_argument("--maxiter",type=int,default=80)
    args=p.parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError,ValueError,FileNotFoundError,OSError,KeyError) as e:
        raise SystemExit(str(e))

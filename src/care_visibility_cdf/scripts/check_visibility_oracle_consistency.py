#!/usr/bin/env python3

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from urdf_parser_py.urdf import URDF

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    compute_visibility,
    find_chain_joints,
    fk_transform,
)
from extract_visibility_zero_level_sets import (  # noqa: E402
    prepare_chain_specs,
    torch_tensor,
    visibility_g_batch,
)


def names_from_npz(data, key, default):
    if key in data:
        return [str(name) for name in data[key]]
    return list(default)


def scalar_from_npz(data, key, override):
    if override is not None:
        return float(override)
    if key not in data:
        raise RuntimeError(f"Missing required metadata key '{key}'. Pass it explicitly.")
    return float(data[key])


def q_row_to_map(joint_names, q_row):
    return {name: float(value) for name, value in zip(joint_names, q_row)}


def numpy_oracle_batch(robot, base_frame, sensor_frames, joint_names, p_batch, q_batch, oracle_args):
    sensor_chains = [find_chain_joints(robot, base_frame, frame) for frame in sensor_frames]

    sensor_margins = []
    active_planes = []
    best_sensor = []
    best_margin = []
    g = []
    visible = []

    for point, q_row in zip(p_batch, q_batch):
        q_map = q_row_to_map(joint_names, q_row)
        sensor_transforms = [fk_transform(chain, q_map) for chain in sensor_chains]
        margins_i, planes_i, sensor_i, margin_i, g_i, visible_i = compute_visibility(
            point.reshape(1, 3), sensor_transforms, oracle_args
        )
        sensor_margins.append(margins_i[0])
        active_planes.append(planes_i[0])
        best_sensor.append(sensor_i[0])
        best_margin.append(margin_i[0])
        g.append(g_i[0])
        visible.append(visible_i[0])

    return {
        "sensor_margins": np.asarray(sensor_margins, dtype=np.float64),
        "active_planes": np.asarray(active_planes, dtype=np.int64),
        "best_sensor": np.asarray(best_sensor, dtype=np.int64),
        "best_margin": np.asarray(best_margin, dtype=np.float64),
        "g": np.asarray(g, dtype=np.float64),
        "visible": np.asarray(visible, dtype=np.bool_),
    }


def torch_oracle_batch(robot, base_frame, sensor_frames, joint_names, p_batch, q_batch, oracle_args, device):
    all_chain_specs = prepare_chain_specs(robot, base_frame, sensor_frames, joint_names, device)

    g_list = []
    sensor_margins_list = []
    best_sensor_list = []
    active_planes_list = []

    with torch.no_grad():
        for point, q_row in zip(p_batch, q_batch):
            g_t, margins_t, sensor_t, planes_t = visibility_g_batch(
                torch_tensor(point.astype(np.float32), device),
                torch_tensor(q_row.reshape(1, -1).astype(np.float32), device),
                all_chain_specs,
                oracle_args.horizontal_fov_deg,
                oracle_args.vertical_fov_deg,
                oracle_args.z_min,
                oracle_args.z_max,
                oracle_args.delta,
            )
            g_list.append(float(g_t.detach().cpu().numpy()[0]))
            sensor_margins_list.append(margins_t.detach().cpu().numpy()[0])
            best_sensor_list.append(int(sensor_t.detach().cpu().numpy()[0]))
            active_planes_list.append(planes_t.detach().cpu().numpy()[0])

    sensor_margins = np.asarray(sensor_margins_list, dtype=np.float64)
    best_sensor = np.asarray(best_sensor_list, dtype=np.int64)
    best_margin = sensor_margins[np.arange(len(sensor_margins)), best_sensor]
    g = np.asarray(g_list, dtype=np.float64)
    visible = g >= 0.0

    return {
        "sensor_margins": sensor_margins,
        "active_planes": np.asarray(active_planes_list, dtype=np.int64),
        "best_sensor": best_sensor,
        "best_margin": best_margin,
        "g": g,
        "visible": visible,
    }


def print_comparison(title, numpy_out, torch_out, tol):
    sensor_margin_err = np.max(np.abs(numpy_out["sensor_margins"] - torch_out["sensor_margins"]))
    best_margin_err = np.max(np.abs(numpy_out["best_margin"] - torch_out["best_margin"]))
    g_err = np.max(np.abs(numpy_out["g"] - torch_out["g"]))
    best_sensor_mismatch = int(np.count_nonzero(numpy_out["best_sensor"] != torch_out["best_sensor"]))
    active_plane_mismatch = int(np.count_nonzero(numpy_out["active_planes"] != torch_out["active_planes"]))
    visible_mismatch = int(np.count_nonzero(numpy_out["visible"] != torch_out["visible"]))

    print("")
    print(title)
    print(f"num_checked:                      {len(numpy_out['g'])}")
    print(f"max sensor_margins error:         {sensor_margin_err:.9e}")
    print(f"max best_margin error:            {best_margin_err:.9e}")
    print(f"max g error:                      {g_err:.9e}")
    print(f"best_sensor mismatches:           {best_sensor_mismatch}")
    print(f"active_plane mismatches:          {active_plane_mismatch}")
    print(f"visible mismatches:               {visible_mismatch}")

    failed = (
        sensor_margin_err > tol
        or best_margin_err > tol
        or g_err > tol
        or best_sensor_mismatch != 0
        or active_plane_mismatch != 0
        or visible_mismatch != 0
    )
    return not failed


def sample_raw_pairs(raw, num_samples, seed):
    n = len(raw["p"])
    count = min(num_samples, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=count, replace=False)
    return indices, raw["p"][indices].astype(np.float64), raw["q"][indices].astype(np.float64)


def sample_q0_pairs(q0_data, num_samples, seed):
    grid_points = q0_data["grid_points"].astype(np.float64)
    q0 = q0_data["q0_templates"].astype(np.float64)
    q0_g = q0_data["q0_g"]
    active_sensor = q0_data["q0_active_sensor"]

    valid = np.isfinite(q0_g) & (active_sensor >= 0)
    ii, jj = np.where(valid)
    if len(ii) == 0:
        raise RuntimeError("No valid q0 entries found.")

    count = min(num_samples, len(ii))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ii), size=count, replace=False)
    pi = ii[pick]
    qi = jj[pick]

    p_batch = grid_points[pi]
    q_batch = q0[pi, qi]
    return pi, qi, p_batch, q_batch


def compare_raw_stored_fields(raw, indices, numpy_out, torch_out, tol):
    print("")
    print("=== Raw Stored Field Check ===")

    checks = [
        ("raw vs numpy sensor_margins", raw["sensor_margins"][indices], numpy_out["sensor_margins"]),
        ("raw vs torch sensor_margins", raw["sensor_margins"][indices], torch_out["sensor_margins"]),
        ("raw vs numpy g", raw["g"][indices], numpy_out["g"]),
        ("raw vs torch g", raw["g"][indices], torch_out["g"]),
    ]
    ok = True
    for name, lhs, rhs in checks:
        err = np.max(np.abs(lhs - rhs))
        print(f"{name:32s}: {err:.9e}")
        ok = ok and err <= tol

    best_sensor_numpy_mismatch = int(np.count_nonzero(raw["best_sensor"][indices] != numpy_out["best_sensor"]))
    best_sensor_torch_mismatch = int(np.count_nonzero(raw["best_sensor"][indices] != torch_out["best_sensor"]))
    visible_numpy_mismatch = int(np.count_nonzero(raw["visible"][indices] != numpy_out["visible"]))
    visible_torch_mismatch = int(np.count_nonzero(raw["visible"][indices] != torch_out["visible"]))

    print(f"raw vs numpy best_sensor mismatch: {best_sensor_numpy_mismatch}")
    print(f"raw vs torch best_sensor mismatch: {best_sensor_torch_mismatch}")
    print(f"raw vs numpy visible mismatch:     {visible_numpy_mismatch}")
    print(f"raw vs torch visible mismatch:     {visible_torch_mismatch}")

    ok = ok and best_sensor_numpy_mismatch == 0
    ok = ok and best_sensor_torch_mismatch == 0
    ok = ok and visible_numpy_mismatch == 0
    ok = ok and visible_torch_mismatch == 0
    return ok


def compare_q0_stored_fields(q0_data, pi, qi, numpy_out, torch_out, tol):
    print("")
    print("=== Q0 Stored Field Check ===")

    stored_g = q0_data["q0_g"][pi, qi]
    stored_margins = q0_data["q0_sensor_margins"][pi, qi]
    stored_sensor = q0_data["q0_active_sensor"][pi, qi]
    stored_planes = q0_data["q0_active_planes"][pi, qi]

    q0_g_numpy_err = np.max(np.abs(stored_g - numpy_out["g"]))
    q0_g_torch_err = np.max(np.abs(stored_g - torch_out["g"]))
    q0_margin_numpy_err = np.max(np.abs(stored_margins - numpy_out["sensor_margins"]))
    q0_margin_torch_err = np.max(np.abs(stored_margins - torch_out["sensor_margins"]))
    sensor_numpy_mismatch = int(np.count_nonzero(stored_sensor != numpy_out["best_sensor"]))
    sensor_torch_mismatch = int(np.count_nonzero(stored_sensor != torch_out["best_sensor"]))
    plane_numpy_mismatch = int(np.count_nonzero(stored_planes != numpy_out["active_planes"]))
    plane_torch_mismatch = int(np.count_nonzero(stored_planes != torch_out["active_planes"]))

    print(f"q0 stored vs numpy g error:        {q0_g_numpy_err:.9e}")
    print(f"q0 stored vs torch g error:        {q0_g_torch_err:.9e}")
    print(f"q0 stored vs numpy margins error:  {q0_margin_numpy_err:.9e}")
    print(f"q0 stored vs torch margins error:  {q0_margin_torch_err:.9e}")
    print(f"q0 stored vs numpy sensor mismatch:{sensor_numpy_mismatch}")
    print(f"q0 stored vs torch sensor mismatch:{sensor_torch_mismatch}")
    print(f"q0 stored vs numpy plane mismatch: {plane_numpy_mismatch}")
    print(f"q0 stored vs torch plane mismatch: {plane_torch_mismatch}")

    ok = (
        q0_g_numpy_err <= tol
        and q0_g_torch_err <= tol
        and q0_margin_numpy_err <= tol
        and q0_margin_torch_err <= tol
        and sensor_numpy_mismatch == 0
        and sensor_torch_mismatch == 0
        and plane_numpy_mismatch == 0
        and plane_torch_mismatch == 0
    )
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Compare numpy and torch visibility oracles on M3 raw samples and/or M4 q0 samples."
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--raw", default="", help="Optional M3 raw visibility .npz.")
    parser.add_argument("--q0", default="", help="Optional M4 zero-level .npz.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--joint-names", nargs="*", default=None)
    parser.add_argument("--sensor-frames", nargs="*", default=None)
    parser.add_argument("--horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--vertical-fov-deg", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    if not args.raw and not args.q0:
        raise RuntimeError("Pass at least one of --raw or --q0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    metadata_source = None
    raw = None
    q0_data = None
    if args.raw:
        raw = np.load(args.raw, allow_pickle=True)
        metadata_source = raw
    if args.q0:
        q0_data = np.load(args.q0, allow_pickle=True)
        if metadata_source is None:
            metadata_source = q0_data

    joint_names = args.joint_names or names_from_npz(metadata_source, "joint_names", DEFAULT_JOINT_NAMES)
    sensor_frames = args.sensor_frames or names_from_npz(metadata_source, "sensor_frames", DEFAULT_SENSOR_FRAMES)
    oracle_args = SimpleNamespace(
        horizontal_fov_deg=scalar_from_npz(metadata_source, "horizontal_fov_deg", args.horizontal_fov_deg),
        vertical_fov_deg=scalar_from_npz(metadata_source, "vertical_fov_deg", args.vertical_fov_deg),
        z_min=scalar_from_npz(metadata_source, "z_min", args.z_min),
        z_max=scalar_from_npz(metadata_source, "z_max", args.z_max),
        delta=scalar_from_npz(metadata_source, "delta", args.delta),
    )

    robot = URDF.from_xml_file(args.urdf)

    print("")
    print("=== Oracle Consistency Config ===")
    print(f"urdf:          {args.urdf}")
    print(f"raw:           {args.raw if args.raw else '<none>'}")
    print(f"q0:            {args.q0 if args.q0 else '<none>'}")
    print(f"device:        {device}")
    print(f"num_samples:   {args.num_samples}")
    print(f"tol:           {args.tol}")
    print(f"fov/range:     h={oracle_args.horizontal_fov_deg}, v={oracle_args.vertical_fov_deg}, z=[{oracle_args.z_min}, {oracle_args.z_max}]")
    print(f"delta:         {oracle_args.delta}")

    all_ok = True

    if raw is not None:
        indices, p_batch, q_batch = sample_raw_pairs(raw, args.num_samples, args.seed)
        numpy_out = numpy_oracle_batch(
            robot, args.base_frame, sensor_frames, joint_names, p_batch, q_batch, oracle_args
        )
        torch_out = torch_oracle_batch(
            robot, args.base_frame, sensor_frames, joint_names, p_batch, q_batch, oracle_args, device
        )
        all_ok = print_comparison("=== Raw Numpy vs Torch Oracle Check ===", numpy_out, torch_out, args.tol) and all_ok
        all_ok = compare_raw_stored_fields(raw, indices, numpy_out, torch_out, args.tol) and all_ok

    if q0_data is not None:
        pi, qi, p_batch, q_batch = sample_q0_pairs(q0_data, args.num_samples, args.seed + 17)
        numpy_out = numpy_oracle_batch(
            robot, args.base_frame, sensor_frames, joint_names, p_batch, q_batch, oracle_args
        )
        torch_out = torch_oracle_batch(
            robot, args.base_frame, sensor_frames, joint_names, p_batch, q_batch, oracle_args, device
        )
        all_ok = print_comparison("=== Q0 Numpy vs Torch Oracle Check ===", numpy_out, torch_out, args.tol) and all_ok
        all_ok = compare_q0_stored_fields(q0_data, pi, qi, numpy_out, torch_out, args.tol) and all_ok

    print("")
    if all_ok:
        print("[OK] Numpy and torch visibility oracles are consistent.")
    else:
        print("[FAIL] Visibility oracle consistency check failed.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Complete all valid exp81 high-point branch straight-path comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diagnose_phase_e_exp81_high_points as diag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = json.loads(args.input.read_text())
    trace5_path = next(diag.TRACE.glob("c46_obligation_005_*.json"))
    runtime = diag.runtime_from_trace(json.loads(trace5_path.read_text()))
    geometry = diag.Geometry()
    q0 = np.asarray(data["q0"], dtype=float)
    all_points = [item["xyz_base_m"] for item in data["final_points"]]
    targets = {
        "O1_high_four": [point for point in all_points if point[0] > 0],
        "O5_high_one": [point for point in all_points if point[0] < 0],
        "all_five": all_points,
    }
    results = {}
    for name, points in targets.items():
        results[name] = {}
        branches = data["production_branch_counterfactuals"][name]
        for sensor in range(8):
            branch = branches[str(sensor)]
            if not branch["root_found"]:
                continue
            print(name, "S" + str(sensor), flush=True)
            results[name][str(sensor)] = diag.path_diagnostic(
                runtime, geometry, points, all_points, q0, branch, sensor)
    output = {
        "run_id": diag.RID,
        "qualification": "OFFLINE_STRAIGHT_PATH_POINT_CENTER_DIAGNOSTIC_ONLY",
        "input": str(args.input), "input_sha256": diag.sha(args.input),
        "results": results,
        "source_sha256": {str(Path(__file__).resolve().relative_to(diag.ROOT)):
                          diag.sha(Path(__file__).resolve())},
        "limits": [
            "Straight interpolation is not a Sparse-SCP or executable trajectory.",
            "129 samples do not prove continuous clearance.",
            "Point centers omit voxel extent and the complete confidence/occupancy map.",
            "Only production-branch root_found paths are included.",
        ],
    }
    diag.save(args.output, output)
    digest = {name: {sensor: {key: value for key, value in path.items() if key != "rows"}
                     for sensor, path in paths.items()} for name, paths in results.items()}
    print(json.dumps(diag.clean(digest), indent=2), flush=True)


if __name__ == "__main__":
    main()

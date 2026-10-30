#!/usr/bin/env python3
"""Check whether the saved worst QP conflict points belong to the cold-start prior."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import yaml

from audit_case001_motion_prior import (CONFIG, JOINTS, URDF, distance_matrix,
                                         frame_placements, load_shapes)


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/phase_e_r1_mainline_30/r1_margin20_normal_progress_4_20260923"
CASES = ROOT / "outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json"


def main():
    audit = json.loads((BATCH / "normal_progress_qp_audit.json").read_text())
    source = {row["case_id"]: row for row in json.loads(CASES.read_text())["cases"]}
    config = yaml.safe_load(CONFIG.read_text())
    shapes = load_shapes(ET.parse(URDF).getroot())
    radius = np.array([config["current_body_prior"]["link_inflation_radius"][s["link"]]
                       for s in shapes], dtype=float)
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    q_indices = [model.joints[model.getJointId(name)].idx_q for name in JOINTS]
    results = []
    for case in audit["cases"]:
        q = np.zeros(model.nq)
        q[q_indices] = source[case["case_id"]]["initial_q"]
        placements = frame_placements(model, data, shapes, q)
        snapshots = []
        for snapshot in case["qp_snapshots"]:
            points = np.array([row["point"] for row in snapshot["conflict_points"]],
                              dtype=float)
            distances = distance_matrix(points, shapes, placements)
            rows = []
            for point, ds in zip(points, distances):
                index = int(np.argmin(ds - radius))
                excess = float(ds[index] - radius[index])
                rows.append(dict(point_xyz_m=point.tolist(),
                                 covered_by_start_prior=excess <= 1e-12,
                                 nearest_collision=shapes[index]["name"],
                                 nearest_link=shapes[index]["link"],
                                 exact_signed_distance_m=float(ds[index]),
                                 trusted_radius_m=float(radius[index]),
                                 outside_prior_by_m=max(0.0, excess)))
            assert snapshot["worst"]["source"] == 0  # UNKNOWN
            snapshots.append(dict(snapshot=snapshot["file"],
                                  conflict_point_count=len(rows),
                                  outside_prior_count=sum(not row["covered_by_start_prior"]
                                                          for row in rows),
                                  max_outside_prior_m=max(row["outside_prior_by_m"]
                                                          for row in rows),
                                  worst_conflict_point=rows[0],
                                  conflict_points=rows))
        results.append(dict(case_id=case["case_id"], snapshots=snapshots))
    destination = BATCH / "qp_prior_coverage.json"
    destination.write_text(json.dumps(dict(batch_id=audit["batch_id"], cases=results),
                                      indent=2) + "\n")
    print(json.dumps(dict(output=str(destination.relative_to(ROOT)),
                          outside_prior=sum(snapshot["outside_prior_count"]
                                            for case in results for snapshot in case["snapshots"]))))


if __name__ == "__main__":
    main()

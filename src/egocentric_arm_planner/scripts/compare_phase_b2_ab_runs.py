#!/usr/bin/env python3
"""Compare two Phase B.2-v2 logger summary.json files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _get(d: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def _fmt(value: Optional[float], digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path, help="baseline summary.json")
    parser.add_argument("treatment", type=Path, help="B.2-v2 summary.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline = _load(args.baseline.resolve())
    treatment = _load(args.treatment.resolve())

    b_seen = _get(baseline, "events", "confidence_seen_delay_from_target_s")
    t_seen = _get(treatment, "events", "confidence_seen_delay_from_target_s")
    b_true_any = _get(baseline, "events", "true_visible_any_delay_from_target_s")
    t_true_any = _get(treatment, "events", "true_visible_any_delay_from_target_s")
    b_true_full = _get(baseline, "events", "true_visible_full_delay_from_target_s")
    t_true_full = _get(treatment, "events", "true_visible_full_delay_from_target_s")

    b_track = _get(baseline, "metrics", "max_tracking_inf_all")
    t_track = _get(treatment, "metrics", "max_tracking_inf_all")
    b_track_acq = _get(baseline, "metrics", "max_tracking_inf_during_acquisition")
    t_track_acq = _get(treatment, "metrics", "max_tracking_inf_during_acquisition")
    b_pred = _get(baseline, "metrics", "max_pred_dev_inf_all")
    t_pred = _get(treatment, "metrics", "max_pred_dev_inf_all")
    b_final = _get(baseline, "metrics", "final_tracking_inf")
    t_final = _get(treatment, "metrics", "final_tracking_inf")

    comparison = {
        "baseline": str(args.baseline.resolve()),
        "treatment": str(args.treatment.resolve()),
        "visibility_timing": {
            "baseline_confidence_seen_s": b_seen,
            "treatment_confidence_seen_s": t_seen,
            "treatment_seen_earlier_by_s": None
            if b_seen is None or t_seen is None
            else b_seen - t_seen,
            "baseline_true_visible_any_s": b_true_any,
            "treatment_true_visible_any_s": t_true_any,
            "treatment_true_visible_any_earlier_by_s": None
            if b_true_any is None or t_true_any is None
            else b_true_any - t_true_any,
            "baseline_true_visible_full_s": b_true_full,
            "treatment_true_visible_full_s": t_true_full,
            "treatment_true_visible_full_earlier_by_s": None
            if b_true_full is None or t_true_full is None
            else b_true_full - t_true_full,
        },
        "tracking_cost": {
            "baseline_max_tracking_inf": b_track,
            "treatment_max_tracking_inf": t_track,
            "treatment_minus_baseline_max_tracking_inf": _delta(b_track, t_track),
            "baseline_max_tracking_inf_during_acquisition": b_track_acq,
            "treatment_max_tracking_inf_during_acquisition": t_track_acq,
            "treatment_minus_baseline_acquisition_tracking_inf": _delta(
                b_track_acq, t_track_acq
            ),
            "baseline_max_pred_dev_inf": b_pred,
            "treatment_max_pred_dev_inf": t_pred,
            "treatment_minus_baseline_max_pred_dev_inf": _delta(b_pred, t_pred),
            "baseline_final_tracking_inf": b_final,
            "treatment_final_tracking_inf": t_final,
            "treatment_minus_baseline_final_tracking_inf": _delta(b_final, t_final),
        },
        "mpc_visibility_status_counts": {
            "baseline": baseline.get("mpc_vis_status_counts", {}),
            "treatment": treatment.get("mpc_vis_status_counts", {}),
        },
    }

    output = args.output
    if output is None:
        output = args.treatment.resolve().parent / "comparison_vs_baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True))

    print("\n========== Phase B.2-v2 A/B ==========")
    print(f"confidence seen: baseline={_fmt(b_seen, 4)} s, v2={_fmt(t_seen, 4)} s")
    if b_seen is not None and t_seen is not None:
        print(f"  v2 earlier by: {b_seen - t_seen:+.4f} s")
    print(
        f"oracle true-visible any: baseline={_fmt(b_true_any, 4)} s, "
        f"v2={_fmt(t_true_any, 4)} s"
    )
    print(
        f"oracle true-visible full: baseline={_fmt(b_true_full, 4)} s, "
        f"v2={_fmt(t_true_full, 4)} s"
    )
    print(f"max tracking_inf: baseline={_fmt(b_track)}, v2={_fmt(t_track)}")
    print(f"max pred_dev_inf: baseline={_fmt(b_pred)}, v2={_fmt(t_pred)}")
    print(f"final tracking_inf: baseline={_fmt(b_final)}, v2={_fmt(t_final)}")
    print(f"baseline vis statuses: {baseline.get('mpc_vis_status_counts', {})}")
    print(f"v2 vis statuses:       {treatment.get('mpc_vis_status_counts', {})}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

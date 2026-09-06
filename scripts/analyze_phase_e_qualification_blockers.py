#!/usr/bin/env python3
"""Strict per-case blocker table for a Phase-E qualification batch.

This is an offline diagnostic. It does not run ROS, Gazebo, the planner, or any
learned model. It accepts either:
  * a CAREPlanner_PHASE_E_EMPTY_QUALIFICATION_*.zip archive, or
  * an extracted qualification directory.

The script answers a deliberately narrow question:
  Are many failed cases being blocked by the same few workspace voxels / robot
  body samples?

Definitions
-----------
Predicted unsafe blocker:
  A row in candidate_vbc_summary.csv with
    trajectory_source=predicted, has_violation=1
  and a finite 3-D target point.

First blocker:
  Chronologically first predicted unsafe blocker in a case.

Last blocker:
  Chronologically last predicted unsafe blocker in a case. This is called
  "last", not "final", because it is the last recorded unsafe representative,
  not a claim about the planner's latent causal state.

Dominant blocker key:
  (grid point xyz, link, sample_index) with the largest count among predicted
  unsafe blockers in that case.

Dominant blocker point:
  grid point xyz only. This second aggregation intentionally ignores which body
  sample generated the same voxel, because the scientific question may be
  "are the same workspace voxels repeatedly blocking?"

Outputs
-------
  <output-dir>/per_case_blockers.csv
  <output-dir>/batch_blocker_summary.json
  <output-dir>/dominant_blocker_keys.csv
  <output-dir>/dominant_blocker_points.csv
  <output-dir>/last_blocker_keys.csv
  <output-dir>/last_blocker_points.csv

The terminal also prints the strict per-case table and batch concentration
summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

XYZ = Tuple[float, float, float]
BlockerKey = Tuple[XYZ, str, int]


def token_rows(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    out: List[Dict[str, object]] = []
    with path.open(newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            return out
        time_i = header.index("%time") if "%time" in header else 0
        data_i = header.index("field.data") if "field.data" in header else 1
        for order, row in enumerate(rd):
            if len(row) <= data_i:
                continue
            # rostopic CSV splits comma-containing token values such as xyz;
            # joining from field.data onward reconstructs the original string.
            text = ",".join(row[data_i:])
            d: Dict[str, object] = dict(TOKEN.findall(text))
            if not d:
                continue
            try:
                d["_t"] = float(row[time_i]) / 1e9
            except Exception:
                d["_t"] = math.nan
            d["_order"] = order
            out.append(d)
    return out


def fnum(v: object, default: float = math.nan) -> float:
    try:
        x = float(v)
        return x
    except Exception:
        return default


def inum(v: object, default: int = -1) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def vec3(v: object) -> Optional[XYZ]:
    if v is None:
        return None
    s = str(v).strip().strip("[]()")
    try:
        vals = [float(x) for x in s.split(",")]
    except Exception:
        return None
    if len(vals) != 3 or not all(math.isfinite(x) for x in vals):
        return None
    # The Phase-E blocker points lie on the confidence-map grid. Rounding here
    # only makes textual formatting differences harmless; it does not cluster
    # geometrically different voxels.
    return tuple(round(x, 4) for x in vals)  # type: ignore[return-value]


def xyz_text(xyz: Optional[XYZ]) -> str:
    if xyz is None:
        return "none"
    return "[{:.4f},{:.4f},{:.4f}]".format(*xyz)


def blocker_key(row: Dict[str, object]) -> Optional[BlockerKey]:
    xyz = vec3(row.get("target"))
    if xyz is None:
        return None
    return (
        xyz,
        str(row.get("link", "none")),
        inum(row.get("sample_index"), -1),
    )


def key_text(key: Optional[BlockerKey]) -> str:
    if key is None:
        return "none"
    xyz, link, sample = key
    return f"{xyz_text(xyz)}|{link}|sample{sample}"


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(errors="replace"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def last_token(path: Path) -> Dict[str, object]:
    rows = token_rows(path)
    return rows[-1] if rows else {}


def locate_batch_root(root: Path) -> Path:
    """Find the directory containing case_artifacts."""
    if (root / "case_artifacts").is_dir():
        return root
    matches = [p.parent for p in root.rglob("case_artifacts") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find case_artifacts under {root}. "
            "Expected a Phase-E empty-world qualification archive/directory."
        )
    # Prefer the shallowest match if an archive has an extra top-level folder.
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def extract_if_needed(src: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if src.is_dir():
        return src.resolve(), None
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.suffix.lower() != ".zip":
        raise ValueError("--input must be a qualification directory or .zip")
    tmp = tempfile.TemporaryDirectory(prefix="careplanner_blocker_table_")
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name), tmp


def summary_value(summary: Dict[str, object], key: str, default: object = "") -> object:
    return summary.get(key, default)


def analyze_case(case_id: str, case_dir: Path, summary_dir: Path) -> Dict[str, object]:
    candidate_rows = token_rows(case_dir / "candidate_vbc_summary.csv")
    unsafe_rows: List[Dict[str, object]] = []
    for row in candidate_rows:
        if row.get("trajectory_source") != "predicted":
            continue
        if str(row.get("has_violation", "")) != "1":
            continue
        if blocker_key(row) is None:
            continue
        unsafe_rows.append(row)

    # Preserve recorded chronological order. Rosbag/rostopic CSV is already
    # chronological; _order is a deterministic tie-breaker.
    unsafe_rows.sort(
        key=lambda r: (
            fnum(r.get("_t"), math.inf),
            inum(r.get("_order"), 0),
        )
    )

    keys: List[BlockerKey] = [
        k for k in (blocker_key(r) for r in unsafe_rows) if k is not None
    ]
    points: List[XYZ] = [k[0] for k in keys]
    key_counts = Counter(keys)
    point_counts = Counter(points)

    first_key = keys[0] if keys else None
    last_key = keys[-1] if keys else None
    dominant_key, dominant_key_count = (
        key_counts.most_common(1)[0] if key_counts else (None, 0)
    )
    dominant_point, dominant_point_count = (
        point_counts.most_common(1)[0] if point_counts else (None, 0)
    )
    n_unsafe = len(keys)

    blk_last = last_token(case_dir / "blocker_stack_summary.csv")
    reg_last = last_token(case_dir / "regime_summary.csv")
    commit_last = last_token(case_dir / "commit_summary.csv")
    acq_last = last_token(case_dir / "visibility_acquisition_summary.csv")
    case_summary = read_json(summary_dir / f"{case_id}.json")

    task_success = bool(case_summary.get("task_success", False))
    overall_safe = bool(case_summary.get("overall_safe", False))

    # Prefer the regime manager's terminal state when present.
    final_state = str(
        reg_last.get(
            "state",
            case_summary.get("final_regime_state", "unknown"),
        )
    )
    commit_count = inum(
        reg_last.get(
            "commit_count",
            commit_last.get("commit_count", case_summary.get("commit_count", -1)),
        ),
        -1,
    )
    cycle_block_count = inum(blk_last.get("cycle_block_count"), 0)
    seen_obligation_count = inum(
        acq_last.get("seen_obligation_count"),
        inum(case_summary.get("obligation_clear_events"), 0),
    )

    top3_keys = key_counts.most_common(3)
    top3_points = point_counts.most_common(3)

    return {
        "case_id": case_id,
        "task_success": int(task_success),
        "overall_safe": int(overall_safe),
        "final_state": final_state,
        "commit_count": commit_count,
        "cycle_block_count": cycle_block_count,
        "seen_obligation_count": seen_obligation_count,
        "candidate_vbc_record_count": len(candidate_rows),
        "predicted_unsafe_blocker_count": n_unsafe,
        "unique_blocker_key_count": len(key_counts),
        "unique_blocker_point_count": len(point_counts),
        "first_blocker_xyz": xyz_text(first_key[0] if first_key else None),
        "first_blocker_link": first_key[1] if first_key else "none",
        "first_blocker_sample": first_key[2] if first_key else -1,
        "dominant_blocker_xyz": xyz_text(
            dominant_key[0] if dominant_key is not None else None
        ),
        "dominant_blocker_link": dominant_key[1] if dominant_key else "none",
        "dominant_blocker_sample": dominant_key[2] if dominant_key else -1,
        "dominant_blocker_count": dominant_key_count,
        "dominant_blocker_fraction": (
            dominant_key_count / n_unsafe if n_unsafe else 0.0
        ),
        "dominant_point_xyz": xyz_text(dominant_point),
        "dominant_point_count": dominant_point_count,
        "dominant_point_fraction": (
            dominant_point_count / n_unsafe if n_unsafe else 0.0
        ),
        "last_blocker_xyz": xyz_text(last_key[0] if last_key else None),
        "last_blocker_link": last_key[1] if last_key else "none",
        "last_blocker_sample": last_key[2] if last_key else -1,
        "top3_blocker_keys": ";".join(
            f"{key_text(k)}:{count}" for k, count in top3_keys
        ),
        "top3_blocker_points": ";".join(
            f"{xyz_text(p)}:{count}" for p, count in top3_points
        ),
        # Private exact objects retained for aggregate computation only.
        "_first_key": first_key,
        "_dominant_key": dominant_key,
        "_dominant_point": dominant_point,
        "_last_key": last_key,
        "_key_counts": key_counts,
        "_point_counts": point_counts,
    }


def write_counter_csv(
    path: Path,
    counter: Counter,
    total_cases: int,
    mode: str,
) -> None:
    if mode not in {"key", "point"}:
        raise ValueError(mode)
    fields = (
        ["rank", "blocker_xyz", "link", "sample", "case_count", "case_fraction"]
        if mode == "key"
        else ["rank", "blocker_xyz", "case_count", "case_fraction"]
    )
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, (item, count) in enumerate(counter.most_common(), 1):
            if mode == "key":
                xyz, link, sample = item
                row = {
                    "rank": rank,
                    "blocker_xyz": xyz_text(xyz),
                    "link": link,
                    "sample": sample,
                    "case_count": count,
                    "case_fraction": count / total_cases if total_cases else 0.0,
                }
            else:
                row = {
                    "rank": rank,
                    "blocker_xyz": xyz_text(item),
                    "case_count": count,
                    "case_fraction": count / total_cases if total_cases else 0.0,
                }
            w.writerow(row)


def clean_row(row: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="CAREPlanner_PHASE_E_EMPTY_QUALIFICATION_*.zip or extracted directory",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory. Default: <repo>/outputs/"
            "phase_e_qualification_blocker_analysis/<input-stem>"
        ),
    )
    ap.add_argument(
        "--repo",
        default=".",
        help="CAREPlanner repository root; used only for the default output path",
    )
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    extracted_root, tmp = extract_if_needed(src)
    try:
        batch_root = locate_batch_root(extracted_root)
        case_artifacts = batch_root / "case_artifacts"
        summary_dir = batch_root / "case_summaries"

        case_dirs = sorted(
            [p for p in case_artifacts.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        if not case_dirs:
            raise RuntimeError(f"No case directories found in {case_artifacts}")

        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser().resolve()
        else:
            repo = Path(args.repo).expanduser().resolve()
            stem = src.stem if src.is_file() else batch_root.name
            out_dir = (
                repo
                / "outputs"
                / "phase_e_qualification_blocker_analysis"
                / stem
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            analyze_case(p.name, p, summary_dir)
            for p in case_dirs
        ]

        fields = [
            "case_id",
            "task_success",
            "overall_safe",
            "final_state",
            "commit_count",
            "cycle_block_count",
            "seen_obligation_count",
            "candidate_vbc_record_count",
            "predicted_unsafe_blocker_count",
            "unique_blocker_key_count",
            "unique_blocker_point_count",
            "first_blocker_xyz",
            "first_blocker_link",
            "first_blocker_sample",
            "dominant_blocker_xyz",
            "dominant_blocker_link",
            "dominant_blocker_sample",
            "dominant_blocker_count",
            "dominant_blocker_fraction",
            "dominant_point_xyz",
            "dominant_point_count",
            "dominant_point_fraction",
            "last_blocker_xyz",
            "last_blocker_link",
            "last_blocker_sample",
            "top3_blocker_keys",
            "top3_blocker_points",
        ]

        per_case_csv = out_dir / "per_case_blockers.csv"
        with per_case_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: clean_row(row).get(k, "") for k in fields})

        total_cases = len(rows)
        cases_with_unsafe = sum(
            1 for r in rows if int(r["predicted_unsafe_blocker_count"]) > 0
        )

        dominant_key_cases = Counter(
            r["_dominant_key"] for r in rows if r["_dominant_key"] is not None
        )
        dominant_point_cases = Counter(
            r["_dominant_point"] for r in rows if r["_dominant_point"] is not None
        )
        last_key_cases = Counter(
            r["_last_key"] for r in rows if r["_last_key"] is not None
        )
        last_point_cases = Counter(
            r["_last_key"][0]
            for r in rows
            if r["_last_key"] is not None
        )

        global_key_counts: Counter = Counter()
        global_point_counts: Counter = Counter()
        for r in rows:
            global_key_counts.update(r["_key_counts"])
            global_point_counts.update(r["_point_counts"])

        write_counter_csv(
            out_dir / "dominant_blocker_keys.csv",
            dominant_key_cases,
            total_cases,
            "key",
        )
        write_counter_csv(
            out_dir / "dominant_blocker_points.csv",
            dominant_point_cases,
            total_cases,
            "point",
        )
        write_counter_csv(
            out_dir / "last_blocker_keys.csv",
            last_key_cases,
            total_cases,
            "key",
        )
        write_counter_csv(
            out_dir / "last_blocker_points.csv",
            last_point_cases,
            total_cases,
            "point",
        )

        def top_items(counter: Counter, mode: str, n: int = 10) -> List[Dict[str, object]]:
            out = []
            for item, count in counter.most_common(n):
                if mode == "key":
                    xyz, link, sample = item
                    out.append({
                        "xyz": list(xyz),
                        "link": link,
                        "sample": sample,
                        "count": count,
                    })
                else:
                    out.append({"xyz": list(item), "count": count})
            return out

        # Scientific concentration metrics. Denominator is cases with a recorded
        # predicted unsafe blocker, because cases with no blocker cannot have a
        # meaningful dominant/last blocker.
        denom = cases_with_unsafe
        dom_key_counts = [c for _, c in dominant_key_cases.most_common()]
        dom_point_counts = [c for _, c in dominant_point_cases.most_common()]
        last_key_counts = [c for _, c in last_key_cases.most_common()]
        last_point_counts = [c for _, c in last_point_cases.most_common()]

        report = {
            "analysis": "phase_e_qualification_strict_blocker_table",
            "input": str(src),
            "batch_root": str(batch_root),
            "case_count": total_cases,
            "cases_with_predicted_unsafe_blocker": cases_with_unsafe,
            "task_success_count": sum(int(r["task_success"]) for r in rows),
            "overall_safe_count": sum(int(r["overall_safe"]) for r in rows),
            "dominant_key_case_concentration": {
                "top1": (sum(dom_key_counts[:1]) / denom) if denom else None,
                "top3": (sum(dom_key_counts[:3]) / denom) if denom else None,
            },
            "dominant_point_case_concentration": {
                "top1": (sum(dom_point_counts[:1]) / denom) if denom else None,
                "top3": (sum(dom_point_counts[:3]) / denom) if denom else None,
            },
            "last_key_case_concentration": {
                "top1": (sum(last_key_counts[:1]) / denom) if denom else None,
                "top3": (sum(last_key_counts[:3]) / denom) if denom else None,
            },
            "last_point_case_concentration": {
                "top1": (sum(last_point_counts[:1]) / denom) if denom else None,
                "top3": (sum(last_point_counts[:3]) / denom) if denom else None,
            },
            "top_dominant_blocker_keys_by_case": top_items(
                dominant_key_cases, "key"
            ),
            "top_dominant_blocker_points_by_case": top_items(
                dominant_point_cases, "point"
            ),
            "top_last_blocker_keys_by_case": top_items(last_key_cases, "key"),
            "top_last_blocker_points_by_case": top_items(
                last_point_cases, "point"
            ),
            "top_global_unsafe_blocker_keys_by_record": top_items(
                global_key_counts, "key"
            ),
            "top_global_unsafe_blocker_points_by_record": top_items(
                global_point_counts, "point"
            ),
            "cases": [clean_row(r) for r in rows],
        }

        summary_json = out_dir / "batch_blocker_summary.json"
        summary_json.write_text(json.dumps(report, indent=2, allow_nan=True))

        print("")
        print("=" * 150)
        print("STRICT PHASE-E PER-CASE BLOCKER TABLE")
        print("=" * 150)
        header = (
            f"{'case_id':<22} {'unsafe':>6} {'cycles':>6} "
            f"{'first xyz':<23} {'dominant xyz':<23} {'dom link/sample':<25} "
            f"{'dom%':>6} {'last xyz':<23} {'last link/sample':<25}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            dom_ls = f"{r['dominant_blocker_link']}/s{r['dominant_blocker_sample']}"
            last_ls = f"{r['last_blocker_link']}/s{r['last_blocker_sample']}"
            print(
                f"{str(r['case_id']):<22} "
                f"{int(r['predicted_unsafe_blocker_count']):>6} "
                f"{int(r['cycle_block_count']):>6} "
                f"{str(r['first_blocker_xyz']):<23} "
                f"{str(r['dominant_blocker_xyz']):<23} "
                f"{dom_ls:<25} "
                f"{100.0 * float(r['dominant_blocker_fraction']):>5.1f}% "
                f"{str(r['last_blocker_xyz']):<23} "
                f"{last_ls:<25}"
            )

        print("")
        print("=" * 80)
        print("BATCH CONCENTRATION")
        print("=" * 80)
        print(f"cases                         : {total_cases}")
        print(f"cases with predicted blocker  : {cases_with_unsafe}")
        print(f"task successes                : {report['task_success_count']}")
        print("")
        print("Dominant blocker POINT by case (ignoring link/sample):")
        for rank, (point, count) in enumerate(
            dominant_point_cases.most_common(10), 1
        ):
            frac = count / denom if denom else 0.0
            print(f"  {rank:2d}. {xyz_text(point):<24} cases={count:2d} ({100*frac:5.1f}%)")
        print("")
        print("Dominant blocker KEY by case (point + link + sample):")
        for rank, (key, count) in enumerate(
            dominant_key_cases.most_common(10), 1
        ):
            frac = count / denom if denom else 0.0
            print(f"  {rank:2d}. {key_text(key):<55} cases={count:2d} ({100*frac:5.1f}%)")
        print("")
        print("Last recorded blocker POINT by case:")
        for rank, (point, count) in enumerate(last_point_cases.most_common(10), 1):
            frac = count / denom if denom else 0.0
            print(f"  {rank:2d}. {xyz_text(point):<24} cases={count:2d} ({100*frac:5.1f}%)")

        print("")
        print("[OUTPUT]", per_case_csv)
        print("[OUTPUT]", summary_json)
        print("[OUTPUT]", out_dir / "dominant_blocker_keys.csv")
        print("[OUTPUT]", out_dir / "dominant_blocker_points.csv")
        print("[OUTPUT]", out_dir / "last_blocker_keys.csv")
        print("[OUTPUT]", out_dir / "last_blocker_points.csv")
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One serial original-30 run with the current CASE001 R1/strong-prior mainline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from phase_e_owned_processes import remember_descendants, reap_owned

def fingerprint():
    suffixes = {'.cpp', '.hpp', '.h', '.py', '.yaml', '.yml', '.launch',
                '.sh', '.urdf', '.xacro', '.world', '.xml', '.json', '.msg', '.srv'}
    paths = {p for root in ('src', 'scripts') for p in (REPO / root).rglob('*')
             if p.is_file() and (p.suffix in suffixes or p.name == 'CMakeLists.txt')
             and not any(x in p.parts for x in ('third_party', '__pycache__'))}
    paths.update((R1, GCDF, CASES, MODEL_BINDING_PATH))
    paths.update(p for p in (REPO / 'devel/lib').rglob('*') if p.is_file()
                 and p.suffix != '.pyc' and '__pycache__' not in p.parts)
    result = {os.path.relpath(p, REPO): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}
    for p in (Path.cwd()/"src/third_party/piqp/include").rglob("*"):
        if p.is_file():result[str(p.relative_to(Path.cwd()))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result



REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json"
PRIOR = REPO / "src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml"
R1 = REPO / "src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt"
MODEL_BINDING_PATH = Path(os.environ["CARE_CDF_MODEL_BINDING"]).resolve()
MODEL_BINDING = json.loads(MODEL_BINDING_PATH.read_text())
GCDF = Path(MODEL_BINDING["checkpoint"]).resolve()
if MODEL_BINDING["activation"] not in ("gelu", "relu"):
    raise ValueError("Explicit actual activation required")
MAINLINE = REPO / "scripts/run_phase_e_case001_r1_mainline.sh"
CASE_SHA = "fbcd2816f594e1970f59e4aae2183f14a9764e623d80b86b2d37cf3a4cad1d45"
R1_SHA = "4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002"
GCDF_SHA = MODEL_BINDING["checkpoint_sha256"]
DEMO_PATH = REPO.parent / 'demo_configuration.json'
DEMO = json.loads(DEMO_PATH.read_text()) if DEMO_PATH.is_file() else None
if DEMO:
    CASE_SHA = DEMO['cases_sha256']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def free_port(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def validate():
    assert sha(CASES) == CASE_SHA, "original 30-case input SHA changed"
    assert sha(R1) == R1_SHA and sha(GCDF) == GCDF_SHA, "mainline checkpoint SHA changed"
    data = json.loads(CASES.read_text())
    ids = [c["case_id"] for c in data["cases"]]
    expected = DEMO['case_ids'] if DEMO else [f"phase_e_goal_{i:03d}" for i in range(30)]
    assert ids == expected, "unexpected task configuration"
    assert all(len(c.get("initial_q", [])) == 7 and
               max(abs(float(q)) for q in c["initial_q"]) < 1e-4
               for c in data["cases"]), "initial q differs from home-pose experiment"
    assert PRIOR.is_file() and MAINLINE.is_file()
    return ids


def evaluate(run_dir, case_id, batch_id, destination, log):
    command = ["/usr/bin/python3", "scripts/evaluate_phase_d_run.py",
               "--repo", str(REPO), "--run-dir", str(run_dir),
               "--cases-json", str(CASES), "--case-id", case_id,
               "--method", "phase_e_r1_strong_prior_30", "--trial-id", batch_id,
               "--require-benchmark-window", "--output-json", str(destination)]
    result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            check=False)
    return result.returncode


def summary_rows(rows):
    return dict(case_count=len(rows), task_success_count=sum(bool(r.get("task_success")) for r in rows),
                overall_safe_count=sum(bool(r.get("overall_safe")) for r in rows),
                task_and_safe_count=sum(bool(r.get("task_success")) and bool(r.get("overall_safe"))
                                        for r in rows),
                case_ids=[r["case_id"] for r in rows],
                failed_case_ids=[r["case_id"] for r in rows if not r.get("task_success")],
                unsafe_case_ids=[r["case_id"] for r in rows if not r.get("overall_safe")])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-id", default="r1_strong_prior_30_" + time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--case-ids", default="", help="Comma-separated original case numbers, e.g. 000,005,018,024")
    ap.add_argument("--run-seconds-wall", type=int, default=60,
                    help="Wall watchdog duration; values other than 60 are diagnostic runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.batch_id):
        ap.error("invalid batch ID")
    if not 1 <= args.run_seconds_wall <= 120:
        ap.error("--run-seconds-wall must be 1..120")
    os.chdir(REPO)
    original_ids = validate()
    if args.case_ids:
        numbers = args.case_ids.split(",")
        if not numbers or any(not re.fullmatch(r"\d{3}", number) for number in numbers):
            ap.error("--case-ids must be comma-separated three-digit numbers")
        ids = [f"phase_e_goal_{number}" for number in numbers]
        if len(set(ids)) != len(ids) or any(case_id not in original_ids for case_id in ids):
            ap.error("--case-ids must be unique members of the original 30")
    else:
        ids = original_ids
    # Other ROS/Gazebo sessions may be active on this host. This batch uses
    # dedicated per-case ports and only reaps descendants it spawned.
    batch = REPO / "outputs/phase_e_r1_mainline_30" / args.batch_id
    assert not batch.exists(), "batch output exists; no overwrite"
    for i, case_id in enumerate(ids):
        run_id = f"{case_id}_{args.batch_id}"
        assert len(run_id) < 100 and re.fullmatch(r"[A-Za-z0-9_-]+", run_id)
        assert not (REPO / "outputs/c5_5_vbc_gcdf_regime" / run_id).exists()
        assert not (REPO / "logs/c5_5_vbc_gcdf_regime" / run_id).exists()
        assert not (REPO / f"CAREPlanner_C5_RESULT_{run_id}.zip").exists()
        assert free_port(12900 + 2*i) and free_port(12901 + 2*i), "batch port in use"
    if args.dry_run:
        print(json.dumps(dict(status="PREFLIGHT_OK", case_count=len(ids), case_ids=ids, batch_id=args.batch_id,
                              protocol=f"R1 + CASE001 strong prior + {args.run_seconds_wall}s wall watchdog + early goal stop")))
        return 0
    # This gate is only for the explicitly authorized OLD-model harness study.
    # It cannot qualify or launch an unknown replacement model.
    from launch_gate import verify_old_diagnostic
    verify_old_diagnostic(args.batch_id, ids, args.run_seconds_wall)

    frozen = fingerprint()
    frozen[str(R1.relative_to(REPO))] = R1_SHA
    frozen[str(GCDF)] = sha(GCDF)
    frozen[str(MODEL_BINDING_PATH)] = sha(MODEL_BINDING_PATH)
    batch.mkdir(parents=True)
    (batch / "logs").mkdir()
    (batch / "case_summaries").mkdir()
    metadata = dict(batch_id=args.batch_id, git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        original_case_sha256=CASE_SHA, r1_sha256=R1_SHA, gcdf_sha256=GCDF_SHA,
        model_activation=MODEL_BINDING["activation"], model_binding_sha256=sha(MODEL_BINDING_PATH),
        prior_sha256=sha(PRIOR), mainline_sha256=sha(MAINLINE),
        original_case_ids=original_ids, selected_case_ids=ids,
        run_seconds_wall=args.run_seconds_wall, early_stop_on_goal=True,
        diagnostic_window=args.run_seconds_wall != 60,
        force_zero_initial_q=False, tof_fusion_enabled=True,
        execution_gcdf_audit_enabled=False, vbc_continuous_motion_bound_enabled=True,
        current_prior=str(PRIOR.relative_to(REPO)),
        source_hashes=frozen, run_started_wall_unix_s=time.time())
    save(batch / "metadata.json", metadata)
    rows = []
    try:
        for i, case_id in enumerate(ids):
            run_id = f"{case_id}_{args.batch_id}"
            run_root = REPO / "outputs/c5_5_vbc_gcdf_regime" / run_id / "run"
            env = os.environ.copy()
            env.update(CHECKPOINT=str(GCDF), CDF_ACTIVATION=MODEL_BINDING["activation"],
                       CDF_EXPECTED_SHA256=GCDF_SHA, CDF_CHECKPOINT_KEY=str(MODEL_BINDING.get("checkpoint_key", "latest")),
                       SUMMARY_STATE_CLOCK_COHERENT="true", FIXED_START_RECOVERY_ENABLED="true",
                       FIXED_START_REBASE_REAUDIT_ENABLED="true")
            env.update(CASE_FILE=str(CASES), CASE_ID=case_id, RUN_ID=run_id,
                       RUN_SECONDS=str(args.run_seconds_wall), EARLY_STOP_ON_GOAL="true",
                       FORCE_ZERO_INITIAL_Q="false", APPLY_INITIAL_JOINT_OVERRIDES="false",
                       ROS_MASTER_URI=f"http://127.0.0.1:{12900+2*i}",
                       GAZEBO_MASTER_URI=f"http://127.0.0.1:{12901+2*i}")
            print(f"CASE START {i+1}/{len(ids)} {case_id}", flush=True)
            log_path = batch / "logs" / f"{case_id}.log"
            owned = {}
            with log_path.open("x") as log:
                command = ["bash", "--noprofile", "--norc", "-c",
                           "source /opt/ros/noetic/setup.bash && exec bash scripts/run_phase_e_case001_r1_mainline.sh"]
                proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 360
                while proc.poll() is None:
                    remember_descendants([proc.pid], owned)
                    if time.monotonic() > deadline:
                        proc.terminate()
                        raise TimeoutError(f"case runner exceeded 360s: {case_id}")
                    time.sleep(1)
                runner_rc = proc.returncode
                cleanup = reap_owned(owned)
                save(batch / "logs" / f"{case_id}_cleanup.json", cleanup)
                eval_path = batch / "case_summaries" / f"{case_id}.json"
                eval_rc = evaluate(run_root, case_id, args.batch_id, eval_path, log) if run_root.is_dir() else 2
                if eval_path.exists():
                    row = json.loads(eval_path.read_text())
                else:
                    row = dict(case_id=case_id, task_success=False, overall_safe=False,
                               evaluation_error=True)
                    save(eval_path, row)
                row["runner_return_code"] = runner_rc
                row["evaluator_return_code"] = eval_rc
                save(batch / "logs" / f"{case_id}_status.json",
                     dict(runner_return_code=runner_rc, evaluator_return_code=eval_rc))
            rows.append(row)
            save(batch / "summary_partial.json", dict(**summary_rows(rows), cases=rows))
            print(f"CASE END {i+1}/{len(ids)} {case_id} task={row.get('task_success')} "
                  f"safe={row.get('overall_safe')} runner={runner_rc} evaluator={eval_rc}", flush=True)
            if not (free_port(12900 + 2*i) and free_port(12901 + 2*i)):
                raise RuntimeError(f"owned ROS/Gazebo port remained open after {case_id}")
            if runner_rc != 0 or eval_rc != 0:
                raise RuntimeError(f"infrastructure/evaluation failure in {case_id}; no automatic retry")
            actual = fingerprint()
            actual[str(R1.relative_to(REPO))] = sha(R1)
            actual[str(GCDF)] = sha(GCDF)
            actual[str(MODEL_BINDING_PATH)] = sha(MODEL_BINDING_PATH)
            if actual != frozen:
                raise RuntimeError(f"source/model/devel files changed during batch after {case_id}")
        summary = dict(**summary_rows(rows), cases=rows,
                       run_completed_wall_unix_s=time.time())
        save(batch / "summary.json", summary)
        print("BATCH COMPLETE", json.dumps({k:v for k,v in summary.items() if k != "cases"}), flush=True)
    finally:
        save(batch / "run_state.json", dict(completed_cases=len(rows), expected_cases=len(ids),
                                            completed=len(rows) == len(ids),
                                            wall_unix_s=time.time()))


if __name__ == "__main__":
    sys.exit(main())

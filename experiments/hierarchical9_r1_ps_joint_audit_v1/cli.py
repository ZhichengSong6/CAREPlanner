#!/usr/bin/env python3
"""One job: exact frozen solver + same-point geometry + parameter gradients."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
import numpy as np
import torch
from common import (
    PACKAGE, PS_REL, Dependencies, assert_disjoint_outputs, assert_unchanged,
    check_source_manifest, checked_starts, checkpoint_manifest, load_models,
    read_json, sha, source_fingerprints, urdf_masks, write_json)
from plan import make_plan, source_checks


def package_hashes():
    return {p.name: sha(p) for p in sorted(PACKAGE.iterdir()) if p.suffix in (".py", ".sh", ".json")}


def check_job(job):
    if package_hashes() != job["package_hashes"]:
        raise ValueError("Frozen evaluation package changed")
    repo = Path(job["paths"]["repo"])
    check_source_manifest(repo, job["source_sha256"])
    for file, digest in job["source_input_manifests"].items():
        if sha(Path(file)) != digest:
            raise ValueError(f"Source manifest changed after submission: {file}")
    for meta in job["models"].values():
        if sha(Path(meta["file"])) != meta["file_sha256"]:
            raise ValueError("Checkpoint changed after submission")
    checked_starts(Path(job["paths"]["starts"]), job["protocol"])


def open_sources(job, *, verify):
    repo = Path(job["paths"]["repo"])
    deps = Dependencies(repo)
    cache = deps.data.Cache(Path(job["paths"]["cache"]), verify=verify)
    ps = Path(job["paths"]["ps_root"])/"formal"
    cache.load_indices(ps/"indices", verify=verify)
    pairs = deps.pairdata.PairCache(ps/"pairs", cache, verify=verify)
    run = read_json(ps/"run.json")
    if (cache.identity != run["cache_identity"] or cache.index_identity != run["index_identity"] or
            pairs.identity != run["pair_identity"]):
        raise ValueError("PS run/cache/index/pair identity mismatch")
    return deps, cache, pairs


def preflight(args):
    out = args.out.resolve()
    if (out/"job.json").exists():
        raise FileExistsError("Refusing to replace an existing job manifest")
    paths = {k: str(getattr(args, k).resolve()) for k in ("repo", "r012_root", "ps_root", "cache", "starts")}
    cfg = read_json(PACKAGE/"protocol.json")
    repo = Path(paths["repo"])
    source = source_fingerprints(repo)
    deps = Dependencies(repo)
    models, alias, run = checkpoint_manifest(paths, cfg, deps)
    cache = deps.data.Cache(Path(paths["cache"]), verify=False)
    ps = Path(paths["ps_root"])/"formal"
    pairs = deps.pairdata.PairCache(ps/"pairs", cache, verify=False)
    production, shards = source_checks(cache, pairs)
    assert_disjoint_outputs(out, [Path(paths[k]) for k in ("r012_root", "ps_root", "cache")] + [production])
    with np.load(shards[0], allow_pickle=False) as z:
        n = sum(len(pi) for pi, _, _, _ in deps.pairdata._pairs(z))
    starts = checked_starts(Path(paths["starts"]), cfg)
    # Import actual server dependencies before requesting GPUs, but don't construct the large q0 bank.
    deps.runtime()
    __import__("per_sensor_visibility_runtime")
    __import__("urdf_parser_py.urdf")
    manifests = [ps/"run.json", ps/"indices/manifest.json", ps/"pairs/manifest.json",
                 Path(paths["cache"])/"manifest.json", production/"manifest.json",
                 production/"v4_summary.json", Path(paths["r012_root"])/"formal/R1/run.json",
                 Path(paths["starts"]).parent/"manifest.json"]
    cp = torch.load(models["R1"]["file"], map_location="cpu", weights_only=False)
    urdf = Path(cp["args"]["urdf"]).resolve()
    datafile = Path(cp["args"]["data"]).resolve()
    if not urdf.is_file() or not datafile.is_file():
        raise FileNotFoundError("R1 original data/URDF unavailable")
    manifests.append(urdf)
    job = {"format": cfg["format"], "paths": paths, "protocol": cfg,
           "package_hashes": package_hashes(), "source_sha256": source,
           "models": models, "aliases": alias, "training_protocol": run["protocol"],
           "source_input_manifests": {str(p): sha(p) for p in manifests},
           "urdf_file": str(urdf),
           "original_bank_metadata": {"file": str(datafile), "bytes": datafile.stat().st_size,
                                      "mtime_ns": datafile.stat().st_mtime_ns, "sha256": "NOT_REHASHED"},
           "joint_names": list(deps.joint_names), "sensor_frames": list(deps.sensor_frames),
           "preflight": {"real_source_shard_pairs": n, "frozen_starts": len(starts),
                         "torch_version": torch.__version__, "cuda_run": "NOT_RUN"}}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"job.json", job)
    print("[preflight] original R1 and PS best/final hashes, real V4 schema, frozen starts verified", flush=True)
    print("[model aliases]", json.dumps(alias), flush=True)
    print("[evaluation] all 1963 solver cases; no optimizer steps, no relabeling", flush=True)


def prepare(job, out):
    deps, cache, pairs = open_sources(job, verify=True)
    make_plan(out, job["protocol"], deps, cache, pairs)


def worker(args):
    out = args.job.resolve().parent
    job = read_json(args.job)
    cfg = job["protocol"]
    if not 0 <= args.rank < cfg["world"]:
        raise ValueError("Invalid rank")
    check_job(job)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one allocated CUDA GPU")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", 0)
    deps, cache, pairs = open_sources(job, verify=False)
    deps.runtime()
    models = load_models(job["models"], job["aliases"], deps, device)
    cp = torch.load(job["models"]["R1"]["file"], map_location="cpu", weights_only=False)
    ca = cp["args"]
    dataset = deps.dataset_class(job["original_bank_metadata"]["file"], val_count=int(ca.get("val_count", 1000)), seed=int(ca.get("seed", 0)))
    lo, hi = dataset.q_limits(device)
    masks = dataset.sensor_masks(device).float()
    mask2 = urdf_masks(Path(job["urdf_file"]), deps.joint_names, deps.sensor_frames)
    mapping = {"joint_names": deps.joint_names, "sensor_frames": deps.sensor_frames,
               "dataset_masks": masks.detach().cpu().numpy(), "urdf_ancestry_masks": mask2,
               "masks_equal": bool(np.array_equal(mask2, masks.detach().cpu().numpy()))}
    if not mapping["masks_equal"]:
        raise ValueError("Sensor/joint dependency order mismatch between data and URDF")
    old_x = dataset.x_cpu.detach().cpu().numpy()
    mapping["workspace_index_arrays_equal"] = bool(np.array_equal(old_x, cache.x))
    mapping["workspace_shapes"] = [list(old_x.shape), list(cache.x.shape)]
    # Point coordinates themselves are recorded in the plan, so an index-mapping discrepancy is retained for review.
    oracle = deps.oracle.SensorOracle(Path(job["urdf_file"]), device, deps.joint_names, deps.sensor_frames)
    del cp
    write_json(out/f"mapping.rank{args.rank}.json", mapping)
    print(f"[rank{args.rank}] GPU={torch.cuda.get_device_name()} aliases={job['aliases']}", flush=True)
    from solver import run_solver
    starts = checked_starts(Path(job["paths"]["starts"]), cfg)
    run_solver(out, starts, models, deps, oracle, masks, lo, hi, cfg, args.rank)
    selection = read_json(out/"selection.json")
    if sha(out/"selected.npz") != selection["selected_sha256"]:
        raise ValueError("Sample plan changed")
    with np.load(out/"selected.npz", allow_pickle=False) as z:
        selected = {k: z[k] for k in z.files}
    from boundary import run_boundary
    ids = np.arange(args.rank, len(selected["case_id"]), cfg["world"])
    run_boundary(out, selected, ids, models, oracle, masks, lo, hi, cfg, args.rank)
    from gradients import run_gradients
    run_gradients(out, models, cache, pairs, job["training_protocol"], deps.objective,
                  cfg, args.rank, device)
    assert_unchanged(models, job["models"])
    check_job(job)
    artifacts = [out/f"{stem}.rank{args.rank}.{ext}" for stem, ext in
                 (("solves", "jsonl"), ("boundary", "jsonl"), ("activations", "json"),
                  ("gradients", "json"), ("gradient_indices", "npz"), ("mapping", "json"))]
    write_json(out/f"rank{args.rank}.complete.json", {
        "status": "COMPLETE", "rank": args.rank, "optimizer_updates": 0,
        "selected_sha256": selection["selected_sha256"],
        "files": {p.name: sha(p) for p in artifacts}, "model_weights_unchanged": True})
    print(f"[rank{args.rank}] ALL THREE STAGES COMPLETE; model weights unchanged", flush=True)


def execute(args):
    out = args.job.resolve().parent
    job = read_json(args.job)
    cfg = job["protocol"]
    check_job(job)
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(allocated) != cfg["world"] or any(not x for x in allocated):
        raise RuntimeError("Expected four explicitly allocated GPUs in CUDA_VISIBLE_DEVICES")
    print("[prepare] verify existing caches and match original pairs; no labels generated", flush=True)
    prepare(job, out)
    processes, logs = [], []
    ranks = out/"ranks"
    ranks.mkdir()
    t0 = time.monotonic()
    def stop_all():
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(_signal, _frame):
        stop_all()
        raise RuntimeError("Job terminated; partial results retained")
    signal.signal(signal.SIGTERM, terminate)
    try:
        for rank, gpu in enumerate(allocated):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
            handle = (ranks/f"rank{rank}.log").open("x")
            logs.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-u", str(PACKAGE/"cli.py"), "worker",
                "--job", str(args.job.resolve()), "--rank", str(rank)], env=env, stdout=handle, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Evaluator rank failed; see rank logs printed below")
            time.sleep(.5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("An evaluator rank failed")
    except BaseException:
        stop_all()
        for rank in range(len(processes)):
            p = ranks/f"rank{rank}.log"
            print(f"===== rank{rank}: last 50 log lines =====", flush=True)
            print("\n".join(p.read_text(errors="replace").splitlines()[-50:]), flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        for handle in logs:
            handle.close()
    check_job(job)
    # Full checksum revalidation proves our read-only path did not alter either source cache.
    open_sources(job, verify=True)
    old = job["original_bank_metadata"]
    st = Path(old["file"]).stat()
    if st.st_size != old["bytes"] or st.st_mtime_ns != old["mtime_ns"]:
        raise ValueError("Original R1 bank changed while evaluating")
    from report import merge
    merge(out)
    write_json(out/"complete.json", {"status": "COMPLETE", "optimizer_updates": 0,
               "report_sha256": sha(out/"report.json"), "seconds_after_prepare": time.monotonic()-t0,
               "cache_hashes_revalidated": True})
    print(f"[done] {out}/summary.md; no checkpoint promoted", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("preflight")
    for key in ("repo", "r012-root", "ps-root", "cache", "starts", "out"):
        p.add_argument("--"+key, type=Path, required=True)
    for action in ("execute", "worker"):
        p = sub.add_parser(action)
        p.add_argument("--job", type=Path, required=True)
        if action == "worker":
            p.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    try:
        globals()[args.action](args)
    except BaseException as exc:
        if hasattr(args, "job"):
            out = args.job.resolve().parent
            name = f"failure.rank{args.rank}.json" if hasattr(args, "rank") else "failure.json"
            write_json(out/name, {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()

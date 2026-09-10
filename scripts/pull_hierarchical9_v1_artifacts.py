#!/usr/bin/env python3
"""Pull the evaluated VisCDF V1 artifacts onto THIS computer (Python >=3.8).

Run from a LOCAL CAREPlanner checkout, not from the Slurm login node.
Uses normal OpenSSH authentication; never requests credentials in arguments,
changes SSH host-key checks, uploads files, unpickles checkpoints, or runs jobs.
Existing different files are never overwritten. No training dataset is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

HOST = "zsong142@10.120.17.131"
REMOTE_REPO = "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner"
BASE = "src/care_visibility_cdf/checkpoints/"
EVAL = "outputs/hierarchical9_eval_full_16036/"
# Pinned to the user's successful 16036 evaluation log, not guessed from names.
MODELS = {
    BASE + "hierarchical9_scratch_seed0/final.pt":
        "979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199",
    BASE + "exp1_yiming_k500_fov_signed/final.pt":
        "fea15cb71b278b9d003337d200f3796ccbbe8a59106287a8ea85abb2c89cb0df",
    BASE + "per_sensor_e2e_fullbatch_seed0/final.pt":
        "43f962729adcd17aa114edb9fc410facbbb97ebe7343f0ad3309fe50d273acdb",
}
REPORTS = [EVAL + name for name in ("manifest.json", "comparison.json", "summary.md")]
REPORTS.append(BASE + "hierarchical9_scratch_seed0/train_args.json")
SSH_OPTIONS = ["-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=3"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_target(repo: Path, relative: str) -> Path:
    path = repo / relative
    if path.is_symlink():
        raise RuntimeError("Refusing a symlink destination: " + str(path))
    try:
        path.resolve().relative_to(repo.resolve())
    except ValueError:
        raise RuntimeError("Destination leaves the checkout: " + str(path))
    return path


def remote_hashes(host: str, remote_repo: str, paths: list) -> dict:
    # One read-only SSH command. Only stdout contains machine-readable JSON;
    # passwords and first-connection host-key prompts remain handled by SSH.
    program = (
        "import hashlib,json,pathlib\n"
        "root=pathlib.Path(" + repr(remote_repo) + ")\n"
        "paths=" + repr(paths) + "\n"
        "out={}\n"
        "for rel in paths:\n"
        " h=hashlib.sha256()\n"
        " with (root/rel).open('rb') as f:\n"
        "  for block in iter(lambda:f.read(1048576),b''):\n"
        "   h.update(block)\n"
        " out[rel]=h.hexdigest()\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, host, "python3 -c " + shlex.quote(program)],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    values = json.loads(result.stdout)
    if not isinstance(values, dict) or set(values) != set(paths):
        raise RuntimeError("Unexpected server manifest; no files downloaded")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
           for v in values.values()):
        raise RuntimeError("Invalid server SHA256 manifest")
    for rel, expected in MODELS.items():
        if values[rel] != expected:
            raise RuntimeError("Server checkpoint differs from evaluated V1: " + rel)
    return values


def install_verified(source: Path, destination: Path, expected: str) -> None:
    """Atomic no-clobber install, with source kept until the caller cleans up."""
    if digest(source) != expected:
        raise RuntimeError("SHA256 mismatch after transfer: " + str(destination))
    # Source and destination are on the same filesystem. A hard link publishes
    # only the already complete file, and fails rather than replacing a file.
    try:
        os.link(str(source), str(destination))
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file() or digest(destination) != expected:
            raise RuntimeError("Destination appeared or changed during transfer: " + str(destination))


def fetch_one(repo: Path, host: str, remote_repo: str, rel: str, expected: str) -> None:
    destination = local_target(repo, rel)
    if destination.exists():
        if not destination.is_file() or digest(destination) != expected:
            raise RuntimeError("Preserving existing DIFFERENT file; move it aside manually: " + str(destination))
        print("[verified existing] " + rel, flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".v1-transfer-", suffix=".part", dir=str(destination.parent))
    os.close(fd)
    temporary = Path(name)
    try:
        print("[download] " + rel, flush=True)
        subprocess.run(["scp", *SSH_OPTIONS, host + ":" + remote_repo.rstrip("/") + "/" + rel,
                        str(temporary)], check=True)
        install_verified(temporary, destination, expected)
        print("[verified] " + rel, flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


def check_local(repo: Path, paths: list, expected: dict) -> None:
    for rel in paths:
        p = local_target(repo, rel)
        if not p.is_file():
            raise FileNotFoundError(p)
        if rel in expected and digest(p) != expected[rel]:
            raise RuntimeError("Local SHA256 mismatch: " + rel)
        if p.suffix == ".json":
            json.loads(p.read_text(encoding="utf-8"))
    manifest = json.loads((repo / EVAL / "manifest.json").read_text(encoding="utf-8"))
    for name, rel in zip(("hierarchical", "old_scalar", "old8"), MODELS):
        item = manifest.get("checkpoints", {}).get(name, {})
        if item.get("step") != 50000 or item.get("sha256") != MODELS[rel]:
            raise RuntimeError("Evaluation manifest does not identify the expected final model: " + name)
    comparison = json.loads((repo / EVAL / "comparison.json").read_text(encoding="utf-8"))
    if comparison.get("field", {}).get("hierarchical_sensor_max", {}).get("count") != 163840:
        raise RuntimeError("Expected full eval 16036, not a smoke report")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=".", help="Local checkout or any directory inside it")
    p.add_argument("--host", default=HOST, help="SSH user@host or configured SSH alias")
    p.add_argument("--remote-repo", default=REMOTE_REPO)
    p.add_argument("--include-samples", action="store_true", help="Also copy optional samples.npz")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Print plan; no network and no writes")
    group.add_argument("--verify-only", action="store_true", help="Offline verification; no downloads")
    args = p.parse_args()
    if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*", args.host):
        p.error("Use user@host or an SSH alias (IPv6/ports belong in SSH config)")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_repo):
        p.error("Remote repo must be an absolute path without whitespace or shell metacharacters")
    if shutil.which("git") is None:
        p.error("git is required")
    root = subprocess.check_output(["git", "-C", str(Path(args.repo).expanduser()),
                                    "rev-parse", "--show-toplevel"], text=True).strip()
    repo = Path(root).resolve()
    required_source = repo / "experiments/hierarchical9_scratch_v1/model.py"
    if not required_source.is_file():
        raise RuntimeError("Synchronize CAREPlanner code first; missing " + str(required_source))
    paths = list(MODELS) + REPORTS + ([EVAL + "samples.npz"] if args.include_samples else [])
    print("LOCAL destination: " + str(repo), flush=True)
    print("REMOTE source: " + args.host + ":" + args.remote_repo, flush=True)
    if args.dry_run:
        for rel in paths:
            local_target(repo, rel)
            print("[plan] " + rel + (" sha256=" + MODELS[rel] if rel in MODELS else ""))
        print("[dry-run] No SSH/SCP, file writes, checkpoint loading, or runtime changes")
        return
    if args.verify_only:
        check_local(repo, paths, MODELS)
        print("[done] local_v1_models_verified (offline: report provenance not rechecked against server)")
        return
    if any(shutil.which(name) is None for name in ("ssh", "scp")):
        p.error("OpenSSH ssh and scp are required")
    hashes = remote_hashes(args.host, args.remote_repo, paths)
    # Fail on any local conflict before beginning transfers.
    for rel in paths:
        destination = local_target(repo, rel)
        if destination.exists() and (not destination.is_file() or digest(destination) != hashes[rel]):
            raise RuntimeError("Preserving existing DIFFERENT file; move it aside manually: " + str(destination))
    for rel in paths:
        fetch_one(repo, args.host, args.remote_repo, rel, hashes[rel])
    check_local(repo, paths, hashes)
    for rel, value in MODELS.items():
        print("[MODEL SHA256] " + value + "  " + rel)
    print("[done] artifacts_ready_for_mainline_a")
    print("No training data copied; no runtime switched. Do NOT git-add checkpoint binaries or outputs.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit("[ERROR] " + str(exc))

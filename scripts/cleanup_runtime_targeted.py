#!/usr/bin/env python3
"""Conservative recovery for CAREPlanner's targeted runtime qualification.

Default: inspect only. Use --apply to cancel verified matching jobs, archive an
incomplete output, and remove only stale R012_RUNTIME_TARGETED_JOB assignments.
Requires Python 3.8+, Linux, and working squeue/scontrol/scancel commands.
Never run concurrently with the old submitter (which does not honor our lock).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

DEFAULT_REPO = "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b"
DEFAULT_ROOT = "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1"
JOB_NAME = "h9_r1_rt_tgt"
KEY = "R012_RUNTIME_TARGETED_JOB"
KEY_LINE = re.compile(r"^\s*(?:export\s+)?" + KEY + r"\s*=")
ID_LINE = re.compile(r"^\s*(?:export\s+)?" + KEY + r"\s*=\s*['\"]?(\d+)['\"]?\s*$")


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("command failed: {}\n{}".format(" ".join(args), result.stderr.strip()))
    return result.stdout


def queue(user):
    text = command("squeue", "--noheader", "--user", user, "--format=%i|%j")
    found = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("|", 1)
        if len(parts) != 2:
            raise RuntimeError("Unrecognized squeue output; refusing cleanup: " + line)
        found[parts[0].strip()] = parts[1].strip()
    return found


def fields(text):
    # Slurm's -o output separates Key=value fields by whitespace. Values may
    # contain spaces; stop only at another capitalized field name.
    return dict(re.findall(r"(?:^|\s)([A-Z][A-Za-z0-9_]*)=(.*?)(?=\s+[A-Z][A-Za-z0-9_]*=|$)", text.strip()))


def matching_jobs(user, repo, root, recorded):
    result = {}
    for jid, name in queue(user).items():
        if name != JOB_NAME and jid not in recorded:
            continue
        if not jid.isdigit():
            raise RuntimeError("Unexpected array/heterogeneous job ID; inspect manually: " + jid)
        try:
            info = fields(command("scontrol", "show", "job", "-o", jid))
        except RuntimeError:
            if jid not in queue(user):  # Job finished between the two reads.
                continue
            raise
        owner = info.get("UserId", "").split("(", 1)[0]
        workdir = Path(info.get("WorkDir", "/nonexistent")).resolve()
        stdout = info.get("StdOut", "")
        output_matches = stdout.startswith(str(root / "logs" / "r1_runtime_targeted_"))
        verified = (owner == user and info.get("JobName") == JOB_NAME
                    and workdir == repo and output_matches)
        if verified:
            result[jid] = info
        elif output_matches:
            raise RuntimeError("Job {} may write this output root but identity differs; inspect manually.".format(jid))
        else:
            print("[skip-job] {} is not this exact experiment".format(jid))
    return result


def completed(out):
    try:
        with (out / "targeted_compare.json").open() as stream:
            return json.load(stream).get("valid_diagnostic") is True
    except (OSError, ValueError, AttributeError):
        return False


def atomic_write(path, text):
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp = tempfile.mkstemp(prefix=".r012_jobs.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def execute(args, repo, root):
    user = pwd.getpwuid(os.getuid()).pw_name
    envfile = root / "r012_jobs.env"
    out = root / "runtime_targeted_v1_r1"
    if envfile.is_symlink() or out.is_symlink():
        raise RuntimeError("Output/envfile is a symlink; inspect manually before cleanup.")
    if out.exists() and not out.is_dir():
        raise RuntimeError("Expected an output directory: " + str(out))
    original_env = envfile.read_text() if envfile.exists() else ""
    recorded = {match.group(1) for line in original_env.splitlines()
                for match in [ID_LINE.match(line)] if match}
    jobs = matching_jobs(user, repo, root, recorded)
    print("[mode] " + ("APPLY" if args.apply else "DRY-RUN; no changes"))
    print("[repo] " + str(repo))
    print("[root] " + str(root))
    print("[verified-jobs] " + (", ".join(sorted(jobs)) or "none"))
    valid = completed(out) if out.exists() else False
    print("[output] " + ("KEEP completed result" if valid else
                         "archive incomplete output" if out.exists() else "not present"))
    new_env = "".join(line for line in original_env.splitlines(keepends=True)
                      if not KEY_LINE.match(line))
    print("[env] " + ("backup, then remove only " + KEY if new_env != original_env else "unchanged"))
    print("[preserve] checkpoints, formal/, datasets, historical logs, and all other jobs")
    if not args.apply:
        return

    # Re-read immediately before cancellation. Never cancel a newly appeared job.
    refreshed = matching_jobs(user, repo, root, recorded)
    if set(refreshed) - set(jobs):
        raise RuntimeError("A new matching job appeared; stop concurrent submissions and retry.")
    canceled = []
    for jid in sorted(refreshed):
        current = matching_jobs(user, repo, root, recorded)
        if set(current) - set(jobs):
            raise RuntimeError("New matching job detected; refusing further changes.")
        if jid in current:
            try:
                command("scancel", jid)
            except RuntimeError:
                if jid in queue(user):
                    raise
            canceled.append(jid)
            print("[cancel-requested] " + jid)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        live = matching_jobs(user, repo, root, recorded)
        if set(live) - set(jobs):
            raise RuntimeError("New matching job appeared; output has not been moved.")
        if not live:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Jobs still active/completing; output has not been moved. Retry later.")
        time.sleep(2)

    # The new submitter must use the same lock. The legacy submitter does not.
    if (envfile.read_text() if envfile.exists() else "") != original_env:
        raise RuntimeError("r012_jobs.env changed during cleanup; refusing to move output.")
    if envfile.is_symlink() or out.is_symlink():
        raise RuntimeError("A path changed to a symlink during cleanup; refusing changes.")
    archive = root / ".runtime_recovery_archive" / (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + str(os.getpid()))
    archive.mkdir(parents=True, exist_ok=False)
    if envfile.exists():
        shutil.copy2(str(envfile), str(archive / "r012_jobs.env.before"))
    if out.exists() and not completed(out):
        out.rename(archive / "runtime_targeted_v1_r1")
        print("[archived-output] " + str(archive / "runtime_targeted_v1_r1"))
    elif out.exists():
        print("[kept] completed output; the old worker intentionally refuses to overwrite it")
    if new_env != original_env:
        atomic_write(envfile, new_env)
    (archive / "recovery.json").write_text(json.dumps({
        "repo": str(repo), "root": str(root), "user": user,
        "cancel_requested": canceled, "verified_jobs_before": jobs,
        "policy": "archive incomplete output; preserve completed output and all model/data artifacts",
    }, indent=2) + "\n")
    print("[done] archive=" + str(archive))
    print("[note] No new job submitted. Existing historical logs were left in place.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="default")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--root", default=os.environ.get("R012_ROOT", DEFAULT_ROOT))
    parser.add_argument("--wait-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative")
    repo, root = Path(args.repo).resolve(strict=True), Path(args.root).resolve(strict=True)
    if not repo.is_dir() or not root.is_dir() or root == Path("/" ):
        raise RuntimeError("Invalid repository/output root")
    for tool in ("squeue", "scontrol", "scancel"):
        if not shutil.which(tool):
            raise RuntimeError("Missing {}: run this on the cluster login host.".format(tool))
    if args.apply:
        lock_path = root / ".runtime_targeted.control.lock"
        if lock_path.is_symlink():
            raise RuntimeError("Lock path is a symlink; refusing cleanup")
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args, repo, root)
    else:
        execute(args, repo, root)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print("[ERROR] " + str(exc), file=sys.stderr)
        sys.exit(2)

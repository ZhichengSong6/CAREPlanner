#!/usr/bin/env bash
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT="${R012_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
mkdir -p "$ROOT/logs"
ROOT=$(realpath -e "$ROOT")
cd "$REPO"

ARTIFACT_REPO="${R012_ARTIFACT_REPO:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner}"
V1_REL="src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt"
SCALAR_REL="src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt"

resolve_checkpoint() {
  local rel="$1"
  if [[ -f "$REPO/$rel" ]]; then
    realpath -e "$REPO/$rel"
    return
  fi
  if [[ -f "$ARTIFACT_REPO/$rel" ]]; then
    realpath -e "$ARTIFACT_REPO/$rel"
    return
  fi
  echo "[ERROR] checkpoint not found in worktree or artifact repo: $rel" >&2
  echo "        worktree=$REPO" >&2
  echo "        artifact_repo=$ARTIFACT_REPO" >&2
  exit 2
}

V1_CHECKPOINT=$(resolve_checkpoint "$V1_REL")
SCALAR_CHECKPOINT=$(resolve_checkpoint "$SCALAR_REL")

python3 - "$V1_CHECKPOINT" "$SCALAR_CHECKPOINT" <<'PY'
import hashlib,sys
expected={
    sys.argv[1]:"979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199",
    sys.argv[2]:"fea15cb71b278b9d003337d200f3796ccbbe8a59106287a8ea85abb2c89cb0df",
}
for p,e in expected.items():
    h=hashlib.sha256(open(p,"rb").read()).hexdigest()
    if h != e:
        raise SystemExit(f"[ERROR] SHA mismatch {p}: {h} != {e}")
    print("[checkpoint-ok]",p,h,flush=True)
PY

HEAD=$(git rev-parse HEAD)
JOB=$(sbatch --parsable \
  --partition=GPU \
  --nodelist=3090node2 \
  --nodes=1 --ntasks=1 --gres=gpu:3090:1 --cpus-per-task=8 --time=01:00:00 \
  --job-name=h9_r1_rt_tgt \
  --chdir="$REPO" \
  --output="$ROOT/logs/r1_runtime_targeted_%j.out" \
  --error="$ROOT/logs/r1_runtime_targeted_%j.out" \
  --export="ALL,R012_CODE_REPO=$REPO,R012_CODE_SHA=$HEAD,R012_ROOT=$ROOT,V1_CHECKPOINT=$V1_CHECKPOINT,SCALAR_CHECKPOINT=$SCALAR_CHECKPOINT" \
  "$REPO/experiments/hierarchical9_scratch50k_r012_v1/runtime_targeted_worker.sbatch")
JOB=${JOB%%;*}
printf 'R012_RUNTIME_TARGETED_JOB=%s\n' "$JOB" >> "$ROOT/r012_jobs.env"
echo "[submitted] runtime targeted job=$JOB node=3090node2 gpu=1"
echo "[artifact] V1=$V1_CHECKPOINT"
echo "[artifact] scalar=$SCALAR_CHECKPOINT"
echo "tail --retry -n 160 -F $ROOT/logs/r1_runtime_targeted_${JOB}.out"

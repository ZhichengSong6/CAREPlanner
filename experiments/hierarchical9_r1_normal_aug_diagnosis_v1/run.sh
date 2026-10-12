#!/usr/bin/env bash
# CPU-only source-safe diagnosis of the ALREADY COMPLETED fixed solver comparison.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
MODE="${1:-diagnose}"
PY="${VIS_PYTHON:-$HOME/miniforge3/envs/viscdf/bin/python}"
R012="${R1D_R012:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1}"
AUG="${R1D_AUG:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_r1_normal_aug_v1}"
SOLVER="${R1D_SOLVER:-$AUG/evaluation_fixed_1963}"
STARTS="${R1D_STARTS:-$R012/evaluation_fresh_holdout_v1/starts.jsonl}"
CACHE="${R1D_CACHE:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_v4_scratch_v1/training_cache}"
ROOT="${R1D_ROOT:-/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_normal_aug_diagnosis_v1}"

[[ -x "$PY" ]] || { echo "[STOP] viscdf Python not found: $PY"; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export R1A_REPO="$REPO"

usage() {
  echo "Usage: bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh diagnose|fast|test|summary|pack"
  echo "  diagnose: CPU-only existing 1963-case + training metrics + same-point frozen-model boundary gradients/ReLUs"
  echo "  fast: same except skip frozen-model same-point CPU probe"
  echo "  test: CPU unit tests only"
}

if [[ "$MODE" == test ]]; then
  "$PY" -m unittest discover -s "$HERE" -p 'test_*.py' -v
  exit 0
fi

if [[ "$MODE" == summary || "$MODE" == pack ]]; then
  LATEST="$ROOT/latest.env"
  [[ -f "$LATEST" ]] || { echo "[STOP] no completed diagnostic yet"; exit 2; }
  # Environment file is authored by printf %q below. No user-supplied value is eval'd.
  source "$LATEST"
  [[ -f "$RESULT/manifest.json" && -f "$RESULT/summary.md" ]] || {
    echo "[STOP] previous diagnostic incomplete"; exit 2;
  }
  if [[ "$MODE" == summary ]]; then
    cat "$RESULT/summary.md"
  else
    DEST="$ROOT/review_$(date +%Y%m%d_%H%M%S)_${BASHPID}.tar.gz"
    [[ ! -e "$DEST" ]] || { echo "[STOP] archive exists"; exit 2; }
    tar -czf "$DEST" -C "$RESULT" .
    tar -tzf "$DEST" >/dev/null
    sha256sum "$DEST"
    echo "[review archive] $DEST"
    echo "[note] archive contains diagnostics only; no weights, old data, or checkpoints."
  fi
  exit 0
fi

[[ "$MODE" == diagnose || "$MODE" == fast ]] || { usage; exit 2; }
# Explicit lock guards against racing diagnostics writing latest.env.
mkdir -p "$ROOT"
exec 9>"$ROOT/.diagnose.lock"
flock -n 9 || { echo "[STOP] another diagnosis is running"; exit 2; }
# Fail before writing new run products if worktree is unsafe/accidentally changed.
git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || {
  echo "[STOP] tracked source changed; commit changes or restore before diagnostic"; exit 2;
}
"$PY" - "$HERE" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
for name in ("diagnose.py","boundary_cpu.py","test_diagnosis.py"):
    compile((p/name).read_bytes(),str(p/name),"exec")
print("[syntax] three Python modules PASS",flush=True)
PY
for f in "$HERE"/*.sh; do bash -n "$f"; done
"$PY" -m unittest discover -s "$HERE" -p 'test_*.py' -v
OUT="$ROOT/run_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
ARGS=(--r012 "$R012" --aug "$AUG" --solver "$SOLVER" --starts "$STARTS" --out "$OUT")
if [[ "$MODE" == diagnose ]]; then
  ARGS+=(--with-models --repo "$REPO" --cache "$CACHE")
fi
echo "[start] CPU only; NO Slurm jobs, NO optimizer steps, NO solver reruns"
echo "[inputs] solver=$SOLVER"
echo "[inputs] R1=$R012/formal/R1 augmented=$AUG/formal"
"$PY" -u "$HERE/diagnose.py" "${ARGS[@]}"
[[ -f "$OUT/manifest.json" && -f "$OUT/summary.md" ]] || {
  echo "[STOP] missing complete outputs; latest pointer unchanged"; exit 2;
}
STATE="$ROOT/latest.env"
TMP="$ROOT/.latest.${BASHPID}"
printf 'export RESULT=%q\n' "$OUT" > "$TMP"
mv "$TMP" "$STATE"
echo "[done] read-only original inputs, saved analysis=$OUT"
echo "bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh summary"
echo "bash experiments/hierarchical9_r1_normal_aug_diagnosis_v1/run.sh pack"

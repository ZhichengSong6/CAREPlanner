#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
GPU_ENV="${GPU_ENV:-viscdf}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt}"
OUT_DIR="${OUT_DIR:-${REPO}/outputs/c5_2_signed_cdf_gpu_benchmark}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/signed_cdf_gpu_latency.json}"

WARMUP="${WARMUP:-20}"
REPEATS="${REPEATS:-100}"
CPU_REPEATS="${CPU_REPEATS:-10}"

cd "${REPO}"

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 2
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] signed checkpoint not found:"
  echo "  ${CHECKPOINT}"
  exit 3
fi

mkdir -p "${OUT_DIR}"
rm -f "${OUT_JSON}"

echo "[C5.2 GPU BENCH] branch: $(git branch --show-current)"
echo "[C5.2 GPU BENCH] head:   $(git rev-parse HEAD)"
echo "[C5.2 GPU BENCH] env:    ${GPU_ENV}"
echo "[C5.2 GPU BENCH] ckpt:   ${CHECKPOINT}"
echo "[C5.2 GPU BENCH] output: ${OUT_JSON}"
echo ""

echo "[C5.2 GPU BENCH] CUDA preflight..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  python - <<'PY'
import sys
import torch
print('[preflight] python:', sys.executable)
print('[preflight] torch:', torch.__version__)
print('[preflight] torch CUDA runtime:', torch.version.cuda)
print('[preflight] cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        '[ERROR] This conda environment does not have usable CUDA PyTorch. '
        'Set GPU_ENV to an existing CUDA-capable environment.'
    )
print('[preflight] GPU:', torch.cuda.get_device_name(0))
print('[preflight] VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
"

echo ""
echo "[C5.2 GPU BENCH] syntax check..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  python -m py_compile \
    src/care_collision_cdf/scripts/collision_cdf_model.py \
    src/care_collision_cdf/scripts/benchmark_collision_cdf_pairs_gpu.py
"

echo ""
echo "[C5.2 GPU BENCH] running forward + q-gradient benchmark..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  exec python -u src/care_collision_cdf/scripts/benchmark_collision_cdf_pairs_gpu.py \
    --checkpoint '${CHECKPOINT}' \
    --activation gelu \
    --batch-sizes 50 100 200 500 1000 1650 2000 3000 5000 \
    --warmup '${WARMUP}' \
    --repeats '${REPEATS}' \
    --cpu-reference \
    --cpu-repeats '${CPU_REPEATS}' \
    --output '${OUT_JSON}'
" | tee "${OUT_DIR}/benchmark_stdout.txt"

# Compare against the latest online C5.2 CPU shadow timing if that file exists.
ONLINE_SUMMARY="${REPO}/outputs/phase_c5_2_multi_step_forbidden_space_cdf_shadow/case_003/c5_2_multi_step_forbidden_space_cdf_summary.json"
python3 - "${OUT_JSON}" "${ONLINE_SUMMARY}" "${OUT_DIR}/comparison_summary.json" <<'PY'
import json
import os
import sys

bench_path, online_path, out_path = sys.argv[1:4]
bench = json.load(open(bench_path))

ref = min(
    bench["results"],
    key=lambda row: abs(int(row["batch_size"]) - 1650),
)

summary = {
    "gpu_name": bench.get("gpu_name"),
    "reference_batch_size": ref["batch_size"],
    "gpu_host_input_median_ms": ref["gpu_host_input"]["median_ms"],
    "gpu_host_input_p95_ms": ref["gpu_host_input"]["p95_ms"],
    "gpu_resident_input_median_ms": ref["gpu_resident_input"]["median_ms"],
    "gpu_resident_input_p95_ms": ref["gpu_resident_input"]["p95_ms"],
    "gpu_output_d2h_median_ms": ref["gpu_host_input"]["output_d2h"]["median_ms"],
    "same_env_cpu_median_ms": (
        ref.get("cpu", {}).get("median_ms")
        if isinstance(ref.get("cpu"), dict) else None
    ),
    "same_env_cpu_over_gpu_host_speedup": ref.get(
        "speedup_cpu_over_gpu_host_median"
    ),
}

if os.path.isfile(online_path):
    online = json.load(open(online_path))
    online_stats = online.get("cdf_inference_ms")
    if isinstance(online_stats, dict):
        online_median = online_stats.get("median")
        online_p95 = online_stats.get("p95")
        summary["previous_online_cpu_cdf_median_ms"] = online_median
        summary["previous_online_cpu_cdf_p95_ms"] = online_p95
        if isinstance(online_median, (int, float)):
            summary["previous_online_cpu_over_gpu_host_speedup"] = (
                float(online_median)
                / float(summary["gpu_host_input_median_ms"])
            )

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print("")
print("========== C5.2 GPU COMPARISON ==========")
print(json.dumps(summary, indent=2))
print("=========================================")
PY

echo ""
echo "[C5.2 GPU BENCH COMPLETE]"
echo "[FULL]    ${OUT_JSON}"
echo "[COMPARE] ${OUT_DIR}/comparison_summary.json"

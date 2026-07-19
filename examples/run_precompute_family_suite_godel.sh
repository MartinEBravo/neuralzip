#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/mbravo/neuralzip-transfer-20260719/Codebase}
OUTPUT_DIR=${OUTPUT_DIR:-/data/mbravo/neuralzip-transfer-20260719/results}
HF_HUB_CACHE=${HF_HUB_CACHE:-/data/mbravo/neuralzip-transfer-20260719/hf-cache}
TORCH_HOME=${TORCH_HOME:-/data/mbravo/neuralzip-transfer-20260719/torch}
PYTHON=${PYTHON:-/dev/shm/mbravo-neuralzip-final-family-20260718/venv/bin/python}
THREADS=${THREADS:-12}
CPU_SET=${CPU_SET:-0-11}

export HF_HUB_CACHE TORCH_HOME
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

families=(
  resnet-basic
  resnet-bottleneck
  efficientnet
  convnext
  llama32
  siglip2
  whisper-v3
  gemma3
  olmo2
  qwen25
  qwen35
)

status=0
for family in "${families[@]}"; do
  output="${OUTPUT_DIR}/${family}.json"
  log="${OUTPUT_DIR}/${family}.log"
  if [[ -s "${output}" ]]; then
    if "${PYTHON}" -c \
      'import json,sys; from examples.run_precompute_family_suite import dependency_sha256,runner_sha256; p=json.load(open(sys.argv[1])); ok=p.get("complete") is True and p.get("runner_sha256")==runner_sha256() and p.get("dependency_sha256")==dependency_sha256() and int(p.get("threads",-1))==int(sys.argv[2]); raise SystemExit(0 if ok else 1)' \
      "${output}" "${THREADS}"; then
      continue
    fi
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    mv "${output}" "${OUTPUT_DIR}/${family}.partial-${stamp}.json"
    if [[ -s "${log}" ]]; then
      mv "${log}" "${OUTPUT_DIR}/${family}.partial-${stamp}.log"
    fi
  fi
  if ! "${PYTHON}" examples/run_precompute_family_suite.py \
      --family "${family}" \
      --cache-dir "${HF_HUB_CACHE}" \
      --output "${output}" \
      --threads "${THREADS}" \
      --cpu-set "${CPU_SET}" \
      --allow-download \
      >"${log}" 2>&1; then
    status=1
  fi
done

exit "${status}"

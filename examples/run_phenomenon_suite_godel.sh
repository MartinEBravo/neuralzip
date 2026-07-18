#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/tmp/neuralzip-phenomenon-suite-20260717/Codebase}"
OUTPUT_ROOT="${2:-/tmp/neuralzip-phenomenon-suite-20260717/outputs}"
PYTHON="/home/mbravo/bachelor-thesis-code/.venv/bin/python"
MMREPAIR="/home/mbravo/neuralzip-runs/thesis-suite-20260716/mm-repair"

export PYTHONPATH="${ROOT}:${ROOT}/examples"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEURALZIP_TURBORC_LIB="/home/mbravo/neuralzip-runs/thesis-suite-20260716/turbo-range-coder/libturborc_min.so"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

mkdir -p "${OUTPUT_ROOT}/suite" "${OUTPUT_ROOT}/clustering"

for model in gpt2 bert-base resnet18 resnet34; do
    for phase in diagnostics component deferred zipnn; do
        case "${phase}" in
            diagnostics) expected="${OUTPUT_ROOT}/suite/${model}/diagnostics.json" ;;
            component) expected="${OUTPUT_ROOT}/suite/${model}/component_summary.csv" ;;
            deferred) expected="${OUTPUT_ROOT}/suite/${model}/deferred.csv" ;;
            zipnn) expected="${OUTPUT_ROOT}/suite/${model}/zipnn_implementation.csv" ;;
        esac
        force_phase=0
        if [[ "${phase}" == "component" && "${FORCE_COMPONENT:-0}" == "1" ]]; then
            force_phase=1
        fi
        if [[ "${phase}" == "deferred" && "${FORCE_DEFERRED:-0}" == "1" ]]; then
            force_phase=1
        fi
        if [[ -f "${expected}" ]]; then
            if [[ "${force_phase}" != "1" ]]; then
                continue
            fi
        fi
        "${PYTHON}" "${ROOT}/examples/run_thesis_model_suite.py" \
            --model "${model}" \
            --phase "${phase}" \
            --threads 12 \
            --mmrepair-dir "${MMREPAIR}" \
            --output-dir "${OUTPUT_ROOT}/suite"
    done
done

"${PYTHON}" "${ROOT}/examples/run_thesis_model_suite.py" \
    --phase render \
    --threads 12 \
    --output-dir "${OUTPUT_ROOT}/suite"

if [[ ! -f "${OUTPUT_ROOT}/clustering/model_summary.csv" ]]; then
    "${PYTHON}" "${ROOT}/examples/run_js_cluster_suite_probe.py" \
        --model-root /tmp/neuralzip-phenomenon-suite-20260717/model-root \
        --output-dir "${OUTPUT_ROOT}/clustering" \
        --models all
fi

if [[ ! -f "${OUTPUT_ROOT}/precomputed_suite.json" ]]; then
    "${PYTHON}" "${ROOT}/examples/benchmark_precomputed_suite.py" \
        --model-root /tmp/neuralzip-phenomenon-suite-20260717/model-root \
        --threads 12 \
        --output "${OUTPUT_ROOT}/precomputed_suite.json"
fi

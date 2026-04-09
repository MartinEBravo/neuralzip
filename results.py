import csv
import json
from collections import defaultdict
from pathlib import Path

from constants import MODEL_PRESETS, get_model_label


def format_method_variant(method: str, prequant_bits: int | None) -> str:
    if prequant_bits is None:
        return method
    return f"Q{prequant_bits}->{method}"


def format_case_label(label: str, prequant_bits: int | None) -> str:
    if prequant_bits is None:
        return label
    return f"{label} Q{prequant_bits}"


def format_row(
    model_name: str,
    method: str,
    bits: int,
    original_bits: int,
    loader: str,
    dtype_summary: str,
    num_blocks: int,
    elapsed_seconds: float,
    prequant_bits: int | None = None,
) -> dict[str, object]:
    return {
        "model": model_name,
        "label": get_model_label(model_name),
        "loader": loader,
        "dtype_summary": dtype_summary,
        "num_blocks": num_blocks,
        "method": method,
        "variant": format_method_variant(method, prequant_bits),
        "prequant_bits": "" if prequant_bits is None else int(prequant_bits),
        "total_bits": int(bits),
        "mbits": bits / 1e6,
        "mbytes": bits / 8e6,
        "percent_original": 100.0 * bits / original_bits,
        "elapsed_seconds": elapsed_seconds,
    }


def save_csv_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_square_matrix_csv(path: Path, labels: list[str], matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block", *labels])
        for idx, label in enumerate(labels):
            writer.writerow([label, *[f"{value:.8f}" for value in matrix[idx]]])


def write_summary_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def print_summary(rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['model']}|{row.get('prequant_bits', '')}"].append(row)

    for _, model_rows in grouped.items():
        sample = model_rows[0]
        case_label = format_case_label(
            str(sample["label"]),
            None if sample.get("prequant_bits", "") == "" else int(sample["prequant_bits"]),
        )
        print(
            f"{case_label} [{sample['loader']}, {sample['dtype_summary']}, "
            f"{sample['num_blocks']} blocks]"
        )
        for row in model_rows:
            print(
                f"  {row['variant']:20s} {row['mbits']:10.2f} Mbits  "
                f"{row['percent_original']:6.2f}%  {float(row['elapsed_seconds']):8.3f}s"
            )


def print_pipeline_summary(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            f"{row['label']} [{row['loader']}, {row['dtype_summary']}, {row['num_blocks']} blocks]"
        )
        print(
            "  H0(sign/exponent/mantissa): "
            f"{float(row['sign_h0_bits_per_symbol']):.6f} / "
            f"{float(row['exponent_h0_bits_per_symbol']):.6f} / "
            f"{float(row['mantissa_h0_bits_per_symbol']):.6f}"
        )
        print(
            "  Exponent JS: "
            f"avg {float(row['average_exponent_js']):.6f}, "
            f"median {float(row['median_exponent_js']):.6f}, "
            f"max {float(row['max_exponent_js']):.6f}, "
            f"clusters {int(row['num_clusters'])}"
        )


def print_presets() -> None:
    print("Available presets:")
    for preset_name in sorted(MODEL_PRESETS):
        print(f"  {preset_name}")
        for model_name in MODEL_PRESETS[preset_name]:
            print(f"    - {model_name}")

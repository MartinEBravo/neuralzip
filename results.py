import csv
from collections import defaultdict
from pathlib import Path

from constants import MODEL_LABELS, MODEL_PRESETS


def format_row(
    model_name: str,
    method: str,
    bits: int,
    original_bits: int,
    loader: str,
    dtype_summary: str,
    num_tensors: int,
) -> dict[str, object]:
    return {
        "model": model_name,
        "label": MODEL_LABELS.get(model_name, model_name),
        "loader": loader,
        "dtype_summary": dtype_summary,
        "num_tensors": num_tensors,
        "method": method,
        "total_bits": int(bits),
        "mbits": bits / 1e6,
        "mbytes": bits / 8e6,
        "percent_original": 100.0 * bits / original_bits,
    }


def save_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    for model_name, model_rows in grouped.items():
        del model_name
        sample = model_rows[0]
        print(
            f"{sample['label']} [{sample['loader']}, {sample['dtype_summary']}, "
            f"{sample['num_tensors']} tensors]"
        )
        for row in model_rows:
            print(
                f"  {row['method']:8s} {row['mbits']:10.2f} Mbits  "
                f"{row['percent_original']:6.2f}%"
            )


def print_presets() -> None:
    print("Available presets:")
    for preset_name in sorted(MODEL_PRESETS):
        print(f"  {preset_name}")
        for model_name in MODEL_PRESETS[preset_name]:
            print(f"    - {model_name}")

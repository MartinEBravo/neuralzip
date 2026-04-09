import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _method_color(method: str) -> str:
    if method.startswith("NeuralZip"):
        return "#F58518"
    return "#4C78A8"


def _case_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["model"]), str(row.get("prequant_bits", ""))


def _case_label(row: dict[str, object]) -> str:
    bits = row.get("prequant_bits", "")
    if bits in ("", None):
        return str(row["label"])
    return f"{row['label']} Q{bits}"


def _bar_chart(
    cases: list[tuple[str, str]],
    case_labels: dict[tuple[str, str], str],
    series: list[tuple[str, str, list[float]]],
    ylabel: str,
    path,
    max_width: float = 0.24,
) -> None:
    x = np.arange(len(cases))
    width = min(0.75 / max(len(series), 1), max_width)
    fig_width = max(9.0, 1.1 * len(cases) + 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    offset_origin = (len(series) - 1) / 2
    for i, (label, color, values) in enumerate(series):
        ax.bar(x + (i - offset_origin) * width, values, width=width, label=label, color=color)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([case_labels[c] for c in cases], rotation=18, ha="right")
    ax.legend(frameon=False, ncols=min(3, len(series)))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ratio_bars(rows: list[dict[str, object]], path) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    cases = list(dict.fromkeys(_case_key(row) for row in rows))
    labels = {_case_key(row): _case_label(row) for row in rows}

    series = [
        (
            method,
            _method_color(method),
            [
                next(r["percent_original"] for r in rows if _case_key(r) == case and str(r["method"]) == method)
                for case in cases
            ],
        )
        for method in methods
    ]
    _bar_chart(cases, labels, series, "% of Original Size", path)


def plot_gain_bars(rows: list[dict[str, object]], path) -> None:
    if "ZipNN" not in {str(row["method"]) for row in rows}:
        return

    cases = list(dict.fromkeys(_case_key(row) for row in rows))
    labels = {_case_key(row): _case_label(row) for row in rows}
    zipnn = {
        _case_key(row): row["percent_original"]
        for row in rows
        if str(row["method"]) == "ZipNN"
    }
    gain_methods = [method for method in dict.fromkeys(str(row["method"]) for row in rows) if method != "ZipNN"]
    if not gain_methods:
        return

    series: list[tuple[str, str, list[float]]] = []
    for method in gain_methods:
        method_gains: list[float] = []
        missing_case = False
        for case in cases:
            if case not in zipnn:
                missing_case = True
                break
            match = next((r for r in rows if _case_key(r) == case and str(r["method"]) == method), None)
            if match is None:
                missing_case = True
                break
            method_gains.append(zipnn[case] - float(match["percent_original"]))
        if not missing_case:
            series.append((f"ZipNN to {method}", _method_color(method), method_gains))

    if series:
        _bar_chart(cases, labels, series, "Reduction in Percentage Points", path, max_width=0.32)


def plot_js_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    path,
    title: str,
    max_tick_labels: int = 60,
) -> None:
    size = len(labels)
    if size == 0:
        raise RuntimeError("Cannot plot an empty JS matrix.")

    max_value = float(matrix.max()) if matrix.size else 0.0
    fig_extent = min(max(7.5, 0.22 * size + 2.5), 28.0)
    fig, ax = plt.subplots(figsize=(fig_extent, fig_extent))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    display_matrix = np.ma.array(matrix, mask=mask)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(display_matrix, cmap=cmap, vmin=0.0, vmax=max_value if max_value > 0 else 1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Jensen-Shannon Distance")

    stride = max(1, int(np.ceil(size / max_tick_labels)))
    tick_positions = list(range(0, size, stride))
    tick_labels = [labels[idx] for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Blocks")
    ax.set_ylabel("Blocks")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_ratio_bars(rows: list[dict[str, object]], path) -> None:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    labels = {row["model"]: row["label"] for row in rows}
    colors = {
        "ZipNN": "#4C78A8",
        "ZipNN+": "#F58518",
        "ZipNN++": "#54A24B",
    }

    data = {
        method: [
            next(r["percent_original"] for r in rows if r["model"] == model and r["method"] == method)
            for model in models
        ]
        for method in methods
    }

    x = np.arange(len(models))
    width = min(0.75 / max(len(methods), 1), 0.24)
    fig_width = max(9.0, 1.1 * len(models) + 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    offset_origin = (len(methods) - 1) / 2
    for i, method in enumerate(methods):
        ax.bar(
            x + (i - offset_origin) * width,
            data[method],
            width=width,
            label=method,
            color=colors.get(method, "#4C78A8"),
        )

    ax.set_ylabel("% of Original Size")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[model] for model in models], rotation=18, ha="right")
    ax.set_ylim(0, max(max(values) for values in data.values()) * 1.15)
    ax.legend(frameon=False, ncols=min(3, len(methods)))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gain_bars(rows: list[dict[str, object]], path) -> None:
    if "ZipNN" not in {row["method"] for row in rows}:
        return

    models = list(dict.fromkeys(row["model"] for row in rows))
    labels = {row["model"]: row["label"] for row in rows}
    zipnn = {row["model"]: row["percent_original"] for row in rows if row["method"] == "ZipNN"}
    gain_methods = [method for method in dict.fromkeys(row["method"] for row in rows) if method != "ZipNN"]
    if not gain_methods:
        return

    gains = {
        method: [
            zipnn[model] - next(r["percent_original"] for r in rows if r["model"] == model and r["method"] == method)
            for model in models
        ]
        for method in gain_methods
    }

    x = np.arange(len(models))
    width = min(0.8 / max(len(gain_methods), 1), 0.32)
    fig_width = max(9.0, 1.1 * len(models) + 3.0)
    colors = {
        "ZipNN+": "#F58518",
        "ZipNN++": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    offset_origin = (len(gain_methods) - 1) / 2
    for i, method in enumerate(gain_methods):
        ax.bar(
            x + (i - offset_origin) * width,
            gains[method],
            width=width,
            label=f"ZipNN to {method}",
            color=colors.get(method, "#4C78A8"),
        )

    ax.set_ylabel("Reduction in Percentage Points")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[model] for model in models], rotation=18, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

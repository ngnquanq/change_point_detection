#!/usr/bin/env python
"""Plot the canonical synthetic comparison used by the report.

This figure is intentionally derived from the same saved evaluation artifacts
that back the README/report tables, so the chart and tables always tell the
same story.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output" / "comparison"
OUTPUT_FIGURE = OUTPUT_DIR / "figure2_comparison.png"
OUTPUT_JSON = OUTPUT_DIR / "comparison_results.json"

SCENARIOS = ["S1", "S1'", "S2", "S3"]
METHOD_ORDER = ["CUSUM", "AutoCPD MLP", "PyTorch MLP", "ResCNN"]
METHOD_COLORS = {
    "CUSUM": "#2c3e50",
    "AutoCPD MLP": "#7f8c8d",
    "PyTorch MLP": "#3498db",
    "ResCNN": "#e74c3c",
}

EXPERIMENT_PATHS = {
    "S1": {
        "PyTorch MLP": REPO_ROOT / "models" / "mlp_s1" / "eval_results.json",
        "ResCNN": REPO_ROOT / "models" / "rescnn_s1_paper" / "eval_results.json",
        "AutoCPD MLP": REPO_ROOT / "models" / "autocpd_s1_paper" / "eval_results.json",
    },
    "S1'": {
        "PyTorch MLP": REPO_ROOT / "models" / "mlp_s1prime" / "eval_results.json",
        "ResCNN": REPO_ROOT / "models" / "rescnn_s1prime_paper" / "eval_results.json",
        "AutoCPD MLP": REPO_ROOT / "models" / "autocpd_s1prime_paper" / "eval_results.json",
    },
    "S2": {
        "PyTorch MLP": REPO_ROOT / "models" / "mlp_s2" / "eval_results.json",
        "ResCNN": REPO_ROOT / "models" / "rescnn_s2_paper" / "eval_results.json",
        "AutoCPD MLP": REPO_ROOT / "models" / "autocpd_s2_paper" / "eval_results.json",
    },
    "S3": {
        "PyTorch MLP": REPO_ROOT / "models" / "mlp_s3" / "eval_results.json",
        "ResCNN": REPO_ROOT / "models" / "rescnn_s3_paper" / "eval_results.json",
        "AutoCPD MLP": REPO_ROOT / "models" / "autocpd_s3_paper" / "eval_results.json",
    },
}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def extract_metrics() -> dict:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIOS:
        scenario_results: dict[str, dict[str, float]] = {}

        mlp = load_json(EXPERIMENT_PATHS[scenario]["PyTorch MLP"])
        rescnn = load_json(EXPERIMENT_PATHS[scenario]["ResCNN"])
        autocpd = load_json(EXPERIMENT_PATHS[scenario]["AutoCPD MLP"])

        scenario_results["PyTorch MLP"] = {
            "accuracy": mlp["nn"]["detection_accuracy"],
            "power": mlp["nn"]["power"],
            "fpr": mlp["nn"]["type1_error"],
        }
        scenario_results["ResCNN"] = {
            "accuracy": rescnn["nn"]["detection_accuracy"],
            "power": rescnn["nn"]["power"],
            "fpr": rescnn["nn"]["type1_error"],
        }
        scenario_results["CUSUM"] = {
            "accuracy": mlp["cusum"]["detection_accuracy"],
            "power": mlp["cusum"]["power"],
            "fpr": mlp["cusum"]["type1_error"],
        }
        scenario_results["AutoCPD MLP"] = {
            "accuracy": autocpd["metrics"]["detection_accuracy"],
            "power": autocpd["metrics"]["power"],
            "fpr": autocpd["metrics"]["type1_error"],
        }

        results[scenario] = scenario_results
    return results


def plot_results(results: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("accuracy", "Detection Accuracy", "higher is better"),
        ("power", "Power", "higher is better"),
        ("fpr", "False Positive Rate", "lower is better"),
    ]

    x = np.arange(len(SCENARIOS))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=False)
    fig.suptitle(
        "Canonical Synthetic Results on paper_faithful Test Splits",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    for ax, (metric_key, title, subtitle) in zip(axes, metrics):
        for idx, method in enumerate(METHOD_ORDER):
            values = [results[scenario][method][metric_key] for scenario in SCENARIOS]
            offset = (idx - 1.5) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=method,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    fig.legend(
        METHOD_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.76, bottom=0.16, wspace=0.18)
    plt.savefig(OUTPUT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = extract_metrics()
    plot_results(results)
    with OUTPUT_JSON.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"saved {OUTPUT_FIGURE}")
    print(f"saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

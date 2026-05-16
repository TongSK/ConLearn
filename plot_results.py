"""
Generate visual result charts from LODO evaluation outputs.

Run after evaluate.py has produced evaluation_metrics.json and prediction CSVs:
  python plot_results.py --results-dir results

Outputs are written to:
  results/plots/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
DISPLAY_METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def load_reports(results_dir):
    reports = []
    for path in sorted(Path(results_dir).glob("*/evaluation_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        report["_dir"] = path.parent
        reports.append(report)
    if not reports:
        raise FileNotFoundError(f"No evaluation_metrics.json files found under {results_dir}")
    return reports


def load_predictions(path):
    labels = []
    scores = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    return np.array(labels), np.array(scores)


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_metric_group_bar(reports, output_dir):
    folds = [r["held_out_source"] for r in reports]
    x = np.arange(len(folds))
    width = 0.13

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

    for i, metric in enumerate(METRICS):
        values = [r["test"][metric] for r in reports]
        ax.bar(
            x + (i - (len(METRICS) - 1) / 2) * width,
            values,
            width,
            label=DISPLAY_METRICS[metric],
            color=colors[i],
        )

    ax.set_title("LODO Test Performance by Held-Out Dataset")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "lodo_metrics_bar.png")
    plt.close(fig)


def save_precision_recall_f1(reports, output_dir):
    folds = [r["held_out_source"] for r in reports]
    x = np.arange(len(folds))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("precision", "Precision", "#4c78a8"),
        ("recall", "Recall", "#f58518"),
        ("f1", "F1", "#54a24b"),
    ]

    for i, (metric, label, color) in enumerate(series):
        values = [r["test"][metric] for r in reports]
        ax.bar(x + (i - 1) * width, values, width, label=label, color=color)

    ax.set_title("Precision, Recall, and F1 Across LODO Folds")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_f1.png")
    plt.close(fig)


def save_auc_plot(reports, output_dir):
    folds = [r["held_out_source"] for r in reports]
    roc_values = [r["test"]["roc_auc"] for r in reports]
    pr_values = [r["test"]["pr_auc"] for r in reports]

    x = np.arange(len(folds))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, roc_values, width, label="ROC-AUC", color="#4c78a8")
    ax.bar(x + width / 2, pr_values, width, label="PR-AUC", color="#e45756")

    ax.set_title("Threshold-Independent Performance")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "auc_comparison.png")
    plt.close(fig)


def save_confusion_matrices(reports, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.ravel()

    for ax, report in zip(axes, reports):
        cm = report["test"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(report["held_out_source"])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Injected"])
        ax.set_yticklabels(["Benign", "Injected"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        max_value = matrix.max()
        for row in range(2):
            for col in range(2):
                color = "white" if matrix[row, col] > max_value * 0.55 else "black"
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.colorbar(image, ax=axes.tolist(), shrink=0.75)
    fig.suptitle("Held-Out Test Confusion Matrices", y=0.98)
    fig.savefig(output_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(reports, output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for report in reports:
        pred_path = report["_dir"] / "test_predictions.csv"
        labels, scores = load_predictions(pred_path)
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{report['held_out_source']} ({roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_title("ROC Curves on Held-Out Test Folds")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png")
    plt.close(fig)


def save_pr_curves(reports, output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for report in reports:
        pred_path = report["_dir"] / "test_predictions.csv"
        labels, scores = load_predictions(pred_path)
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{report['held_out_source']} ({pr_auc:.3f})")

    ax.set_title("Precision-Recall Curves on Held-Out Test Folds")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png")
    plt.close(fig)


def main(results_dir):
    results_dir = Path(results_dir)
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    reports = load_reports(results_dir)

    save_metric_group_bar(reports, output_dir)
    save_precision_recall_f1(reports, output_dir)
    save_auc_plot(reports, output_dir)
    save_confusion_matrices(reports, output_dir)
    save_roc_curves(reports, output_dir)
    save_pr_curves(reports, output_dir)

    print(f"Saved plots to: {output_dir}")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)

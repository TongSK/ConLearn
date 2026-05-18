"""
Summarise LODO evaluation results into CSV and Markdown tables.

Run after evaluating all folds:
  python src/summarize_results.py --results-dir results
"""

import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


def load_reports(results_dir):
    reports = []
    for path in sorted(Path(results_dir).glob("*/evaluation_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        report["_path"] = path
        reports.append(report)
    return reports


def flatten_report(report):
    test = report["test"]
    cm = test["confusion_matrix"]
    row = {
        "held_out_source": report["held_out_source"],
        "model_name": report.get("model_name", report.get("model_type", "")),
        "threshold_source": report.get("threshold_source", ""),
        "threshold": test.get("threshold"),
    }
    for field in METRIC_FIELDS:
        row[field] = test.get(field)
    row.update(cm)
    return row


def mean_row(rows):
    numeric_fields = ["threshold", *METRIC_FIELDS, "tn", "fp", "fn", "tp"]
    row = {
        "held_out_source": "MEAN",
        "model_name": "",
        "threshold_source": "",
    }
    for field in numeric_fields:
        values = [r[field] for r in rows if r.get(field) is not None]
        row[field] = sum(values) / len(values) if values else None
    return row


def write_csv(path, rows):
    fields = [
        "held_out_source",
        "model_name",
        "threshold_source",
        "threshold",
        *METRIC_FIELDS,
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown(path, rows):
    fields = [
        "held_out_source",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |\n")


def main(results_dir):
    results_dir = Path(results_dir)
    reports = load_reports(results_dir)
    if not reports:
        raise FileNotFoundError(f"No evaluation_metrics.json files found under {results_dir}")

    rows = [flatten_report(report) for report in reports]
    rows.append(mean_row(rows))

    csv_path = results_dir / "lodo_summary.csv"
    md_path = results_dir / "lodo_summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"Saved CSV     : {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print("\nSummary:")
    for row in rows:
        print(
            f"{row['held_out_source']:12s} "
            f"F1={fmt(row['f1'])} "
            f"ROC-AUC={fmt(row['roc_auc'])} "
            f"PR-AUC={fmt(row['pr_auc'])} "
            f"P={fmt(row['precision'])} "
            f"R={fmt(row['recall'])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)

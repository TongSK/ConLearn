"""
TF-IDF + Logistic Regression baseline for LODO prompt-injection detection.

This baseline is intentionally simple and fast:
  1. Use the same LODO + internal validation split as the contrastive model.
  2. Train TF-IDF features with Logistic Regression on non-held-out sources.
  3. Tune threshold on validation F1.
  4. Report final metrics on the held-out source.

Run one fold:
  python src/baseline_tfidf.py --csv data/dataset.csv --held-out deepset --output results_baseline/deepset

Run all folds:
  python src/baseline_tfidf.py --csv data/dataset.csv --all --output results_baseline
"""

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data_loader import VALID_SOURCES, _lodo_split, _stratified_train_val_split


def choose_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    f1_values = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx])


def compute_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }, predictions


def save_predictions(path, labels, scores, predictions):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "score", "prediction"])
        for label, score, prediction in zip(labels, scores, predictions):
            writer.writerow([int(label), f"{float(score):.8f}", int(prediction)])


def load_splits(csv_path, held_out_source, val_fraction, seed):
    df = pd.read_csv(csv_path)
    required = {"text", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv is missing columns: {missing}")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    train_pool_df, test_df = _lodo_split(df, held_out_source)
    train_df, val_df = _stratified_train_val_split(train_pool_df, val_fraction, seed)
    return train_df, val_df, test_df


def evaluate_fold(
    csv_path,
    held_out_source,
    output_dir,
    val_fraction=0.10,
    seed=42,
    max_features=50000,
):
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = load_splits(csv_path, held_out_source, val_fraction, seed)

    print(f"\n=== TF-IDF baseline - held out: '{held_out_source}' ===")
    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")
    print(f"Output: {output_dir}\n")

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )

    pipeline.fit(train_df["text"], train_df["label"])

    val_scores = pipeline.predict_proba(val_df["text"])[:, 1]
    test_scores = pipeline.predict_proba(test_df["text"])[:, 1]
    val_labels = val_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()

    threshold = choose_threshold(val_labels, val_scores)
    val_metrics, val_predictions = compute_metrics(val_labels, val_scores, threshold)
    test_metrics, test_predictions = compute_metrics(test_labels, test_scores, threshold)

    report = {
        "held_out_source": held_out_source,
        "model_type": "tfidf_logistic_regression",
        "threshold_source": "validation_f1",
        "max_features": max_features,
        "validation": val_metrics,
        "test": test_metrics,
    }

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_predictions(
        os.path.join(output_dir, "validation_predictions.csv"),
        val_labels,
        val_scores,
        val_predictions,
    )
    save_predictions(
        os.path.join(output_dir, "test_predictions.csv"),
        test_labels,
        test_scores,
        test_predictions,
    )

    print("Held-out test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved metrics: {metrics_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dataset.csv")
    parser.add_argument("--held-out", choices=sorted(VALID_SOURCES), default=None)
    parser.add_argument("--all", action="store_true", help="Run all LODO folds")
    parser.add_argument("--output", default="results_baseline")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    args = parser.parse_args()

    if not args.all and args.held_out is None:
        raise ValueError("Use --held-out <source> or --all")

    folds = sorted(VALID_SOURCES) if args.all else [args.held_out]
    for fold in folds:
        output_dir = os.path.join(args.output, fold) if args.all else args.output
        evaluate_fold(
            csv_path=args.csv,
            held_out_source=fold,
            output_dir=output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_features=args.max_features,
        )


if __name__ == "__main__":
    main()

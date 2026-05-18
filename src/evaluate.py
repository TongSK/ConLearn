"""
Evaluate a trained contrastive prompt-injection detector.

Inference protocol:
  1. Load the best trained encoder for one LODO fold.
  2. Embed the non-held-out training split and build class centroids.
  3. Tune a decision threshold on the internal validation split.
  4. Report final metrics on the held-out LODO test source.

This keeps the held-out source clean: it is used only for final evaluation.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
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
from tqdm import tqdm

from data_loader import get_lodo_loaders
from model import PromptInjectionModel


def first_value(*values):
    """Return the first value that is not None, otherwise None."""
    for value in values:
        if value is not None:
            return value
    return None


def load_training_config(model_dir):
    checkpoint_path = os.path.join(model_dir, "checkpoint_last.pt")
    if not os.path.exists(checkpoint_path):
        return {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("config", {})


@torch.no_grad()
def collect_embeddings(model, loader, device, desc):
    model.eval()
    embeddings = []
    labels = []

    for input_ids, attention_mask, batch_labels in tqdm(loader, desc=desc, leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        batch_embeddings = model.get_embeddings(input_ids, attention_mask)
        embeddings.append(batch_embeddings.cpu())
        labels.append(batch_labels.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels


def build_centroids(embeddings, labels):
    centroids = []
    for label in [0, 1]:
        class_embeddings = embeddings[labels == label]
        if len(class_embeddings) == 0:
            raise ValueError(f"No samples found for label {label}; cannot build centroid.")
        centroids.append(class_embeddings.mean(dim=0))

    centroids = torch.stack(centroids, dim=0)
    return F.normalize(centroids, p=2, dim=1)


def score_with_centroids(embeddings, centroids):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarities = embeddings @ centroids.T
    return (similarities[:, 1] - similarities[:, 0]).numpy()


def choose_threshold(labels, scores, strategy="f1", target_precision=0.90, target_recall=0.90):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    if strategy == "precision":
        valid = np.where(precision >= target_precision)[0]
        if len(valid) > 0:
            best_idx = valid[int(np.argmax(recall[valid]))]
            return float(thresholds[best_idx])

    if strategy == "recall":
        valid = np.where(recall >= target_recall)[0]
        if len(valid) > 0:
            best_idx = valid[int(np.argmax(precision[valid]))]
            return float(thresholds[best_idx])

    f1_values = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx])


def compute_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    metrics = {
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
    }
    return metrics, predictions


def save_predictions(path, labels, scores, predictions):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "score", "prediction"])
        for label, score, prediction in zip(labels, scores, predictions):
            writer.writerow([int(label), f"{float(score):.8f}", int(prediction)])


def evaluate(
    csv_path,
    held_out_source,
    model_dir,
    output_dir=None,
    model_name=None,
    batch_size=None,
    max_length=None,
    val_fraction=None,
    threshold_strategy="f1",
    target_precision=0.90,
    target_recall=0.90,
):
    output_dir = output_dir or model_dir
    os.makedirs(output_dir, exist_ok=True)

    config = load_training_config(model_dir)
    model_name = first_value(model_name, config.get("model_name"), "roberta-base")
    batch_size = first_value(batch_size, config.get("batch_size"), 16)
    max_length = first_value(max_length, config.get("max_length"), 64)
    val_fraction = first_value(val_fraction, config.get("val_fraction"), 0.10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Evaluation - held out: '{held_out_source}' ===")
    print(f"Device : {device}")
    print(f"Encoder: {model_name}")
    print(f"Model  : {model_dir}")
    print(f"Output : {output_dir}\n")

    train_loader, val_loader, test_loader = get_lodo_loaders(
        csv_path,
        held_out_source,
        batch_size=batch_size,
        max_length=max_length,
        val_fraction=val_fraction,
        model_name=model_name,
        balanced_sampling=False,
    )

    model = PromptInjectionModel(encoder_name=model_name, freeze_encoder=False).to(device)
    encoder_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Missing trained encoder: {encoder_path}")

    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    train_embeddings, train_labels = collect_embeddings(model, train_loader, device, "Embedding train")
    val_embeddings, val_labels = collect_embeddings(model, val_loader, device, "Embedding val")
    test_embeddings, test_labels = collect_embeddings(model, test_loader, device, "Embedding test")

    centroids = build_centroids(train_embeddings, train_labels)

    val_scores = score_with_centroids(val_embeddings, centroids)
    test_scores = score_with_centroids(test_embeddings, centroids)

    val_labels_np = val_labels.numpy()
    test_labels_np = test_labels.numpy()

    threshold = choose_threshold(
        val_labels_np,
        val_scores,
        strategy=threshold_strategy,
        target_precision=target_precision,
        target_recall=target_recall,
    )
    val_metrics, val_predictions = compute_metrics(val_labels_np, val_scores, threshold)
    test_metrics, test_predictions = compute_metrics(test_labels_np, test_scores, threshold)

    report = {
        "held_out_source": held_out_source,
        "model_dir": model_dir,
        "model_name": model_name,
        "threshold_source": f"validation_{threshold_strategy}",
        "target_precision": target_precision if threshold_strategy == "precision" else None,
        "target_recall": target_recall if threshold_strategy == "recall" else None,
        "validation": val_metrics,
        "test": test_metrics,
    }

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_predictions(
        os.path.join(output_dir, "validation_predictions.csv"),
        val_labels_np,
        val_scores,
        val_predictions,
    )
    save_predictions(
        os.path.join(output_dir, "test_predictions.csv"),
        test_labels_np,
        test_scores,
        test_predictions,
    )

    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("\nHeld-out test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved metrics: {metrics_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dataset.csv")
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--threshold-strategy", choices=["f1", "precision", "recall"], default="f1")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--target-recall", type=float, default=0.90)
    args = parser.parse_args()

    evaluate(
        csv_path=args.csv,
        held_out_source=args.held_out,
        model_dir=args.model_dir,
        output_dir=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        threshold_strategy=args.threshold_strategy,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
    )

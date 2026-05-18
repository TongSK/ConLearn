"""
Export a trained fold into a lightweight detector artifact.

The artifact contains:
  - encoder weights
  - benign/injected centroids
  - validation-tuned threshold
  - model/tokenizer settings

Run:
  python export_detector.py --csv dataset.csv --held-out safeguard --model-dir results/safeguard --output detector_artifact.pt
"""

import argparse
import os

import torch

from data_loader import get_lodo_loaders
from evaluate import (
    build_centroids,
    choose_threshold,
    collect_embeddings,
    first_value,
    load_training_config,
    score_with_centroids,
)
from model import PromptInjectionModel


def export_detector(
    csv_path,
    held_out_source,
    model_dir,
    output_path,
    model_name=None,
    batch_size=None,
    max_length=None,
    val_fraction=None,
    threshold_strategy="f1",
):
    config = load_training_config(model_dir)
    model_name = first_value(model_name, config.get("model_name"), "roberta-base")
    batch_size = int(first_value(batch_size, config.get("batch_size"), 16))
    max_length = int(first_value(max_length, config.get("max_length"), 64))
    val_fraction = float(first_value(val_fraction, config.get("val_fraction"), 0.10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Export detector - held out: '{held_out_source}' ===")
    print(f"Device : {device}")
    print(f"Encoder: {model_name}")
    print(f"Model  : {model_dir}")
    print(f"Output : {output_path}\n")

    train_loader, val_loader, _ = get_lodo_loaders(
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

    centroids = build_centroids(train_embeddings, train_labels)
    val_scores = score_with_centroids(val_embeddings, centroids)
    threshold = choose_threshold(val_labels.numpy(), val_scores, strategy=threshold_strategy)

    artifact = {
        "model_name": model_name,
        "max_length": max_length,
        "threshold": float(threshold),
        "held_out_source": held_out_source,
        "encoder_state_dict": model.encoder.state_dict(),
        "centroids": centroids.cpu(),
        "labels": {0: "benign", 1: "prompt_injection"},
    }

    torch.save(artifact, output_path)
    print(f"Detector artifact saved: {output_path}")
    print(f"Threshold: {threshold:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset.csv")
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="detector_artifact.pt")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--threshold-strategy", choices=["f1", "precision", "recall"], default="f1")
    args = parser.parse_args()

    export_detector(
        csv_path=args.csv,
        held_out_source=args.held_out,
        model_dir=args.model_dir,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        threshold_strategy=args.threshold_strategy,
    )

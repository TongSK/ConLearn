"""
Step 3 — Training Loop
Prompt Injection Detection — FYP
=================================
Trains the contrastive model for one LODO fold.

Training strategy:
  Epoch 1     : encoder frozen  — only projection head trains (warmup)
  Epoch 2+    : encoder unfrozen — full model fine-tunes at lower LR

Outputs:
  best_model.pt      — encoder weights at lowest validation loss
  training_log.csv   — loss per epoch (use for thesis learning curves)

Usage:
  # Single fold
  python train.py --csv dataset.csv --held-out deepset

  # All four LODO folds (run separately or in a shell loop)
  python train.py --csv dataset.csv --held-out deepset    --output results/deepset/
  python train.py --csv dataset.csv --held-out neuralchemy --output results/neuralchemy/
  python train.py --csv dataset.csv --held-out safeguard  --output results/safeguard/
  python train.py --csv dataset.csv --held-out toxic-chat --output results/toxic-chat/
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data_loader import get_lodo_loaders
from model import PromptInjectionModel, SupConLoss


# ---------------------------------------------------------------------------
# Hyperparameters — all in one place
# ---------------------------------------------------------------------------

EPOCHS          = 10
LR_PROJECTION   = 1e-4   # projection head — higher LR, trained from scratch
LR_ENCODER      = 2e-5   # encoder — lower LR, pre-trained weights
WARMUP_STEPS    = 100    # linear warmup steps for scheduler
TEMPERATURE     = 0.07   # SupCon temperature (Khosla et al. default)
PATIENCE        = 3      # early stopping: stop if val loss does not improve


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimiser, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [train]", leave=False)
    for input_ids, attention_mask, labels in pbar:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        optimiser.zero_grad()
        embeddings = model(input_ids, attention_mask)
        loss       = loss_fn(embeddings, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [val]  ", leave=False)
    for input_ids, attention_mask, labels in pbar:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        embeddings = model(input_ids, attention_mask)
        loss       = loss_fn(embeddings, labels)

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(csv_path: str, held_out_source: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training — held out: '{held_out_source}' ===")
    print(f"Device : {device}")
    print(f"Output : {output_dir}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_lodo_loaders(csv_path, held_out_source)

    # ── Model ─────────────────────────────────────────────────────────────
    model   = PromptInjectionModel(freeze_encoder=True).to(device)
    loss_fn = SupConLoss(temperature=TEMPERATURE)

    # Two param groups so encoder and projection head have different LRs
    # Encoder params start with requires_grad=False (freeze_encoder=True above)
    # They will be added to the optimiser but ignored until unfreeze
    optimiser = torch.optim.AdamW([
        {"params": model.encoder.parameters(),        "lr": LR_ENCODER},
        {"params": model.projection_head.parameters(),"lr": LR_PROJECTION},
    ])

    total_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # ── Logging setup ─────────────────────────────────────────────────────
    log_path = os.path.join(output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "encoder_frozen",
                         "epoch_time_s", "best"])

    best_val_loss  = float("inf")
    patience_count = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Unfreeze encoder after epoch 1
        encoder_frozen = (epoch == 1)
        if epoch == 2:
            model.unfreeze_encoder()
            print("  Encoder unfrozen — full model now training\n")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, scheduler, device, epoch)
        val_loss   = validate(model, val_loader, loss_fn, device, epoch)
        elapsed    = time.time() - t0
        is_best    = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_count = 0
            # Save encoder weights only — projection head is discarded at inference
            torch.save(model.encoder.state_dict(), best_model_path)
        else:
            patience_count += 1

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"{'[BEST]' if is_best else f'[patience {patience_count}/{PATIENCE}]'}  "
            f"frozen={encoder_frozen}  {elapsed:.1f}s"
        )

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             encoder_frozen, f"{elapsed:.1f}", is_best])

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping — val loss did not improve for {PATIENCE} epochs.")
            break

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Model saved   : {best_model_path}")
    print(f"Log saved     : {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="dataset.csv",
                        help="Path to dataset.csv from dataset_pipeline.py")
    parser.add_argument("--held-out",   default="deepset",
                        help="Source to hold out as test fold")
    parser.add_argument("--output",     default="results/",
                        help="Directory for best_model.pt and training_log.csv")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        held_out_source=args.held_out,
        output_dir=args.output,
    )
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
MODEL_NAME      = "roberta-base"
LR_PROJECTION   = 1e-4   # projection head — higher LR, trained from scratch
LR_ENCODER      = 2e-5   # encoder — lower LR, pre-trained weights
WARMUP_STEPS    = 100    # linear warmup steps for scheduler
TEMPERATURE     = 0.07   # SupCon temperature (Khosla et al. default)
PATIENCE        = 3      # early stopping: stop if val loss does not improve
BATCH_SIZE      = 16
MAX_LENGTH      = 64
VAL_FRACTION    = 0.10
FREEZE_ENCODER_EPOCHS = 1
FREEZE_ENCODER_LAYERS = 0


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimiser, scheduler, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [train]", leave=False)
    for input_ids, attention_mask, labels in pbar:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        optimiser.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                embeddings = model(input_ids, attention_mask)
                loss = loss_fn(embeddings, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            embeddings = model(input_ids, attention_mask)
            loss = loss_fn(embeddings, labels)
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
        try:
            loss = loss_fn(embeddings, labels)
        except ValueError:
            continue

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if n_batches == 0:
        raise ValueError("Validation produced no valid SupCon batches.")
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def save_checkpoint(
    path,
    model,
    optimiser,
    scheduler,
    epoch,
    best_val_loss,
    patience_count,
    config,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_count": patience_count,
            "config": config,
        },
        path,
    )


def train(
    csv_path: str,
    held_out_source: str,
    output_dir: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    val_fraction: float = VAL_FRACTION,
    model_name: str = MODEL_NAME,
    freeze_encoder_epochs: int = FREEZE_ENCODER_EPOCHS,
    freeze_encoder_layers: int = FREEZE_ENCODER_LAYERS,
    balanced_sampling: bool = False,
    resume: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training — held out: '{held_out_source}' ===")
    print(f"Device : {device}")
    print(f"Encoder: {model_name}")
    print(f"Output : {output_dir}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_lodo_loaders(
        csv_path,
        held_out_source,
        batch_size=batch_size,
        max_length=max_length,
        val_fraction=val_fraction,
        model_name=model_name,
        balanced_sampling=balanced_sampling,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model   = PromptInjectionModel(
        encoder_name=model_name,
        freeze_encoder=freeze_encoder_epochs > 0,
        freeze_encoder_layers=freeze_encoder_layers,
    ).to(device)
    loss_fn = SupConLoss(temperature=TEMPERATURE)

    # Two param groups so encoder and projection head have different LRs
    # Encoder params start with requires_grad=False (freeze_encoder=True above)
    # They will be added to the optimiser but ignored until unfreeze
    optimiser = torch.optim.AdamW([
        {"params": model.encoder.parameters(),        "lr": LR_ENCODER},
        {"params": model.projection_head.parameters(),"lr": LR_PROJECTION},
    ])

    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # ── Logging setup ─────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    best_checkpoint_path = os.path.join(output_dir, "checkpoint_best.pt")
    last_checkpoint_path = os.path.join(output_dir, "checkpoint_last.pt")
    start_epoch = 1
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    run_config = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "val_fraction": val_fraction,
        "freeze_encoder_epochs": freeze_encoder_epochs,
        "freeze_encoder_layers": freeze_encoder_layers,
        "balanced_sampling": balanced_sampling,
    }

    if resume and os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimiser.load_state_dict(checkpoint["optimiser_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = checkpoint["best_val_loss"]
        patience_count = checkpoint["patience_count"]
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch > freeze_encoder_epochs:
            model.unfreeze_encoder()
        print(f"  Resumed from epoch {checkpoint['epoch']} checkpoint\n")

    log_path = os.path.join(output_dir, "training_log.csv")
    if not resume or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "encoder_frozen",
                             "epoch_time_s", "best"])

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Unfreeze encoder after the projection-head warmup phase.
        encoder_frozen = epoch <= freeze_encoder_epochs
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs + 1:
            model.unfreeze_encoder()
            print("  Encoder unfrozen — full model now training\n")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, scheduler, device, epoch, scaler)
        val_loss   = validate(model, val_loader, loss_fn, device, epoch)
        elapsed    = time.time() - t0
        is_best    = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_count = 0
            # Save encoder weights only — projection head is discarded at inference
            torch.save(model.encoder.state_dict(), best_model_path)
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            patience_count += 1

        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"{'[BEST]' if is_best else f'[patience {patience_count}/{PATIENCE}]'}  "
            f"frozen={encoder_frozen}  {elapsed:.1f}s"
        )

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             encoder_frozen, f"{elapsed:.1f}", is_best])

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimiser,
            scheduler,
            epoch,
            best_val_loss,
            patience_count,
            run_config,
        )

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping — val loss did not improve for {PATIENCE} epochs.")
            break

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Model saved   : {best_model_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Resume point  : {last_checkpoint_path}")
    print(f"Log saved     : {log_path}")

    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        test_loss = validate(model, test_loader, loss_fn, device, epoch)
        print(f"Held-out test loss ({held_out_source}): {test_loss:.4f}")


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
    parser.add_argument("--epochs",     type=int, default=EPOCHS,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Training and evaluation batch size")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="Tokenizer max sequence length")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                        help="Validation fraction taken from non-held-out sources")
    parser.add_argument("--model-name", default=MODEL_NAME,
                        help="HuggingFace encoder name, e.g. roberta-base or distilroberta-base")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=FREEZE_ENCODER_EPOCHS,
                        help="Number of initial epochs with the full encoder frozen")
    parser.add_argument("--freeze-encoder-layers", type=int, default=FREEZE_ENCODER_LAYERS,
                        help="Keep embeddings and the first N encoder layers frozen after warmup")
    parser.add_argument("--balanced-sampling", action="store_true",
                        help="Use class-balanced weighted sampling for the training loader")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from output/checkpoint_last.pt if it exists")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        held_out_source=args.held_out,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        model_name=args.model_name,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        freeze_encoder_layers=args.freeze_encoder_layers,
        balanced_sampling=args.balanced_sampling,
        resume=args.resume,
    )

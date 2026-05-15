"""
Step 1 — Data Loader
Prompt Injection Detection — FYP
=================================
Reads dataset.csv and produces PyTorch DataLoaders for LODO evaluation.

LODO (Leave-One-Dataset-Out):
  Train on all sources except one held-out source.
  Repeat for each source to get 4 evaluation folds.

Usage:
  from data_loader import get_lodo_loaders

  # Hold out 'deepset' as the test fold
  train_loader, test_loader = get_lodo_loaders(
      csv_path="dataset.csv",
      held_out_source="deepset",
  )

  # Iterate
  for input_ids, attention_mask, labels in train_loader:
      ...  # pass to model

Available held_out_source values (must match 'source' column in dataset.csv):
  "deepset" | "neuralchemy" | "safeguard" | "toxic-chat"
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants — change here only, not scattered through the code
# ---------------------------------------------------------------------------

MODEL_NAME  = "roberta-base"
MAX_LENGTH  = 64   # most prompts are short; 64 cuts forward pass time ~4x vs 128
BATCH_SIZE  = 16   # smaller batch = less memory and faster per-step on CPU
NUM_WORKERS = 0    # keep 0 on Windows (multiprocessing deadlock)
VAL_FRACTION = 0.10
SEED = 42

VALID_SOURCES = {"deepset", "neuralchemy", "safeguard", "toxic-chat"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """
    Pre-tokenises all prompts once at init and stores tensors in memory.
    Each item: (input_ids, attention_mask, label)
    label: 0 = benign, 1 = injected

    Pre-tokenising at init (instead of per __getitem__) avoids repeated
    tokeniser overhead on every batch fetch, which caused silent hangs on
    CPU/Windows with 27K+ samples.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        texts  = df["text"].tolist()
        labels = df["label"].tolist()

        print(f"  Tokenising {len(texts)} samples...", end=" ", flush=True)
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        print("done")

        self.input_ids      = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
        )


# ---------------------------------------------------------------------------
# LODO split
# ---------------------------------------------------------------------------

def _lodo_split(df: pd.DataFrame, held_out_source: str):
    """
    Splits df into train (all sources except held_out) and test (held_out only).
    Raises ValueError if held_out_source is not present in the data.
    """
    if held_out_source not in VALID_SOURCES:
        raise ValueError(
            f"held_out_source='{held_out_source}' is not a recognised source. "
            f"Choose from: {sorted(VALID_SOURCES)}"
        )
    present = set(df["source"].unique())
    if held_out_source not in present:
        raise ValueError(
            f"held_out_source='{held_out_source}' not found in dataset. "
            f"Sources present: {sorted(present)}"
        )
    train_df = df[df["source"] != held_out_source].reset_index(drop=True)
    test_df  = df[df["source"] == held_out_source].reset_index(drop=True)
    return train_df, test_df


def _stratified_train_val_split(
    train_pool: pd.DataFrame,
    val_fraction: float,
    seed: int,
):
    """
    Creates a validation split only from the training sources.

    Stratifying by source and label keeps validation representative without
    leaking the held-out LODO source into model selection.
    """
    val_parts = []
    train_parts = []

    for _, group in train_pool.groupby(["source", "label"], sort=False):
        if len(group) < 2:
            train_parts.append(group)
            continue

        n_val = max(1, int(round(len(group) * val_fraction)))
        val_group = group.sample(n=n_val, random_state=seed)
        train_group = group.drop(val_group.index)

        val_parts.append(val_group)
        train_parts.append(train_group)

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_lodo_loaders(
    csv_path: str,
    held_out_source: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    max_length: int = MAX_LENGTH,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    model_name: str = MODEL_NAME,
    balanced_sampling: bool = False,
):
    """
    Returns (train_loader, val_loader, test_loader) for one LODO fold.

    Args:
        csv_path:          Path to dataset.csv produced by dataset_pipeline.py
        held_out_source:   Source name to hold out as the test set
        batch_size:        Samples per batch (default 32)
        num_workers:       DataLoader workers (default 0 = main process)

    Returns:
        train_loader: shuffled train split from non-held-out sources
        val_loader:   validation split from non-held-out sources
        test_loader:  held-out source; use once for final LODO evaluation
    """
    df = pd.read_csv(csv_path)

    required = {"text", "label", "source"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"dataset.csv is missing columns: {missing}. "
            f"Re-run dataset_pipeline.py to regenerate it."
        )

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    train_pool_df, test_df = _lodo_split(df, held_out_source)
    train_df, val_df = _stratified_train_val_split(train_pool_df, val_fraction, seed)

    print(f"LODO fold — held out: '{held_out_source}'")
    print(f"  Train: {len(train_df)} samples  "
          f"({(train_df['label']==0).sum()} benign / "
          f"{(train_df['label']==1).sum()} injected)")
    print(f"  Val  : {len(val_df)} samples  "
          f"({(val_df['label']==0).sum()} benign / "
          f"{(val_df['label']==1).sum()} injected)")
    print(f"  Test : {len(test_df)} samples  "
          f"({(test_df['label']==0).sum()} benign / "
          f"{(test_df['label']==1).sum()} injected)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    train_ds = PromptDataset(train_df, tokenizer, max_length=max_length)
    val_ds   = PromptDataset(val_df,   tokenizer, max_length=max_length)
    test_ds  = PromptDataset(test_df,  tokenizer, max_length=max_length)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler = None
    shuffle_train = True
    if balanced_sampling:
        class_counts = train_df["label"].value_counts().to_dict()
        sample_weights = train_df["label"].map(lambda label: 1.0 / class_counts[label]).tolist()
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle_train = False
        print("  Balanced sampling: enabled")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        drop_last=True,      # SupCon needs full batches for meaningful pairs
        num_workers=num_workers,
        generator=None if balanced_sampling else generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick smoke test — run directly to verify on your machine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    source   = sys.argv[2] if len(sys.argv) > 2 else "deepset"

    train_loader, val_loader, test_loader = get_lodo_loaders(csv_path, source)

    # Inspect one batch
    ids, mask, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids     : {ids.shape}")
    print(f"  attention_mask: {mask.shape}")
    print(f"  labels        : {labels.shape}  unique={labels.unique().tolist()}")
    print(f"  val batches   : {len(val_loader)}")
    print(f"  test batches  : {len(test_loader)}")
    print("\nData loader ready.")

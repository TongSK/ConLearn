"""
Dataset Preparation Pipeline
Prompt Injection Detection — FYP
=================================
Output: dataset.csv with columns: text, label, source, attack_category
  label          : 0 = benign, 1 = injected
  source         : dataset name (used for LODO evaluation splits)
  attack_category: attack type string, or "benign"

Sources:
  1. deepset/prompt-injections          (546 rows, ungated)
  2. neuralchemy/Prompt-injection-dataset (16918 rows, ungated, attack categories)
  3. xTRam1/safe-guard-prompt-injection  (~5000 rows, ungated, synthetic attacks)
  4. lmsys/toxic-chat                    (toxicchat0124 config, ungated)

Usage:
  pip install datasets pandas
  python dataset_pipeline.py
  python dataset_pipeline.py --output my_dataset.csv
"""

import argparse
import os
import sys
import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Schema definitions — fail loudly if a column is missing
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "name": "deepset",
        "hf_path": "deepset/prompt-injections",
        "config": None,
        "split": "train",
        "required_cols": ["text", "label"],
    },
    {
        "name": "neuralchemy",
        "hf_path": "neuralchemy/Prompt-injection-dataset",
        "config": "full",
        "split": "train",
        "required_cols": ["text", "label"],
    },
    {
        "name": "safeguard",
        "hf_path": "xTRam1/safe-guard-prompt-injection",
        "config": None,
        "split": "train",
        "required_cols": ["text", "label"],
    },
    {
        "name": "toxic-chat",
        "hf_path": "lmsys/toxic-chat",
        "config": "toxicchat0124",       # required — dataset has multiple configs
        "split": "train",
        "required_cols": ["user_input", "jailbreaking"],
    },
]


# ---------------------------------------------------------------------------
# Per-source normalisation — one function per dataset, no shared magic
# ---------------------------------------------------------------------------

def normalise_neuralchemy(ds) -> pd.DataFrame:
    df = ds.to_pandas()
    _require_cols(df, ["text", "label"], "neuralchemy")
    # "category" column contains attack type e.g. direct_injection, jailbreak, benign
    if "category" in df.columns:
        attack_cat = df["category"].fillna("unknown").astype(str)
    else:
        attack_cat = df["label"].map({0: "benign", 1: "injection"})
    return pd.DataFrame({
        "text":            df["text"].astype(str),
        "label":           df["label"].astype(int),
        "source":          "neuralchemy",
        "attack_category": attack_cat,
    })


def normalise_safeguard(ds) -> pd.DataFrame:
    df = ds.to_pandas()
    _require_cols(df, ["text", "label"], "safeguard")
    return pd.DataFrame({
        "text":            df["text"].astype(str),
        "label":           df["label"].astype(int),
        "source":          "safeguard",
        "attack_category": df["label"].map({0: "benign", 1: "injection"}),
    })


def normalise_deepset(ds) -> pd.DataFrame:
    df = ds.to_pandas()
    _require_cols(df, ["text", "label"], "deepset")
    return pd.DataFrame({
        "text":            df["text"].astype(str),
        "label":           df["label"].astype(int),
        "source":          "deepset",
        "attack_category": df["label"].map({0: "benign", 1: "injection"}),
    })


def normalise_toxic_chat(ds) -> pd.DataFrame:
    df = ds.to_pandas()
    _require_cols(df, ["user_input", "jailbreaking"], "toxic-chat")
    return pd.DataFrame({
        "text":            df["user_input"].astype(str),
        "label":           df["jailbreaking"].astype(int),
        "source":          "toxic-chat",
        "attack_category": df["jailbreaking"].map({0: "benign", 1: "jailbreak"}),
    })


NORMALISERS = {
    "deepset":     normalise_deepset,
    "neuralchemy": normalise_neuralchemy,
    "safeguard":   normalise_safeguard,
    "toxic-chat":  normalise_toxic_chat,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: list, source: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] Schema mismatch — expected columns {missing} "
            f"but got {list(df.columns)}. "
            f"The dataset may have changed. Update the normaliser for '{source}'."
        )


def load_and_normalise(source_cfg: dict) -> pd.DataFrame:
    name = source_cfg["name"]
    print(f"  Loading {name} ({source_cfg['hf_path']})...", end=" ", flush=True)
    config = source_cfg.get("config")
    if config:
        ds = load_dataset(source_cfg["hf_path"], config, split=source_cfg["split"])
    else:
        ds = load_dataset(source_cfg["hf_path"], split=source_cfg["split"])
    print(f"{len(ds)} rows")
    return NORMALISERS[name](ds)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    print(f"  Cleaning: {before} → {len(df)} rows ({before - len(df)} dropped)")
    return df.reset_index(drop=True)


def print_summary(df: pd.DataFrame):
    print("\n=== Dataset Summary ===")
    print(f"Total rows : {len(df)}")
    print(f"Benign (0) : {(df['label'] == 0).sum()}")
    print(f"Injected(1): {(df['label'] == 1).sum()}")
    print("\nRows per source:")
    print(df.groupby("source")["label"].value_counts().to_string())
    print("\nAttack categories:")
    print(df["attack_category"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_path: str):
    print("=== Prompt Injection Dataset Pipeline ===\n")
    frames = []

    for cfg in SOURCES:
        try:
            df = load_and_normalise(cfg)
            frames.append(df)
        except Exception as e:
            print(f"  ERROR loading {cfg['name']}: {e}")
            print(f"  Skipping {cfg['name']} and continuing.\n")

    if not frames:
        print("No datasets loaded. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = clean(combined)
    print_summary(combined)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    combined.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dataset.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.output)

"""
Step 2 — Model
Prompt Injection Detection — FYP
=================================
Components:
  1. PromptInjectionModel  — RoBERTa encoder + projection head
  2. SupConLoss            — Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)

Training flow:
  text → tokeniser (Step 1) → encoder → [CLS] embedding (768-dim)
       → projection head → L2-normalised embedding (128-dim)
       → SupConLoss

Inference flow (projection head discarded):
  text → tokeniser → encoder → [CLS] embedding (768-dim)
       → cosine similarity against class centroids → label

Usage:
  from model import PromptInjectionModel, SupConLoss

  model = PromptInjectionModel()
  loss_fn = SupConLoss()

  embeddings = model(input_ids, attention_mask)   # (batch, 128)
  loss = loss_fn(embeddings, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PromptInjectionModel(nn.Module):
    """
    Transformer encoder with a two-layer projection head.

    forward() returns L2-normalised 128-dim embeddings — ready for SupConLoss.
    get_embeddings() returns raw 768-dim [CLS] embeddings — used at inference.
    """

    def __init__(
        self,
        encoder_name: str = "roberta-base",
        projection_dim: int = 128,
        freeze_encoder: bool = True,
        freeze_encoder_layers: int = 0,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.freeze_encoder_layers = freeze_encoder_layers

        # Freeze encoder weights for epoch 1 — unfreeze in train loop after warmup
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_encoder_layers > 0:
            self.freeze_bottom_encoder_layers(freeze_encoder_layers)

        encoder_hidden = self.encoder.config.hidden_size  # 768 for roberta-base

        # Two-layer MLP: 768 → 256 → 128
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns L2-normalised projection embeddings.
        Shape: (batch_size, projection_dim)
        Used during training with SupConLoss.
        """
        cls_embedding = self._encode(input_ids, attention_mask)
        projected = self.projection_head(cls_embedding)
        return F.normalize(projected, p=2, dim=1)

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw [CLS] embeddings from encoder (no projection head).
        Shape: (batch_size, 768)
        Used at inference time.
        """
        return self._encode(input_ids, attention_mask)

    def unfreeze_encoder(self):
        """Call after warmup epoch to allow encoder fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        if self.freeze_encoder_layers > 0:
            self.freeze_bottom_encoder_layers(self.freeze_encoder_layers)

    def freeze_bottom_encoder_layers(self, n_layers: int):
        """
        Keeps embeddings and the first n transformer blocks frozen.

        This reduces gradient memory/time while still fine-tuning higher-level
        layers. It works for RoBERTa/BERT-style encoders exposed by AutoModel.
        """
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        layer_stack = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layer_stack is None:
            return

        for layer in layer_stack[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract [CLS] token embedding from RoBERTa."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]  # [CLS] token, shape (batch, 768)


# ---------------------------------------------------------------------------
# SupCon Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss — Khosla et al., NeurIPS 2020.
    https://arxiv.org/abs/2004.11362

    For a batch of N embeddings with labels:
    - Each sample i is the anchor
    - Positives: all other samples j with the same label
    - Negatives: all samples j with a different label

    Requires:
    - embeddings to be L2-normalised (model.forward() handles this)
    - At least 2 samples per class in each batch (use drop_last=True in DataLoader)

    Args:
        temperature: scaling factor for cosine similarities (default 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalised, shape (batch_size, projection_dim)
            labels:     shape (batch_size,), values in {0, 1}
        Returns:
            Scalar loss
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Cosine similarity matrix: (N, N)
        # Since embeddings are L2-normalised, dot product = cosine similarity
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask out self-similarity on the diagonal
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))

        # Positive mask: True where labels match (excluding self)
        labels = labels.unsqueeze(1)                              # (N, 1)
        positive_mask = (labels == labels.T) & ~self_mask        # (N, N)

        # For numerical stability: subtract max before exp
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix)

        # Sum of exp similarities to all non-self samples (denominator)
        denom = exp_sim.masked_fill(self_mask, 0).sum(dim=1, keepdim=True)

        # Log probability for each positive pair
        log_prob = sim_matrix - torch.log(denom + 1e-8)

        # Average log probability over positives for each anchor
        n_positives = positive_mask.sum(dim=1).float()

        # Guard: if any anchor has no positives, loss is undefined for that anchor
        has_positive = n_positives > 0
        if not has_positive.any():
            raise ValueError(
                "SupConLoss: no positive pairs found in batch. "
                "Ensure drop_last=True and that each batch contains both classes. "
                f"Batch labels: {labels.squeeze().tolist()}"
            )

        # Zero out -inf positions before multiplying — prevents 0 * (-inf) = nan
        log_prob = log_prob.masked_fill(self_mask, 0.0)
        loss_per_anchor = -(log_prob * positive_mask).sum(dim=1) / (n_positives + 1e-8)
        return loss_per_anchor[has_positive].mean()


# ---------------------------------------------------------------------------
# Smoke test — run directly to verify shapes and loss on your machine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Loading model (downloads roberta-base on first run)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = PromptInjectionModel(freeze_encoder=True).to(device)
    loss_fn = SupConLoss(temperature=0.07)

    # Simulate one batch from data_loader.py
    batch_size = 8
    seq_len    = 128
    input_ids  = torch.randint(0, 50265, (batch_size, seq_len)).to(device)
    attn_mask  = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    labels     = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).to(device)

    # Forward pass
    embeddings = model(input_ids, attn_mask)
    print(f"\nEmbedding shape : {embeddings.shape}")          # (8, 128)
    print(f"L2 norms (≈1.0) : {embeddings.norm(dim=1).tolist()}")

    # SupCon loss
    loss = loss_fn(embeddings, labels)
    print(f"SupCon loss     : {loss.item():.4f}")

    # get_embeddings (inference mode)
    with torch.no_grad():
        raw_emb = model.get_embeddings(input_ids, attn_mask)
    print(f"Raw [CLS] shape : {raw_emb.shape}")              # (8, 768)

    # Unfreeze and check param count changes
    frozen_params   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFrozen params   : {frozen_params:,}")
    print(f"Trainable params: {trainable_params:,}  (projection head only)")

    model.unfreeze_encoder()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable_after:,}  (full model)")

    print("\nModel ready.")

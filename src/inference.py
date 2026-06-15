"""
Runtime inference for the projection-space contrastive detector.

The deployed decision path intentionally matches evaluation:
  projected embedding -> class-centroid similarities -> validation threshold

No rule-based score boosting or post-hoc probability calibration is applied.
"""

import torch
from transformers import AutoTokenizer

from model import PromptInjectionModel


class PromptInjectionDetector:
    def __init__(self, artifact_path="artifacts/detector_artifact.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.artifact = torch.load(artifact_path, map_location=self.device)

        if self.artifact.get("embedding_space") != "projection":
            raise ValueError(
                "This detector artifact was exported from the old CLS embedding pipeline. "
                "Re-export it with the updated src/export_detector.py."
            )
        if "model_state_dict" not in self.artifact:
            raise ValueError(
                "Detector artifact is missing the trained projection head. "
                "Re-export it from checkpoint_best.pt."
            )

        self.model_name = self.artifact["model_name"]
        self.max_length = int(self.artifact["max_length"])
        self.threshold = float(self.artifact["threshold"])
        self.projection_dim = int(self.artifact.get("projection_dim", 128))
        self.centroids = self.artifact["centroids"].to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = PromptInjectionModel(
            encoder_name=self.model_name,
            projection_dim=self.projection_dim,
            freeze_encoder=False,
        ).to(self.device)
        self.model.load_state_dict(self.artifact["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, text, sensitivity="balanced"):
        encoded = self.tokenizer(
            [text],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        embedding = self.model(input_ids, attention_mask)
        similarities = embedding @ self.centroids.T
        benign_similarity = float(similarities[:, 0].item())
        injected_similarity = float(similarities[:, 1].item())
        score = float((similarities[:, 1] - similarities[:, 0]).item())
        decision_margin = score - self.threshold
        is_injected = score >= self.threshold

        # This is a relative centroid score, not a calibrated probability.
        risk_score = float(torch.softmax(similarities, dim=1)[:, 1].item())
        decision_confidence = risk_score if is_injected else 1.0 - risk_score

        reason = (
            "Projected embedding exceeded the validation-tuned prompt-injection threshold."
            if is_injected
            else "Projected embedding remained below the validation-tuned prompt-injection threshold."
        )

        return {
            "label": "prompt_injection" if is_injected else "benign",
            "is_prompt_injection": bool(is_injected),
            "score": score,
            "decision_margin": decision_margin,
            "benign_similarity": benign_similarity,
            "injected_similarity": injected_similarity,
            "threshold": self.threshold,
            "risk_score": risk_score,
            "decision_confidence": decision_confidence,
            "score_type": "uncalibrated_centroid_softmax",
            "embedding_space": "projection",
            "projection_dim": self.projection_dim,
            "action": "block" if is_injected else "allow",
            "reason": reason,
        }

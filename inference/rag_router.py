"""RAG-style router: image-type detection, embedding-based similar-case
retrieval, and reasoning/recommendation text generation, scoped to what
this project can actually back with a real model.

Honesty notes (read before extending this file):
  - Image-type detection always returns "chest" here. There is no
    LungsMedicalNet to route lung CT to (see
    models/custom_architectures.py), so building a real chest-vs-CT
    classifier would have nothing useful to route to yet. This is a
    placeholder, not a working chest/lungs classifier.
  - `DiseaseMemoryBank` is a real, working in-memory nearest-neighbor
    store over ChestMedicalNet's shared-trunk embeddings (cosine
    similarity) -- not a vector database, not persisted to disk. Fine for
    a research prototype; would need a real backing store (e.g. FAISS +
    a persistent DB) before any production use.
  - TB predictions are surfaced with an explicit low-confidence /
    "not clinically validated" flag, since NIH ChestX-ray14 has no TB
    label and the TB pathway has never seen a real TB-positive example
    (see config.config.TB_LABEL_AVAILABLE).
  - Nothing in this file is a clinical decision support tool. Output text
    says so explicitly and should not be stripped out by future edits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from config.config import GENERAL_PATHWAY_NAMES, TB_LABEL_AVAILABLE
from models.uncertainty import mc_dropout_predict
from utils.helpers import assemble_disease_probs

logger = logging.getLogger(__name__)

CRITICAL_DISEASES = ["Pneumonia", "Mass", "Pneumothorax"]
HIGH_CONFIDENCE_THRESHOLD = 0.8
MODERATE_CONFIDENCE_THRESHOLD = 0.5
HIGH_UNCERTAINTY_THRESHOLD = 0.15

DISCLAIMER = (
    "Research prototype output, not a clinical decision-support tool. "
    "Not FDA-cleared or clinically validated. All findings require "
    "radiologist review."
)


@dataclass
class MemoryCase:
    embedding: Tensor
    labels: dict[str, float]
    image_name: str


class DiseaseMemoryBank:
    """In-memory nearest-neighbor store over model embeddings. Real,
    working cosine-similarity retrieval -- not a persistent vector DB.
    """

    def __init__(self) -> None:
        self._cases: list[MemoryCase] = []

    def add(self, embedding: Tensor, labels: dict[str, float], image_name: str) -> None:
        self._cases.append(MemoryCase(embedding=embedding.detach().cpu(), labels=labels, image_name=image_name))

    def retrieve(self, embedding: Tensor, k: int = 3) -> list[tuple[MemoryCase, float]]:
        if not self._cases:
            return []
        query = embedding.detach().cpu().unsqueeze(0)
        bank = torch.stack([c.embedding for c in self._cases])
        similarities = torch.nn.functional.cosine_similarity(query, bank)
        top_k = torch.topk(similarities, k=min(k, len(self._cases)))
        return [(self._cases[i], float(top_k.values[j])) for j, i in enumerate(top_k.indices)]

    def __len__(self) -> int:
        return len(self._cases)


class ChestRAGRouter:
    """Routes a chest X-ray through ChestMedicalNet, retrieves similar
    historical cases for critical findings, and generates human-readable
    reasoning + a recommendation. See module docstring for scope/honesty
    notes -- lungs routing is not implemented.
    """

    def __init__(self, chest_model: nn.Module, device: torch.device) -> None:
        self.chest_model = chest_model.to(device)
        self.device = device
        self.memory_bank = DiseaseMemoryBank()

    def detect_image_type(self, _image: Tensor) -> dict:
        """Always returns 'chest' -- see module docstring."""
        return {"type": "chest", "confidence": 1.0, "note": "Lungs CT routing not implemented"}

    def route_and_predict(self, image: Tensor, image_name: str = "unknown", mc_passes: int = 20) -> dict:
        """Args:
            image: (1, 3, H, W) preprocessed image tensor.
            image_name: identifier, used for memory-bank bookkeeping.
            mc_passes: MC Dropout passes for uncertainty (0 to skip, faster).
        """
        image = image.to(self.device)
        image_type = self.detect_image_type(image)

        self.chest_model.eval()
        with torch.no_grad():
            output = self.chest_model(image)

        disease_probs = assemble_disease_probs(output)[0]  # (14,)
        pneumonia_prob = torch.sigmoid(output["pneumonia_logit"])[0].item()
        predictions = {name: float(disease_probs[i]) for i, name in enumerate(
            [c for c in GENERAL_PATHWAY_NAMES] + ["Pneumonia"]
        )}

        tb_result = None
        if "tb_logit" in output:
            tb_prob = torch.sigmoid(output["tb_logit"])[0].item()
            tb_result = {
                "probability": tb_prob,
                "clinically_validated": TB_LABEL_AVAILABLE,
                "note": "TB pathway has never seen a real TB-labeled example; treat as untrained." if not TB_LABEL_AVAILABLE else "",
            }

        uncertainty = {"epistemic": None, "aleatoric": None, "total": None}
        if mc_passes > 0:
            mc_result = mc_dropout_predict(self.chest_model, image, num_passes=mc_passes)
            uncertainty = {
                "epistemic": float(mc_result["epistemic_std"].mean()),
                "aleatoric": float(mc_result["aleatoric_std"].mean()),
                "total": float(mc_result["total_std"].mean()),
            }

        similar_cases = {}
        for disease in CRITICAL_DISEASES:
            if predictions.get(disease, 0.0) > MODERATE_CONFIDENCE_THRESHOLD and len(self.memory_bank) > 0:
                similar_cases[disease] = self.memory_bank.retrieve(output["embedding"][0], k=3)

        recommendation = self._generate_recommendation(predictions, tb_result, uncertainty)
        reasoning = self._generate_reasoning(image_type, predictions, uncertainty, similar_cases)

        return {
            "model_used": "chest",
            "image_type_detected": image_type["type"],
            "predictions": predictions,
            "tb_pathway": tb_result,
            "uncertainty": uncertainty,
            "similar_cases": similar_cases,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "disclaimer": DISCLAIMER,
        }

    def _generate_reasoning(self, image_type: dict, predictions: dict, uncertainty: dict, similar_cases: dict) -> str:
        top_findings = sorted(predictions.items(), key=lambda kv: -kv[1])[:5]
        findings_str = ", ".join(f"{name} ({prob:.1%})" for name, prob in top_findings)
        lines = [
            f"Image type: {image_type['type']} ({image_type['confidence']:.0%} confidence; {image_type['note']})",
            f"Top predictions: {findings_str}",
        ]
        if uncertainty["total"] is not None:
            lines.append(f"Predictive uncertainty (epistemic+aleatoric): {uncertainty['total']:.2%}")
        for disease, cases in similar_cases.items():
            case_str = "; ".join(f"{c.image_name} (sim={sim:.2f})" for c, sim in cases)
            lines.append(f"Similar cases for {disease}: {case_str}" if cases else f"No similar cases found for {disease} (empty memory bank)")
        return "\n".join(lines)

    def _generate_recommendation(self, predictions: dict, tb_result: dict | None, uncertainty: dict) -> str:
        recs = []
        for disease in CRITICAL_DISEASES:
            prob = predictions.get(disease, 0.0)
            if prob > HIGH_CONFIDENCE_THRESHOLD:
                recs.append(f"HIGH probability {disease} ({prob:.1%}): recommend radiologist review")
            elif prob > MODERATE_CONFIDENCE_THRESHOLD:
                recs.append(f"Possible {disease} ({prob:.1%}): radiologist review suggested")

        if tb_result and tb_result["probability"] > MODERATE_CONFIDENCE_THRESHOLD:
            recs.append(
                f"TB pathway flagged {tb_result['probability']:.1%} -- NOT clinically validated "
                "(no TB training data), do not act on this without independent confirmation"
            )

        if uncertainty["total"] is not None and uncertainty["total"] > HIGH_UNCERTAINTY_THRESHOLD:
            recs.append(f"High predictive uncertainty ({uncertainty['total']:.1%}): consider repeat imaging")

        return " | ".join(recs) if recs else "No critical findings above threshold"

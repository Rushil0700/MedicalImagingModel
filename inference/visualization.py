"""Grad-CAM localization and prediction visualization.

The project spec describes a "localization head: Grad-CAM output" as if
Grad-CAM were something the network learns to produce. It isn't -- Grad-CAM
(Selvaraju et al., 2017) is a post-hoc technique: you pick a conv layer,
run a forward pass, backprop a chosen class's score to that layer, and
weight its activation maps by the averaged gradient. No extra trainable
head is needed or appropriate here; `GradCAM` below implements the real
technique against ChestMedicalNet's backbone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GradCAM:
    """Grad-CAM against a chosen conv layer (default: ChestMedicalNet's
    backbone's last stage, `model.backbone.features[-1]`).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or model.backbone.features[-1]
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(self, image: Tensor, target_score_fn) -> np.ndarray:
        """Args:
            image: (1, 3, H, W) single input image, requires_grad not needed.
            target_score_fn: callable(model_output_dict) -> scalar Tensor,
                e.g. `lambda out: out["pneumonia_logit"][0]` or
                `lambda out: out["general_mean"][0, class_idx]`.

        Returns:
            (H, W) float32 heatmap in [0, 1], resized to the input's H, W.
        """
        self.model.eval()
        image = image.clone().requires_grad_(True)

        output = self.model(image)
        score = target_score_fn(output)

        self.model.zero_grad(set_to_none=True)
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap (H, W) in [0,1] over a grayscale/RGB image
    (H, W) or (H, W, 3) in [0,1], returning an (H, W, 3) RGB array in [0,1].
    Uses a simple red-channel overlay rather than pulling in a colormap
    dependency (matplotlib) for this one utility.
    """
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    overlay = image_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + alpha * heatmap, 0, 1)
    return overlay

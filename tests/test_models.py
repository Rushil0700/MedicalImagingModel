"""Tests for ChestMedicalNet. LungsMedicalNet has no data/pipeline behind
it yet (see models/custom_architectures.py), so it is tested only for
raising NotImplementedError, not for real behavior.
"""
from __future__ import annotations

import pytest
import torch

from config.config import GENERAL_PATHWAY_NAMES
from models.custom_architectures import ChestMedicalNet, LungsMedicalNet


@pytest.fixture(scope="module")
def model() -> ChestMedicalNet:
    torch.manual_seed(0)
    return ChestMedicalNet(input_size=256, backbone_variant="convnext_base", pretrained=False)


def test_chest_model_output_keys_and_shapes(model: ChestMedicalNet) -> None:
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    assert out["general_mean"].shape == (2, len(GENERAL_PATHWAY_NAMES))
    assert out["general_log_var"].shape == (2, len(GENERAL_PATHWAY_NAMES))
    assert out["pneumonia_logit"].shape == (2,)
    assert out["tb_logit"].shape == (2,)
    assert out["severity_logits"].shape == (2, len(GENERAL_PATHWAY_NAMES) + 1, 3)  # 4 levels -> 3 thresholds
    assert out["embedding"].shape[0] == 2


def test_log_var_is_bounded(model: ChestMedicalNet) -> None:
    x = torch.randn(2, 3, 256, 256) * 10  # extreme input, should not blow up log_var
    out = model(x)
    assert out["general_log_var"].min() >= -6.0 - 1e-4
    assert out["general_log_var"].max() <= 6.0 + 1e-4


def test_backward_pass_produces_gradients(model: ChestMedicalNet) -> None:
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    loss = out["general_mean"].sum() + out["pneumonia_logit"].sum() + out["tb_logit"].sum()
    loss.backward()
    assert next(model.backbone.features.parameters()).grad is not None


def test_lungs_model_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        LungsMedicalNet()

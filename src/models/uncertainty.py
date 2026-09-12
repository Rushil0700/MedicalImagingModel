"""Uncertainty quantification: MC Dropout (epistemic) + Bayesian head (aleatoric).

Implements docs/enhanced_architecture_spec.md section 4. The two mechanisms
are complementary and are combined only at inference time in
`mc_dropout_predict` below:

- Epistemic uncertainty (model uncertainty) comes from keeping dropout
  active at inference and running T stochastic forward passes.
- Aleatoric uncertainty (data/label-noise uncertainty) comes from the
  Bayesian output layer's learned log-variance, trained with a Gaussian
  heteroscedastic NLL term (see src/losses/focal_loss.py).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class BayesianLinear(nn.Module):
    """Linear layer producing (mean, log-variance) instead of a point estimate.

    At train time, `log_var` is consumed by a heteroscedastic loss. At
    inference, sampling via the reparameterization trick lets the same
    module contribute an aleatoric-uncertainty estimate.

    `log_var` is clamped to `[min_log_var, max_log_var]`. Without this, the
    heteroscedastic NLL term (0.5 * exp(-log_var) * residual^2 + 0.5 * log_var)
    is unbounded below as log_var -> -inf: the model can minimize training
    loss by driving predicted variance to ~0 on easy samples rather than by
    improving `mean`, which collapses the aleatoric-uncertainty estimate to
    false overconfidence -- the opposite of what this head exists to
    provide for medical-safety flagging (docs/enhanced_architecture_spec.md
    section 4). The clamp bounds predicted std to roughly [0.05, 20] in
    sigmoid-residual units.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        min_log_var: float = -6.0,
        max_log_var: float = 6.0,
    ) -> None:
        super().__init__()
        self.mean_layer = nn.Linear(in_features, out_features)
        self.log_var_layer = nn.Linear(in_features, out_features)
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x).clamp(self.min_log_var, self.max_log_var)
        return mean, log_var

    def sample(self, mean: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


def enable_mc_dropout(model: nn.Module) -> None:
    """Switch every Dropout submodule to train mode while leaving the rest of
    the model in eval mode, so BatchNorm/LayerNorm statistics stay frozen but
    dropout remains stochastic for MC sampling.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: Tensor,
    num_passes: int = 20,
) -> dict[str, Tensor]:
    """Run T stochastic forward passes to estimate total predictive uncertainty.

    Args:
        model: The full ChestXRayMaxViTv2 model, already in `.eval()` mode.
        x: Input batch, shape (B, 3, H, W).
        num_passes: Number of MC Dropout forward passes (T=20 per design doc).

    Returns:
        Dict with:
            "mean_prob": (B, C) mean predicted probability per class.
            "epistemic_std": (B, C) std-dev across the T passes (model uncertainty).
            "aleatoric_std": (B, C) mean learned std-dev from the Bayesian head.
            "total_std": (B, C) combined uncertainty, sqrt(epistemic^2 + aleatoric^2).
    """
    model.eval()
    enable_mc_dropout(model)

    probs, sigmas = [], []
    for _ in range(num_passes):
        out = model(x)
        mean, log_var = out["disease_mean"], out["disease_log_var"]
        probs.append(torch.sigmoid(mean))
        sigmas.append(torch.exp(0.5 * log_var))

    probs = torch.stack(probs, dim=0)  # (T, B, C)
    sigmas = torch.stack(sigmas, dim=0)  # (T, B, C)

    mean_prob = probs.mean(dim=0)
    epistemic_std = probs.std(dim=0)
    aleatoric_std = sigmas.mean(dim=0)
    total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)

    return {
        "mean_prob": mean_prob,
        "epistemic_std": epistemic_std,
        "aleatoric_std": aleatoric_std,
        "total_std": total_std,
    }

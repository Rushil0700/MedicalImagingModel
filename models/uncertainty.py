"""Uncertainty quantification: MC Dropout (epistemic) + Bayesian head (aleatoric)."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class BayesianLinear(nn.Module):
    """Linear layer producing (mean, log-variance) instead of a point estimate.

    `log_var` is clamped to `[min_log_var, max_log_var]`. Without this, the
    heteroscedastic NLL term (0.5*exp(-log_var)*residual^2 + 0.5*log_var) is
    unbounded below as log_var -> -inf: the model can minimize training loss
    by driving predicted variance toward 0 on easy samples rather than by
    improving `mean`, collapsing the uncertainty estimate to false
    overconfidence. This clamp bounds predicted std to roughly [0.05, 20].
    """

    def __init__(
        self, in_features: int, out_features: int, min_log_var: float = -6.0, max_log_var: float = 6.0
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


def enable_mc_dropout(model: nn.Module) -> None:
    """Switch every Dropout submodule to train mode while leaving the rest of
    the model in eval mode, so BatchNorm/LayerNorm statistics stay frozen but
    dropout remains stochastic for MC sampling.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: Tensor, num_passes: int = 20) -> dict[str, Tensor]:
    """Run T stochastic forward passes to estimate total predictive uncertainty
    for the general-pathway disease head. Returns mean probability, epistemic
    std (across passes), aleatoric std (mean learned std), and combined total.
    """
    model.eval()
    enable_mc_dropout(model)

    probs, sigmas = [], []
    for _ in range(num_passes):
        out = model(x)
        mean, log_var = out["general_mean"], out["general_log_var"]
        probs.append(torch.sigmoid(mean))
        sigmas.append(torch.exp(0.5 * log_var))

    probs = torch.stack(probs, dim=0)
    sigmas = torch.stack(sigmas, dim=0)

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

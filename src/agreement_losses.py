"""
Losses for the agreement head.

Two distinct things live here:

  1. AgreementLoss — trains head B to predict js_excess, independent of the
     density loss.

  2. heteroscedastic_deep_supervision_loss — lets head B's prediction
     reweight head A's density loss. Off by default.

Relation to the existing deep_supervision_loss
----------------------------------------------
losses.deep_supervision_loss applies the same criterion to the same target
for every head, with a fixed constant weight (aux_weight = 0.3 in
config.yaml): every head is a redundant estimator of one quantity.

Head B is not redundant with head A — it predicts a different quantity, and
it is kept at inference rather than discarded. That makes this multi-task
learning rather than deep supervision, and is why the loss cannot simply be
another term inside the existing function.

The heteroscedastic variant generalizes something the current code already
does: aux_weight is a fixed per-run constant, and this replaces it with a
learned, per-image weight derived from how much observers actually agreed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

# Predicted log-variance is clamped to keep exp() and its reciprocal finite.
# exp(-8) ~ 3e-4 and exp(4) ~ 55 comfortably bracket the observed spread of
# js_excess (order 0.1-1 bits), and the clamp stops a single degenerate image
# from producing an infinite gradient early in training.
LOG_VAR_MIN, LOG_VAR_MAX = -8.0, 4.0


def gaussian_nll(pred_mean, pred_log_var, target, reduction: str = "mean"):
    """
    Negative log-likelihood of `target` under N(pred_mean, exp(pred_log_var)).

    Preferred over MSE because the target is genuinely heteroscedastic: on
    images where observers barely differ, js_excess is a small number sitting
    on top of estimation noise, and forcing the model to fit it as precisely as
    a large, well-determined value wastes capacity on unmeasurable variation.
    """
    log_var = pred_log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    inv_var = torch.exp(-log_var)
    loss = 0.5 * (log_var + inv_var * (target - pred_mean) ** 2)
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    return loss.sum()


def beta_nll(pred_mean, pred_log_var, target, beta: float = 0.5, reduction: str = "mean"):
    """
    Gaussian NLL with the gradient scaled by variance^beta (Seitzer et al., 2022).

    Plain Gaussian NLL has a well-known failure mode early in training: the
    model discovers it can drive the loss down by inflating the variance on
    hard examples instead of fitting them, and those examples then contribute
    almost no gradient and never get learned. Weighting by a detached
    variance^beta restores their influence. beta=0 recovers plain NLL, beta=1
    recovers MSE; 0.5 is the usual compromise and the default here.
    """
    log_var = pred_log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    var = torch.exp(log_var)
    loss = 0.5 * (log_var + (target - pred_mean) ** 2 / var)
    loss = loss * var.detach() ** beta
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    return loss.sum()


class AgreementLoss(nn.Module):
    """
    config["agreement"]["loss"] options:
      "mse"       - plain regression, useful as a sanity check on the pipeline
      "gaussian"  - Gaussian NLL over (mean, log_var)
      "beta_nll"  - variance-reweighted NLL (recommended)

    Mirrors the shape of SaliencyLoss in losses.py so it slots into the same
    config-driven pattern.
    """

    def __init__(self, loss_type: str = "beta_nll", beta: float = 0.5):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred_mean, pred_log_var, target):
        target = target.to(pred_mean.dtype)
        if self.loss_type == "mse":
            return self.mse(pred_mean, target)
        if self.loss_type == "gaussian":
            return gaussian_nll(pred_mean, pred_log_var, target)
        if self.loss_type == "beta_nll":
            return beta_nll(pred_mean, pred_log_var, target, beta=self.beta)
        raise ValueError(f"Unknown agreement loss_type: {self.loss_type}")


class HeteroscedasticGate(nn.Module):
    """
    Turns predicted disagreement into a per-image weight on the density loss.

        weight = exp(-gate * normalized_disagreement)

    `gate` is a single learned scalar initialized at zero, so the model starts
    with weight == 1 for every image (numerically identical to the current
    training behaviour) and has to earn any deviation from it. That keeps a
    stage-1-vs-stage-3 comparison from being confounded by a different
    effective learning rate on the density loss at initialization.

    Weights are renormalized to mean 1 within the batch so that turning the
    gate on cannot silently rescale the overall gradient magnitude.
    """

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(init)))

    def forward(self, disagreement: torch.Tensor) -> torch.Tensor:
        d = disagreement.detach()
        d = (d - d.mean()) / (d.std() + EPS)
        w = torch.exp(-self.gate * d)
        return w / (w.mean() + EPS)


def heteroscedastic_deep_supervision_loss(criterion, fused, aux, target,
                                          weights=None, aux_weight: float = 0.3):
    """
    losses.deep_supervision_loss with an optional per-image weight vector.

    Identical to the original when `weights` is None, so this can replace the
    call site unconditionally. When weights are supplied they scale the
    per-image density loss before averaging, which requires the criterion to
    be evaluated per image rather than batch-reduced, hence the loop.
    """
    if weights is None:
        loss = criterion(fused, target)
        if aux is not None:
            for a in aux:
                loss = loss + aux_weight * criterion(a, target)
        return loss

    total = fused.new_zeros(())
    batch = fused.shape[0]
    for i in range(batch):
        item = criterion(fused[i:i + 1], target[i:i + 1])
        if aux is not None:
            for a in aux:
                item = item + aux_weight * criterion(a[i:i + 1], target[i:i + 1])
        total = total + weights[i] * item
    return total / batch


def spearman_soft(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Differentiable-ish rank correlation, for monitoring only.

    The reported metric is scipy's exact Spearman on the validation fold. This
    version uses soft ranks via a sigmoid over pairwise differences so it can be
    logged every step without leaving the GPU; not meant to be optimized
    directly, since its gradient is dominated by near-ties and it will happily
    produce a good rank ordering with badly calibrated magnitudes.
    """
    def soft_rank(v):
        diff = v.unsqueeze(0) - v.unsqueeze(1)
        return torch.sigmoid(diff * 50.0).sum(dim=0)

    rp, rt = soft_rank(pred), soft_rank(target)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    return (rp * rt).sum() / (rp.norm() * rt.norm() + EPS)

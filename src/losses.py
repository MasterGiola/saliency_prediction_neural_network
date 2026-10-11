"""
Loss functions for saliency prediction.

Starts with MSE to validate the pipeline, then moves to KLDiv (or a
KLDiv+CC combo), which the saliency literature generally finds better suited
to distribution-like targets than plain pixel-wise regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def _normalize_to_distribution(x):
    """Flatten spatial dims and normalize each map to sum to 1."""
    b = x.shape[0]
    x = x.view(b, -1)
    x = x / (x.sum(dim=1, keepdim=True) + EPS)
    return x


def kldiv_loss(pred, target):
    """KL(target || pred), the standard saliency KLDiv formulation."""
    pred = _normalize_to_distribution(pred)
    target = _normalize_to_distribution(target)
    return (target * torch.log(target / (pred + EPS) + EPS)).sum(dim=1).mean()


def cc_loss(pred, target):
    """1 - Pearson correlation coefficient (so lower = better, like a loss)."""
    b = pred.shape[0]
    pred = pred.view(b, -1)
    target = target.view(b, -1)

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    numerator = (pred * target).sum(dim=1)
    denominator = torch.sqrt((pred ** 2).sum(dim=1) * (target ** 2).sum(dim=1)) + EPS
    cc = numerator / denominator
    return (1 - cc).mean()


class SaliencyLoss(nn.Module):
    """
    config["train"]["loss"] options:
      "mse"          - plain MSE (good for a first sanity-check run only —
                       see model.FinalSaliencyHead's peak-renormalization note;
                       the smoothing head is scale-invariant under kldiv/cc but
                       not under mse)
      "kldiv"        - KL divergence between normalized maps
      "kldiv_cc"     - weighted combination of KLDiv and CC (recommended,
                       and the default here / in config.yaml)
    """

    def __init__(self, loss_type="kldiv_cc", kldiv_weight=1.0, cc_weight=0.5):
        super().__init__()
        self.loss_type = loss_type
        self.kldiv_weight = kldiv_weight
        self.cc_weight = cc_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        if self.loss_type == "mse":
            return self.mse(pred, target)
        if self.loss_type == "kldiv":
            return kldiv_loss(pred, target)
        if self.loss_type == "kldiv_cc":
            return self.kldiv_weight * kldiv_loss(pred, target) + self.cc_weight * cc_loss(pred, target)
        raise ValueError(f"Unknown loss_type: {self.loss_type}")


def deep_supervision_loss(criterion, fused, aux, target, aux_weight=1.0):
    """
    Applies the main criterion to the fused output AND to each auxiliary
    (per-depth) output, per the DVA paper's deep supervision.

    Wang & Shen (TIP'18) Eq. 9 minimizes

        (1/M) * Q(W, w_c)  +  P(W, w_c, w_f)

    where Q = sum_m L(W, w_c^m) is the sum over the M per-scale output losses
    and P is the fusion-layer loss. So the auxiliary term is the *mean* over
    branches (weight 1/M each), and the fused term has weight 1 — meaning the
    total auxiliary contribution stays at 1.0 no matter how many branches you
    use.

    That's why this averages rather than summing with a fixed per-branch
    constant: a fixed constant makes the aux contribution scale as
    aux_weight * M, so changing model.out_indices would silently reweight the
    whole objective. Averaging keeps the balance invariant to M, matching the
    paper.

    aux_weight scales the (already averaged) auxiliary term:
      1.0  -> exactly Eq. 9
      <1.0 -> weaker deep supervision than the paper
      0.0  -> fusion-loss-only, i.e. the paper's "w/o deep supervision"
              ablation (Table IX)

    Note: DVA's L is per-pixel cross-entropy. This codebase substitutes a
    distribution-based loss (KLDiv / KLDiv+CC) via SaliencyLoss, so this
    reproduces Eq. 9's *structure* with a different inner loss — see
    Jetley et al. (CVPR'16) on probability-distance losses for saliency.

    Only meaningful for intermediate_fusion / advanced_transformer models;
    for base_resnet, aux is None and this reduces to criterion(fused, target).
    """
    loss = criterion(fused, target)
    if aux:
        aux_loss = sum(criterion(a, target) for a in aux) / len(aux)
        loss = loss + aux_weight * aux_loss
    return loss

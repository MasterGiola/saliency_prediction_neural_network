"""
Observer-agreement head, bolted onto the existing saliency models.

model.py is not modified. TwoHeadSaliencyNet wraps whatever build_model()
returns and hangs a second head off the same encoder, so all three tiers
(base_resnet / intermediate_fusion / advanced_transformer) gain an agreement
output without touching code that already works and already has checkpoints.

How the wrapping works
----------------------
Both BaseSaliencyNet and FusionSaliencyNet begin forward() with
`feats = self.encoder(x)`, where `.encoder` is a timm features_only backbone
returning a list of stage outputs. A forward hook is registered on that
module to stash its output, so one call to `base(x)` produces the saliency
map and leaves the encoder pyramid available for head B, with no second
forward pass and no duplicated compute.

Return type follows the convention already used by FusionSaliencyNet and
handled by unwrap_prediction():

    {"fused": [B,1,H,W], "aux": [...] or None, "agreement": [B] or [B,1,g,g]}

so train.py / evaluate.py keep working unchanged on the first two keys.

Why head B is deliberately tiny
--------------------------------
It trains on MIT1003 — 1003 images, and only ~800 after holding out a fold.
The scalar head is a couple of Linear layers over a pooled feature vector,
on the order of 1e4 parameters, since anything larger memorizes the training
fold and the cross-validated correlation collapses. The dense variant exists
for once the scalar version has already been shown to work.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import build_model


class AgreementHead(nn.Module):
    """
    Encoder features -> predicted observer disagreement.

    Two output modes:

      "scalar" : one number per image, predicting js_excess. Start here.
      "dense"  : a [1, grid, grid] map, predicting per-pixel disagreement at
                 the same grid the targets were computed on (64 by default).
                 Much noisier supervision, since per-pixel JS from 15
                 observers is a sparse estimate, so this mode is only worth
                 reaching for once the scalar head clears its baseline.

    Predicts both a mean and a log-variance, since the loss is a Gaussian
    NLL: the head has to be able to say "I don't know" on images where the
    target itself is unreliable, otherwise those images dominate the gradient.
    """

    def __init__(self, in_ch: int, mode: str = "scalar", hidden: int = 64,
                 grid: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mode = mode
        self.grid = grid

        if mode == "scalar":
            self.trunk = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            # 2 outputs: predicted mean, predicted log-variance
            self.out = nn.Linear(hidden, 2)
        elif mode == "dense":
            self.trunk = nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            )
            self.out = nn.Conv2d(hidden, 2, kernel_size=1)
        else:
            raise ValueError(f"Unknown agreement head mode: {mode}")

    def forward(self, feat: torch.Tensor):
        """-> (mean, log_var), each [B] in scalar mode or [B,1,grid,grid] in dense mode."""
        h = self.trunk(feat)
        y = self.out(h)

        if self.mode == "scalar":
            return y[:, 0], y[:, 1]

        y = F.interpolate(y, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return y[:, 0:1], y[:, 1:2]


class TwoHeadSaliencyNet(nn.Module):
    """
    Wraps a built saliency model and adds an agreement head on the shared encoder.

    Typical use (stage 2 of the training protocol):

        base = build_model(config)
        load_checkpoint(base, ..., path=salicon_checkpoint)
        model = TwoHeadSaliencyNet(base, config)
        model.freeze_base()          # encoder + head A frozen; only head B trains

    freeze_base() is what keeps the MIT1003 density metrics a clean
    generalization test: head A never sees a MIT1003 gradient, so its numbers
    on that set stay exactly as reportable as they are today.
    """

    def __init__(self, base_model: nn.Module, config: dict):
        super().__init__()
        agr_cfg = config.get("agreement", {})

        self.base = base_model
        self.mode = agr_cfg.get("head_mode", "scalar")

        if not hasattr(base_model, "encoder"):
            raise AttributeError(
                "TwoHeadSaliencyNet needs base_model.encoder (the timm features_only "
                "backbone). Both BaseSaliencyNet and FusionSaliencyNet expose it."
            )

        # Capture the encoder pyramid as a side effect of the base forward pass.
        self._feats = None
        self.base.encoder.register_forward_hook(self._capture)

        in_ch = self._deepest_channels(base_model)
        self.agreement_head = AgreementHead(
            in_ch=in_ch,
            mode=self.mode,
            hidden=agr_cfg.get("head_hidden", 64),
            grid=agr_cfg.get("grid", 64),
            dropout=agr_cfg.get("head_dropout", 0.2),
        )

    def _capture(self, module, inputs, output):
        self._feats = output

    @staticmethod
    def _deepest_channels(base_model: nn.Module) -> int:
        """Channel count of the deepest encoder stage, via timm's feature_info."""
        encoder = base_model.encoder
        if hasattr(encoder, "feature_info"):
            info = encoder.feature_info
            channels = info.channels() if callable(getattr(info, "channels", None)) else info
            return int(channels[-1])
        raise AttributeError(
            "Could not read encoder channel widths. Pass in_ch explicitly if you "
            "are using a backbone without timm feature_info."
        )

    def freeze_base(self) -> None:
        """Freeze encoder + decoder + head A. Only the agreement head stays trainable."""
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    def unfreeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad = True
        self.base.train()

    def train(self, mode: bool = True):
        """
        Keep the frozen base in eval() even when the wrapper is put in train().

        Without this, BatchNorm layers in the frozen encoder would keep updating
        their running statistics from MIT1003 batches, quietly training on the
        evaluation set and invalidating the density metrics.
        """
        super().train(mode)
        base_frozen = not any(p.requires_grad for p in self.base.parameters())
        if base_frozen:
            self.base.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict:
        pred = self.base(x)

        feats = self._feats
        if feats is None:
            raise RuntimeError("Encoder hook did not fire — did base.forward() change?")
        deepest = feats[-1] if isinstance(feats, (list, tuple)) else feats

        mean, log_var = self.agreement_head(deepest)

        if isinstance(pred, dict):
            out = dict(pred)
        else:
            out = {"fused": pred, "aux": None}
        out["agreement"] = mean
        out["agreement_log_var"] = log_var
        return out


def build_two_head_model(config: dict, base_checkpoint: str = None,
                         device: torch.device = None) -> TwoHeadSaliencyNet:
    """
    Factory mirroring build_model(): config in, ready model out.

    `base_checkpoint` should be the SALICON-trained checkpoint from stage 1;
    training head B on a randomly initialized encoder would measure nothing
    useful, since the premise is that features already good enough to predict
    where people look also carry whether people agree.
    """
    base = build_model(config)

    if base_checkpoint:
        state = torch.load(base_checkpoint, map_location="cpu")
        base.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    model = TwoHeadSaliencyNet(base, config)
    if config.get("agreement", {}).get("freeze_base", True):
        model.freeze_base()
    if device is not None:
        model = model.to(device)
    return model

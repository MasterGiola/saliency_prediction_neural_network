"""
Saliency prediction models.

Three tiers, selected via config["model"]["name"]:
  - "base_resnet"          : ResNet encoder + U-Net-style decoder (skip connections
                              from every encoder stage, not just the deepest one)
  - "intermediate_fusion"  : multi-depth feature fusion + per-branch U-Net skip
                              connections + deep supervision
  - "advanced_transformer" : same fusion/skip decoder as intermediate, hierarchical
                              transformer encoder

All three share the same interface: forward(x) returns EITHER
  - a single tensor [B, 1, H, W]                      (base_resnet)
  - a dict {"fused": ..., "aux": [s1, s2, s3]}          (intermediate / advanced, for deep supervision)
train.py / evaluate.py both handle either return type (see `unwrap_prediction`).

Set model.use_skip_connections: false in config to disable skip connections on
either tier, for an A/B comparison against this behavior.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class UpBlock(nn.Module):
    """One decoder stage: upsample x2, optionally concat a skip connection, then conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        conv_in = in_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def saliency_head(in_ch):
    """1x1 conv + sigmoid -> single-channel saliency map in [0, 1].
    Used for the auxiliary/per-depth deep-supervision heads in FusionSaliencyNet,
    which stay plain (no center bias / smoothing): they're discarded at inference
    and only exist to shape gradients into the shallower encoder stages, so it's
    not worth the extra parameters/compute there."""
    return nn.Sequential(nn.Conv2d(in_ch, 1, kernel_size=1), nn.Sigmoid())


class CenterBiasPrior(nn.Module):
    """
    Learnable per-pixel bias added to the saliency logits before the final
    sigmoid, initialized as a centered Gaussian.

    Human fixations are heavily center-biased (viewers look near the middle
    of the frame far more than chance, regardless of image content), and
    letting the network learn a per-pixel prior on top of the image-driven
    prediction is a cheap, well-documented way to pick up that signal without
    asking the image features alone to encode "the middle is usually
    important". Added in logit space (pre-sigmoid) so it acts like a prior on
    log-odds, which composes cleanly with whatever the conv head predicts.

    The prior is stored at a small fixed resolution (`grid_size`) and
    bilinearly upsampled to the map's actual size at forward time. Keeping it
    low-res limits it to a handful of parameters — enough to shift/reshape
    the bias if the data wants that, but not enough to memorize per-pixel
    training-set statistics the way a full-resolution learned bias could.
    """

    def __init__(self, grid_size: int = 16, init_sigma: float = 0.3):
        super().__init__()
        ys = torch.linspace(-1, 1, grid_size)
        xs = torch.linspace(-1, 1, grid_size)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * init_sigma ** 2))
        # logit() rather than the raw gaussian so the *initial* contribution of this
        # module, once passed back through sigmoid downstream, reproduces the smooth
        # bump we just built instead of a saturated/clipped version of it.
        init_bias = torch.logit(gaussian.clamp(1e-4, 1 - 1e-4))
        self.bias = nn.Parameter(init_bias.unsqueeze(0).unsqueeze(0))  # [1, 1, grid, grid]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        bias = F.interpolate(self.bias, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        return logits + bias


class GaussianBlur2d(nn.Module):
    """
    Fixed-size Gaussian blur applied to the final [B, 1, H, W] saliency map.

    Ground-truth saliency maps (SALICON/MIT) are themselves blurred fixation
    density maps — that blur is part of how the target distribution is
    defined, not noise. A decoder head trained with a plain per-pixel conv +
    sigmoid tends to produce a spatially sharper/noisier map than that, which
    KLDiv/CC/SIM all penalize even when the "shape" of the prediction is
    basically right. Smoothing the prediction to roughly the same scale as
    the target is standard post-processing in the saliency literature and
    typically improves KLDiv/SIM/CC together.

    `sigma` is in output-map pixels (e.g. sigma=3 at a 256x256 output size).
    Set `learnable_sigma=True` to let training adjust the blur amount instead
    of fixing it; sigma is parametrized as log-sigma so it can't collapse to
    (or go below) zero during training.
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 3.0, learnable_sigma: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd so the blur stays centered"
        self.kernel_size = kernel_size
        log_sigma = torch.log(torch.tensor(float(sigma)))
        if learnable_sigma:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

    def _kernel(self, device, dtype):
        sigma = self.log_sigma.exp().clamp(min=0.1)
        coords = torch.arange(self.kernel_size, device=device, dtype=dtype) - self.kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g[:, None] * g[None, :]
        return kernel_2d.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._kernel(x.device, x.dtype)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2)


class FinalSaliencyHead(nn.Module):
    """
    Produces the model's actual final saliency map (unlike the plain
    per-stage aux heads above): 1x1 conv -> optional learned center-bias
    prior (pre-sigmoid) -> sigmoid -> optional Gaussian smoothing.

    Both post-processing stages are individually switchable via config so
    you can ablate them against the plain head if you want to check how much
    each is actually contributing.
    """

    def __init__(self, in_ch, use_center_bias=True, center_bias_grid=16,
                 use_smoothing=True, smoothing_kernel=15, smoothing_sigma=3.0,
                 smoothing_learnable=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.center_bias = CenterBiasPrior(grid_size=center_bias_grid) if use_center_bias else None
        self.sigmoid = nn.Sigmoid()
        self.blur = (
            GaussianBlur2d(kernel_size=smoothing_kernel, sigma=smoothing_sigma,
                            learnable_sigma=smoothing_learnable)
            if use_smoothing else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        if self.center_bias is not None:
            logits = self.center_bias(logits)
        out = self.sigmoid(logits)
        if self.blur is not None:
            out = self.blur(out)
            # blurring is a weighted average so it can only pull values toward the
            # mean, never past the input's own max — rescale back up so the map
            # still spans close to [0, 1] (matches the rest of the codebase's
            # "saliency map in [0, 1]" contract, e.g. for src.visualize display).
            #
            # peak is DETACHED: torch.amax is differentiable, and left attached it
            # routes gradient from every pixel back through the single argmax pixel
            # (quotient-rule cross term). For KLDiv/CC — both scale-invariant —
            # that term is provably zero (Euler's identity for degree-0 homogeneous
            # functions), so detaching changes nothing there. For MSE it isn't zero:
            # measured cosine similarity of the gradient w.r.t. the attached version
            # is ~0.66, and the argmax pixel's own gradient can flip sign. Detaching
            # makes this purely the display-range rescaling the comment above
            # describes, regardless of which loss is configured.
            peak = out.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6).detach()
            out = out / peak
        return out


# ---------------------------------------------------------------------------
# Base Task: single-stream ResNet encoder + plain upsampling decoder
# ---------------------------------------------------------------------------

class BaseSaliencyNet(nn.Module):
    """
    U-Net-style encoder-decoder (Ronneberger et al., 2015): the encoder exposes
    every stage it has (not just the deepest one), and the decoder concatenates
    the matching shallower encoder feature map at each upsampling step instead
    of upsampling from the bottleneck alone. Repeated downsampling in the
    encoder throws away fine spatial detail (edges, small/peripheral objects);
    skip connections hand that detail straight to the decoder at the resolution
    where it still exists, rather than asking the decoder to hallucinate it
    back from a heavily pooled representation.

    Set use_skip_connections=False to fall back to the old plain-upsampling
    decoder (bottleneck only, no skips) for an A/B comparison.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, head_kwargs=None,
                 use_skip_connections=True):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        # features_only=True with no out_indices given returns EVERY stage the
        # backbone exposes (shallow -> deep). We need all of them now, not just
        # the deepest one, since the shallow ones are the skip sources.
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        chs = self.encoder.feature_info.channels()  # shallow -> deep, e.g. [64, 64, 128, 256, 512]

        self.up_blocks = nn.ModuleList()
        if use_skip_connections:
            # deepest feature is the decoder's starting point; every shallower stage
            # becomes a skip, consumed deepest-to-shallowest (i.e. in the order the
            # decoder actually walks back up through them).
            skip_chs = list(reversed(chs[:-1]))  # e.g. [256, 128, 64, 64]
            cur_ch = chs[-1]
            for skip_ch in skip_chs:
                out_ch = max(skip_ch, 16)
                self.up_blocks.append(UpBlock(cur_ch, skip_ch, out_ch))
                cur_ch = out_ch
            # one more upsample with no skip: the shallowest encoder feature is at
            # stride reductions[0] (e.g. 2), not stride 1, so this last step reaches
            # full input resolution the same way the old fixed 5-block decoder did.
            self.final_up = UpBlock(cur_ch, 0, max(cur_ch // 2, 16))
            cur_ch = max(cur_ch // 2, 16)
        else:
            # old behavior: 5 plain upsamples with the same fixed channel targets the
            # original implementation used, regardless of this backbone's channel width
            fixed_targets = [256, 128, 64, 32, 16]
            cur_ch = chs[-1]
            for out_ch in fixed_targets:
                self.up_blocks.append(UpBlock(cur_ch, 0, out_ch))
                cur_ch = out_ch
            self.final_up = None

        self.head = FinalSaliencyHead(cur_ch, **(head_kwargs or {}))

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.encoder(x)  # shallow -> deep

        out = feats[-1]
        if self.use_skip_connections:
            skips = list(reversed(feats[:-1]))  # deepest-to-shallowest, matches self.up_blocks order
            for up_block, skip in zip(self.up_blocks, skips):
                out = up_block(out, skip)
            out = self.final_up(out)
        else:
            for up_block in self.up_blocks:
                out = up_block(out)

        out = self.head(out)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out  # [B, 1, H, W]


# ---------------------------------------------------------------------------
# Intermediate Task: multi-depth feature fusion + skip connections + deep supervision
# (DVA-style multi-level fusion, extended with U-Net-style skip connections)
# ---------------------------------------------------------------------------

class FusionSaliencyNet(nn.Module):
    """
    Works with any timm backbone that supports features_only=True and exposes
    multiple stages (CNNs like resnet* or hierarchical transformers like pvt_v2*,
    swin*, mobilevit*). Pulls `out_indices` intermediate stages, decodes each into
    its own saliency map (deep supervision), and learns a weighted fusion of them.

    Each decoder branch is its own mini U-Net: the encoder exposes every stage it
    has (not just the ones in `out_indices`), and as a branch upsamples its stage
    back to full resolution, it concatenates whatever shallower encoder stage
    matches each intermediate resolution — the same skip-connection pattern as
    BaseSaliencyNet, just repeated once per deep-supervision branch. A branch
    only runs out of skips once it upsamples past the shallowest stage the
    backbone actually exposes (e.g. pvt_v2's stages start at stride 4, so the
    last one or two upsample steps of any branch have no skip to attach — that's
    an inherent limit of the backbone's pyramid, not a bug).

    Set use_skip_connections=False to fall back to the old independent per-branch
    upsampling (no skips at all) for an A/B comparison.
    """

    def __init__(self, backbone_name="resnet34", pretrained=True, out_indices=(2, 3, 4),
                 head_kwargs=None, use_skip_connections=True):
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.use_skip_connections = use_skip_connections

        # features_only=True with no out_indices given returns the backbone's FULL
        # stage pyramid (shallow -> deep) — out_indices then just picks, by position
        # in that full list, which stages get their own decoder branch + aux map;
        # the stages NOT picked are still returned and used purely as skip sources.
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        all_chs = self.encoder.feature_info.channels()          # shallow -> deep, ALL stages
        all_reductions = self.encoder.feature_info.reduction()  # shallow -> deep, ALL stages

        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.skip_indices = []  # per branch: list of full-pyramid indices, deepest-first

        for idx in self.out_indices:
            ch = all_chs[idx]
            reduction = all_reductions[idx]
            # number of x2 upsamples needed to bring this stage back to input resolution
            n_ups = max(int(round(math.log2(reduction))), 1)

            if use_skip_connections:
                # every shallower stage the backbone exposes, deepest-first (i.e. in the
                # order this branch's upsampling actually walks back through them) —
                # capped to n_ups since that's as many upsample steps as this branch has
                shallower = [j for j in range(idx) if all_reductions[j] < reduction]
                shallower.sort(key=lambda j: -all_reductions[j])
                branch_skip_idx = shallower[:n_ups]
            else:
                branch_skip_idx = []
            self.skip_indices.append(branch_skip_idx)

            blocks = nn.ModuleList()
            cur_ch = ch
            for step in range(n_ups):
                skip_ch = all_chs[branch_skip_idx[step]] if step < len(branch_skip_idx) else 0
                out_ch = max(cur_ch // 2, 16)
                blocks.append(UpBlock(cur_ch, skip_ch, out_ch))
                cur_ch = out_ch
            self.decoders.append(blocks)
            self.heads.append(saliency_head(cur_ch))

        # in_ch = len(out_indices): the fusion head reads the stacked per-branch maps
        # (one channel each), same as before.
        self.final_head = FinalSaliencyHead(len(self.out_indices), **(head_kwargs or {}))

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.encoder(x)  # ALL stages, shallow -> deep

        aux_maps = []
        for branch_idx, idx in enumerate(self.out_indices):
            out = feats[idx]
            branch_skip_idx = self.skip_indices[branch_idx]
            for step, up_block in enumerate(self.decoders[branch_idx]):
                skip = feats[branch_skip_idx[step]] if step < len(branch_skip_idx) else None
                out = up_block(out, skip)
            out = self.heads[branch_idx](out)
            if out.shape[-2:] != input_size:
                out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
            aux_maps.append(out)

        stacked = torch.cat(aux_maps, dim=1)  # [B, len(out_indices), H, W]
        fused = self.final_head(stacked)  # [B, 1, H, W]

        return {"fused": fused, "aux": aux_maps}


# ---------------------------------------------------------------------------
# Advanced Task: same fusion decoder, hierarchical transformer encoder
# ---------------------------------------------------------------------------

def build_transformer_backbone(name="pvt_v2_b0", pretrained=True):
    """
    Hierarchical (multi-stage) lightweight transformers work as drop-in
    replacements for the CNN backbone above, because they also expose
    features_only with decreasing spatial resolution per stage:
      - pvt_v2_b0   (~3.7M params)
      - swin_tiny_patch4_window7_224
      - mobilevit_xs
    A plain (non-hierarchical) ViT does NOT work here — it has a single fixed
    patch resolution throughout, so there's no natural multi-scale pyramid to
    feed the fusion decoder.
    """
    return name  # just returns the timm name; FusionSaliencyNet builds it


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _head_kwargs_from_config(model_cfg: dict) -> dict:
    """Pulls the FinalSaliencyHead options out of config["model"], with the
    same defaults as FinalSaliencyHead itself (both post-processing stages
    on) so leaving them unset in an existing config.yaml opts you in rather
    than silently keeping the old plain head."""
    return dict(
        use_center_bias=model_cfg.get("use_center_bias", True),
        center_bias_grid=model_cfg.get("center_bias_grid", 16),
        use_smoothing=model_cfg.get("use_output_smoothing", True),
        smoothing_kernel=model_cfg.get("smoothing_kernel", 15),
        smoothing_sigma=model_cfg.get("smoothing_sigma", 3.0),
        smoothing_learnable=model_cfg.get("smoothing_learnable", False),
    )


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    name = model_cfg["name"]
    head_kwargs = _head_kwargs_from_config(model_cfg)
    use_skip_connections = model_cfg.get("use_skip_connections", True)

    if name == "base_resnet":
        return BaseSaliencyNet(
            backbone_name=model_cfg.get("backbone", "resnet18"),
            pretrained=model_cfg.get("pretrained", True),
            head_kwargs=head_kwargs,
            use_skip_connections=use_skip_connections,
        )

    if name == "intermediate_fusion":
        return FusionSaliencyNet(
            backbone_name=model_cfg.get("backbone", "resnet34"),
            pretrained=model_cfg.get("pretrained", True),
            out_indices=tuple(model_cfg.get("out_indices", (2, 3, 4))),
            head_kwargs=head_kwargs,
            use_skip_connections=use_skip_connections,
        )

    if name == "advanced_transformer":
        backbone = build_transformer_backbone(model_cfg.get("backbone", "pvt_v2_b0"))
        return FusionSaliencyNet(
            backbone_name=backbone,
            pretrained=model_cfg.get("pretrained", True),
            out_indices=tuple(model_cfg.get("out_indices", (1, 2, 3))),
            head_kwargs=head_kwargs,
            use_skip_connections=use_skip_connections,
        )

    raise ValueError(f"Unknown model name: {name}")


def unwrap_prediction(pred):
    """Both return types -> (fused_map, aux_maps_or_None). Used by train/evaluate."""
    if isinstance(pred, dict):
        return pred["fused"], pred["aux"]
    return pred, None

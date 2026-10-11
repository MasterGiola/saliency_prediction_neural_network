"""
Experiment orchestration: multi-seed tier sweeps and diagnostics, standalone
from project_notebook.ipynb (which reimplements its own idempotent training
loop instead, so it can skip variants whose checkpoint already exists —
these functions always retrain unconditionally).

Provides:

  inspect_fusion_weights, check_convergence   — cheap diagnostics on an existing run
  run_seed_sweep, summarize_seed_sweep        — multi-seed tier comparison
  compare_tiers_on_agreement                  — agreement head across tiers

Agreement-target computation and agreement-head training are covered by
src.agreement_targets and src.train_agreement respectively.

Checkpoint naming
-----------------
src.utils.build_run_name is model + backbone + loss, with no seed component,
so a seed sweep sharing one checkpoint_dir would have every run overwrite the
last. Each run therefore gets its own subdirectory:

    <checkpoint_root>/<tier>_seed<N>/<run_name>_best.pt

which keeps src.train.train's own naming and resume logic completely intact.
"""
import gc
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from src.data import get_dataloaders
from src.model import build_model
from src.train import train
from src.utils import build_run_name, get_device, set_seed

# Backbone per tier. Kept here rather than in config.yaml because a sweep needs
# to vary the tier while holding everything else fixed, and each tier has a
# different sensible backbone (see the comments in config.yaml).
TIER_BACKBONES = {
    "base_resnet": "resnet34",
    "intermediate_fusion": "resnet34",
    "advanced_transformer": "pvt_v2_b0",
}

TIER_OUT_INDICES = {
    "base_resnet": None,  # unused by BaseSaliencyNet
    "intermediate_fusion": [2, 3, 4],
    "advanced_transformer": [1, 2, 3],
}

METRIC_KEYS = ["cc", "nss", "sim", "kldiv", "auc_judd"]


# ---------------------------------------------------------------------------
# Phase 0 — diagnostics
# ---------------------------------------------------------------------------

def inspect_fusion_weights(model) -> np.ndarray:
    """
    Print the learned per-branch mixing weights of a FusionSaliencyNet.

    FinalSaliencyHead.conv is Conv2d(len(out_indices), 1, kernel_size=1), so it
    holds one scalar per branch, shared across every pixel. Those scalars say
    directly how much the model uses each depth: weights concentrated on the
    last entry mean the model has learned to ignore the shallow branches and
    the fusion is effectively inert.
    """
    head = getattr(model, "final_head", None)
    if head is None or not hasattr(head, "conv"):
        print("Not a FusionSaliencyNet (no final_head.conv) — nothing to inspect.")
        return None

    weights = head.conv.weight.detach().cpu().numpy().reshape(-1)
    bias = float(head.conv.bias.detach().cpu().numpy().reshape(-1)[0])
    share = np.abs(weights) / (np.abs(weights).sum() + 1e-8)

    out_indices = getattr(model, "out_indices", tuple(range(len(weights))))
    print("Learned fusion weights (one scalar per branch, shared over all pixels):")
    for idx, w, s in zip(out_indices, weights, share):
        bar = "#" * int(round(s * 40))
        print(f"  encoder stage {idx}:  w = {w:+.4f}   |share| = {s:5.1%}  {bar}")
    print(f"  bias: {bias:+.4f}")

    dominant = share.max()
    if dominant > 0.70:
        print(f"\n  >> One branch carries {dominant:.0%} of the mixture. The fusion is "
              f"close to inert — this alone explains a tie with base_resnet.")
    else:
        print(f"\n  >> Mixture is spread (max share {dominant:.0%}). Fusion is active; "
              f"look elsewhere for the tie.")
    return weights


def check_convergence(history: dict, tail: int = 5) -> dict:
    """
    Was training still improving when it stopped?

    Compares the mean of the last `tail` epochs against the `tail` before that.
    A train_loss still falling at the final epoch means the run was truncated,
    not converged, and comparing a converged model against an unconverged one
    says nothing about architecture.
    """
    result = {}
    for key in ("train_loss", "val_loss"):
        series = history.get(key)
        if not series or len(series) < 2 * tail:
            continue
        recent = float(np.mean(series[-tail:]))
        earlier = float(np.mean(series[-2 * tail:-tail]))
        rel = (earlier - recent) / (abs(earlier) + 1e-8)
        result[key] = {"earlier": earlier, "recent": recent, "rel_improvement": rel}
        verdict = "STILL IMPROVING — train longer" if rel > 0.01 else "converged"
        print(f"{key:<11} last-{tail} mean {recent:.5f} vs previous-{tail} {earlier:.5f} "
              f"({rel:+.2%})  ->  {verdict}")
    if not result:
        print(f"History too short to judge (need >= {2 * tail} epochs).")
    return result


# ---------------------------------------------------------------------------
# Phase 1 — multi-seed tier sweep
# ---------------------------------------------------------------------------

def make_run_config(config: dict, tier: str, seed: int, checkpoint_root: str,
                    epochs: int = None, loss: str = "kldiv_cc") -> dict:
    """Deep-copied config for one (tier, seed) run, with its own checkpoint dir."""
    cfg = deepcopy(config)
    cfg["seed"] = seed
    cfg["model"]["name"] = tier
    cfg["model"]["backbone"] = TIER_BACKBONES[tier]
    if TIER_OUT_INDICES[tier] is not None:
        cfg["model"]["out_indices"] = TIER_OUT_INDICES[tier]

    cfg["train"]["loss"] = loss
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    cfg["train"]["checkpoint_dir"] = str(Path(checkpoint_root) / f"{tier}_seed{seed}")
    cfg["train"]["resume_from"] = None
    return cfg


def stage1_checkpoint_path(config: dict) -> str:
    """Where train() wrote this run's best checkpoint."""
    return str(Path(config["train"]["checkpoint_dir"]) / f"{build_run_name(config)}_best.pt")


def run_seed_sweep(config: dict, tiers=None, seeds=(0, 1, 2), epochs: int = None,
                   checkpoint_root: str = "/content/checkpoints/sweep",
                   loss: str = "kldiv_cc") -> list:
    """
    Train every (tier, seed) combination and return one record per run.

    This produces both the tier-comparison result and the per-tier
    checkpoints that compare_tiers_on_agreement needs in phase 4.

    Loaders are built once and reused, since the data config does not vary
    across runs and rebuilding them per run would waste several minutes each
    time. Shuffling still differs per run because set_seed() reseeds the
    global torch generator the DataLoader draws from.
    """
    tiers = tiers or list(TIER_BACKBONES)
    device = get_device()
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
          + f"\ntiers: {tiers}   seeds: {list(seeds)}   "
            f"({len(tiers) * len(seeds)} runs)\n")

    train_loader, val_loader = get_dataloaders(config)
    results = []

    for tier in tiers:
        for seed in seeds:
            print("=" * 70)
            print(f"RUN  tier={tier}  seed={seed}")
            print("=" * 70)

            cfg = make_run_config(config, tier, seed, checkpoint_root, epochs, loss)
            set_seed(seed)

            model = build_model(cfg)
            model, history = train(model, train_loader, val_loader, cfg)

            record = {
                "tier": tier,
                "seed": seed,
                "checkpoint": stage1_checkpoint_path(cfg),
                "history": history,
                "epochs_run": len(history.get("train_loss", [])),
            }
            for key in METRIC_KEYS:
                series = history.get(key)
                record[key] = float(series[-1]) if series else float("nan")
            record["val_loss"] = (float(history["val_loss"][-1])
                                  if history.get("val_loss") else float("nan"))
            results.append(record)

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def summarize_seed_sweep(results: list, metric: str = "cc") -> dict:
    """
    Mean +/- sd per tier, plus pairwise significance.

    Shows whether the between-tier difference sits inside the between-seed
    spread — a single run per tier cannot establish that either way.
    """
    from scipy.stats import ttest_ind

    tiers = sorted({r["tier"] for r in results}, key=lambda t: list(TIER_BACKBONES).index(t))
    by_tier = {t: np.array([r[metric] for r in results if r["tier"] == t]) for t in tiers}

    print(f"\n{metric.upper()} across seeds")
    print("-" * 58)
    for tier in tiers:
        values = by_tier[tier]
        print(f"  {tier:<24} {values.mean():.4f} +/- {values.std(ddof=1):.4f}   "
              f"n={len(values)}  {np.round(values, 4).tolist()}")

    print("\npairwise comparisons")
    print("-" * 58)
    verdicts = {}
    for i, a in enumerate(tiers):
        for b in tiers[i + 1:]:
            if len(by_tier[a]) < 2 or len(by_tier[b]) < 2:
                continue
            diff = by_tier[a].mean() - by_tier[b].mean()
            p = float(ttest_ind(by_tier[a], by_tier[b], equal_var=False).pvalue)
            call = "DISTINGUISHABLE" if p < 0.05 else "indistinguishable"
            print(f"  {a} vs {b}:  diff {diff:+.4f}   p = {p:.3f}   {call}")
            verdicts[(a, b)] = {"diff": diff, "p": p}

    pooled_sd = float(np.mean([by_tier[t].std(ddof=1) for t in tiers if len(by_tier[t]) > 1]))
    spread = float(max(v.mean() for v in by_tier.values())
                   - min(v.mean() for v in by_tier.values()))
    print(f"\n  between-tier spread of means : {spread:.4f}")
    print(f"  typical between-seed sd      : {pooled_sd:.4f}")
    if spread < pooled_sd:
        print("  >> Architecture differences are SMALLER than seed noise. This is the "
              "result that motivates the agreement work.")
    else:
        print("  >> Architecture differences exceed seed noise — report them as real.")

    return {"by_tier": by_tier, "pairwise": verdicts,
            "spread": spread, "pooled_sd": pooled_sd}


# ---------------------------------------------------------------------------
# Phase 4 — do the tiers separate on the new target?
# ---------------------------------------------------------------------------

def compare_tiers_on_agreement(config: dict, sweep_results: list, seed: int = 0,
                               skip_baseline: bool = False) -> dict:
    """
    Train head B on each tier's stage-1 checkpoint and compare.

    The hypothesis is directional: disagreement arises when multiple spatially
    separated regions compete, which is a long-range global property, and
    self-attention is the mechanism for exactly that, so advanced_transformer
    is expected to predict agreement better than base_resnet even where they
    tie on CC.

    Head A stays frozen throughout, so each tier's density metrics are untouched
    and the comparison is attributable purely to the encoder representation.
    """
    from src.train_agreement import run_cross_validation

    checkpoints = {r["tier"]: r["checkpoint"] for r in sweep_results if r["seed"] == seed}
    if not checkpoints:
        raise ValueError(f"No sweep results for seed={seed}.")

    device = get_device()
    out = {}

    for tier, ckpt in checkpoints.items():
        if not Path(ckpt).exists():
            print(f"[skip] {tier}: checkpoint not found at {ckpt}")
            continue

        print("\n" + "=" * 70)
        print(f"AGREEMENT HEAD on {tier}")
        print("=" * 70)

        cfg = deepcopy(config)
        cfg["model"]["name"] = tier
        cfg["model"]["backbone"] = TIER_BACKBONES[tier]
        if TIER_OUT_INDICES[tier] is not None:
            cfg["model"]["out_indices"] = TIER_OUT_INDICES[tier]
        cfg["agreement"]["base_checkpoint"] = ckpt
        cfg["agreement"]["freeze_base"] = True
        cfg["agreement"]["checkpoint_dir"] = str(
            Path(cfg["agreement"].get("checkpoint_dir", "checkpoints/agreement")) / tier)

        out[tier] = run_cross_validation(cfg, device=device, skip_baseline=skip_baseline)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("TIER COMPARISON ON OBSERVER AGREEMENT")
    print("=" * 70)
    print(f"{'tier':<24}{'pooled rho':>12}{'mean +/- sd':>20}{'% of ceiling':>15}")
    print("-" * 71)
    for tier, res in out.items():
        per_fold = np.array(res["per_fold_spearman"])
        ceiling = res.get("noise_ceiling", float("nan"))
        frac = res["pooled_spearman"] / ceiling if ceiling and np.isfinite(ceiling) else float("nan")
        print(f"{tier:<24}{res['pooled_spearman']:>12.4f}"
              f"{per_fold.mean():>13.4f} +/- {per_fold.std(ddof=1):.3f}"
              f"{frac:>14.1%}")

    baselines = [r.get("baseline_spearman") for r in out.values()]
    baseline = next((b for b in baselines if b is not None and np.isfinite(b)), float("nan"))
    if np.isfinite(baseline):
        print(f"\n  three-feature baseline : {baseline:.4f}")
    ceilings = [r.get("noise_ceiling") for r in out.values()]
    ceiling = next((c for c in ceilings if c is not None and np.isfinite(c)), float("nan"))
    if np.isfinite(ceiling):
        print(f"  split-half ceiling     : {ceiling:.4f}")

    if len(out) > 1:
        values = {t: r["pooled_spearman"] for t, r in out.items()}
        best, worst = max(values, key=values.get), min(values, key=values.get)
        gap = values[best] - values[worst]
        print(f"\n  best: {best} ({values[best]:.4f})   worst: {worst} ({values[worst]:.4f})"
              f"   gap: {gap:.4f}")
        print("  >> " + ("Tiers SEPARATE on agreement where they tied on density."
                         if gap > 0.05 else
                         "Tiers do NOT separate on agreement either — also a reportable "
                         "finding about what these representations encode."))

    return out

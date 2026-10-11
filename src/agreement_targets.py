"""
Observer-agreement targets, computed offline and cached.

Nothing here is differentiable and nothing changes during training, so every
quantity is computed once per image and written to an .npz. train_agreement.py
reads that file; it never touches raw fixation data.

Four quantities per image:

  ioc_nss     Inter-observer congruency. Build a density from N-1 observers,
              score the held-out observer's discrete fixations with NSS,
              average over all N choices of who to hold out. This is the
              classic "human ceiling" of the saliency literature: how well one
              person is predicted by everybody else.

  js_observed Jensen-Shannon divergence across the N per-observer densities,
              in bits. The identity

                  JS = H(mean_i g_i) - mean_i H(g_i)

              decomposes the entropy of the aggregate map into disagreement
              between observers plus the average spread within one observer's
              own scanpath. Only the first term is new information; the second
              is already implied by the density the model predicts.

  js_null     The same JS under a permutation of observer labels: pool every
              fixation, redeal them into synthetic observers of identical
              sizes, recompute. See below for why this is the right null.

  js_excess   js_observed - js_null. The training target.

plus, for evaluation only:

  js_excess_half_a / _half_b
              js_excess computed independently on two disjoint halves of the
              observers. Correlating these across images and applying the
              Spearman-Brown correction gives the reliability of the full-N
              estimate, i.e. the maximum correlation any model could achieve
              against a target this noisy. A model score is uninterpretable
              without it.

--------------------------------------------------------------------------
Why permutation rather than resampling
--------------------------------------------------------------------------
With ~10 fixations per observer, two people drawn from an identical underlying
distribution still produce visibly different densities. Raw js_observed
therefore measures true heterogeneity plus finite-sample noise, and at these
sample sizes the noise term is not small.

Permuting observer labels destroys the grouping while preserving the pooled
fixation set exactly. That has a useful consequence: H(mean_i g_i) is computed
from the same pooled sample either way, so it is identical under permutation
and cancels in the difference:

    js_excess = mean_i H(g_i^permuted) - mean_i H(g_i^observed)

Real observers being individually more concentrated than random regroupings is
precisely what disagreement means (each person commits to their own subset of
the scene), so js_excess > 0 signals genuine between-person structure and
js_excess ~ 0 signals that the apparent spread is a sampling artifact. Both
terms are still computed explicitly below rather than using the cancellation,
because having js_observed and js_null separately is what the decision gate
below needs.

Usage:
    python -m src.agreement_targets \
        --dataset mit1003 --root /data/mit1003 \
        --out cache/agreement_mit1003.npz --n-observers 15
"""
import argparse
import time
from pathlib import Path

import numpy as np

from src.observers import (SIGMA_FRAC_1DVA, load_observers, render_density,
                           render_observer_densities)

EPS = 1e-12

# Grid the densities are discretized on. Entropy is resolution-dependent, so
# this constant is baked into every number this module produces — targets
# computed at different GRID values are not comparable. Recorded in the cache
# file so a mismatch is detectable rather than silent.
GRID = 64

# Permutation redraws for the null. 200 is enough for the mean to be stable to
# ~1% at N=15; the cost is linear and this is the bulk of the runtime.
N_PERMUTATIONS = 200


# ---------------------------------------------------------------------------
# Information-theoretic primitives (bits)
# ---------------------------------------------------------------------------

def entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits of a density that already sums to 1."""
    p = p.ravel()
    nz = p > EPS
    return float(-(p[nz] * np.log2(p[nz])).sum())


def jensen_shannon(densities: np.ndarray) -> float:
    """
    JS divergence across [n, grid, grid] densities, in bits.

    Equals H(mean) - mean(H), which is 0 when every observer has the identical
    density and grows as they concentrate on disjoint regions. Bounded above by
    log2(n).
    """
    mean_density = densities.mean(axis=0)
    return entropy(mean_density) - float(np.mean([entropy(d) for d in densities]))


def nss_at_fixations(density: np.ndarray, fixations: np.ndarray, grid: int = GRID) -> float:
    """
    Normalized Scanpath Saliency of discrete fixations against a density map.

    Same definition as metrics.nss in the existing pipeline (z-score the map,
    average its value at fixated locations) but taking fixation coordinates
    rather than a binary mask, since the real scanpath is always available
    here and thresholding a continuous map to fake one would throw away the
    precision this module exists to exploit.
    """
    if len(fixations) == 0:
        return 0.0
    std = density.std()
    if std < EPS:
        return 0.0
    z = (density - density.mean()) / std
    xs = np.clip((fixations[:, 0] * grid).astype(int), 0, grid - 1)
    ys = np.clip((fixations[:, 1] * grid).astype(int), 0, grid - 1)
    return float(z[ys, xs].mean())


# ---------------------------------------------------------------------------
# Per-image targets
# ---------------------------------------------------------------------------

def inter_observer_congruency(record, grid: int = GRID,
                              sigma_frac: float = SIGMA_FRAC_1DVA) -> float:
    """Leave-one-observer-out NSS, averaged over every choice of held-out observer."""
    n = record.n_observers
    if n < 2:
        return float("nan")

    scores = []
    for i in range(n):
        others = [record.fixations[j] for j in range(n) if j != i]
        pooled_others = np.concatenate(others, axis=0) if others else np.zeros((0, 2))
        density = render_density(pooled_others, grid, sigma_frac)
        scores.append(nss_at_fixations(density, record.fixations[i], grid))
    return float(np.mean(scores))


def permutation_null_js(record, n_permutations: int = N_PERMUTATIONS,
                        grid: int = GRID, sigma_frac: float = SIGMA_FRAC_1DVA,
                        rng: np.random.Generator = None) -> float:
    """
    Mean JS over random redeals of the pooled fixations into observers of the
    original sizes. Preserves the aggregate map exactly; destroys only the
    observer grouping.
    """
    rng = rng or np.random.default_rng(0)
    pooled = record.pooled()
    sizes = [len(f) for f in record.fixations]
    total = int(sum(sizes))
    if total == 0 or record.n_observers < 2:
        return float("nan")

    split_points = np.cumsum(sizes)[:-1]
    values = []
    for _ in range(n_permutations):
        shuffled = pooled[rng.permutation(total)]
        groups = np.split(shuffled, split_points)
        densities = np.stack([render_density(g, grid, sigma_frac) for g in groups], axis=0)
        values.append(jensen_shannon(densities))
    return float(np.mean(values))


def split_half_targets(record, n_permutations: int, grid: int, sigma_frac: float,
                       rng: np.random.Generator):
    """
    js_excess computed on two disjoint halves of the observers.

    Used only to estimate the noise ceiling: correlate half_a against half_b
    across the dataset, Spearman-Brown correct, and that is the best any model
    could possibly score against the full-N target. Returns (nan, nan) when
    there are too few observers to split meaningfully.
    """
    n = record.n_observers
    if n < 4:
        return float("nan"), float("nan")

    order = rng.permutation(n)
    halves = [order[: n // 2], order[n // 2:]]
    out = []
    for half in halves:
        sub = type(record)(
            image_id=record.image_id, width=record.width, height=record.height,
            fixations=[record.fixations[i] for i in half],
        )
        densities = render_observer_densities(sub, grid, sigma_frac)
        observed = jensen_shannon(densities)
        null = permutation_null_js(sub, max(n_permutations // 4, 25), grid, sigma_frac, rng)
        out.append(observed - null)
    return out[0], out[1]


def compute_image_targets(record, n_permutations: int = N_PERMUTATIONS,
                          grid: int = GRID, sigma_frac: float = SIGMA_FRAC_1DVA,
                          rng: np.random.Generator = None,
                          with_split_half: bool = True) -> dict:
    """All targets for one image. Returns a flat dict of floats."""
    rng = rng or np.random.default_rng(0)

    densities = render_observer_densities(record, grid, sigma_frac)
    js_observed = jensen_shannon(densities)
    js_null = permutation_null_js(record, n_permutations, grid, sigma_frac, rng)
    aggregate = densities.mean(axis=0)

    half_a, half_b = (split_half_targets(record, n_permutations, grid, sigma_frac, rng)
                      if with_split_half else (float("nan"), float("nan")))

    return {
        "image_id": record.image_id,
        "n_observers": record.n_observers,
        "n_fixations": record.n_fixations,
        "ioc_nss": inter_observer_congruency(record, grid, sigma_frac),
        "js_observed": js_observed,
        "js_null": js_null,
        "js_excess": js_observed - js_null,
        "aggregate_entropy": entropy(aggregate),
        "js_excess_half_a": half_a,
        "js_excess_half_b": half_b,
    }


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

def compute_dataset(records: list, n_observers: int = None, seed: int = 42,
                    n_permutations: int = N_PERMUTATIONS, grid: int = GRID,
                    sigma_frac: float = SIGMA_FRAC_1DVA,
                    with_split_half: bool = True, verbose: bool = True) -> dict:
    """
    -> dict of arrays, one row per image, ready for np.savez.

    `n_observers` subsamples every image down to a common N. Leave it None only
    if the count is already confirmed constant across the dataset: JS grows
    with N, so a varying count injects a confound that looks exactly like
    signal (images seen by more people appear to have more disagreement).
    """
    rng = np.random.default_rng(seed)
    rows = []
    started = time.time()

    for i, record in enumerate(records):
        if n_observers is not None:
            record = record.subsample_observers(n_observers, rng)
        if record.n_observers < 2:
            continue
        rows.append(compute_image_targets(record, n_permutations, grid, sigma_frac,
                                          rng, with_split_half))
        if verbose and (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - started)
            print(f"  {i + 1}/{len(records)} images  ({rate:.1f} img/s)")

    if not rows:
        raise RuntimeError("No image produced targets — check the observer loader.")

    out = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    out["grid"] = np.array(grid)
    out["sigma_frac"] = np.array(sigma_frac)
    out["n_permutations"] = np.array(n_permutations)
    return out


def load_targets(path: str) -> dict:
    """Read a cache written by this module. Keys as in compute_dataset."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def report(targets: dict) -> dict:
    """
    Print the observer-agreement decision gate and return its numbers.

    Two things decide whether the target is usable, and both are visible here
    before any model exists:

      1. Is js_excess meaningfully above zero? If observed and null coincide,
         15 observers at ~10 fixations each cannot resolve real heterogeneity
         from sampling noise, and no model can predict what isn't measured.
      2. Is the split-half ceiling high enough to leave room? A ceiling of 0.3
         means even a perfect model scores 0.3.
    """
    js_excess = targets["js_excess"]
    js_observed = targets["js_observed"]
    js_null = targets["js_null"]
    finite = np.isfinite(js_excess)

    ratio = float(np.mean(js_excess[finite] / np.maximum(js_observed[finite], EPS)))
    frac_positive = float(np.mean(js_excess[finite] > 0))

    ceiling = float("nan")
    half_a, half_b = targets.get("js_excess_half_a"), targets.get("js_excess_half_b")
    if half_a is not None and half_b is not None:
        ok = np.isfinite(half_a) & np.isfinite(half_b)
        if ok.sum() > 10:
            from scipy.stats import spearmanr
            r_half = float(spearmanr(half_a[ok], half_b[ok]).statistic)
            ceiling = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")

    print("=" * 66)
    print("AGREEMENT TARGET REPORT")
    print("=" * 66)
    print(f"images                      : {finite.sum()}")
    print(f"js_observed (bits)          : {js_observed[finite].mean():.4f} "
          f"+/- {js_observed[finite].std():.4f}")
    print(f"js_null     (bits)          : {js_null[finite].mean():.4f} "
          f"+/- {js_null[finite].std():.4f}")
    print(f"js_excess   (bits)          : {js_excess[finite].mean():.4f} "
          f"+/- {js_excess[finite].std():.4f}")
    print(f"  excess as frac of observed: {ratio:.3f}")
    print(f"  images with excess > 0    : {frac_positive:.1%}")
    print(f"ioc_nss                     : {np.nanmean(targets['ioc_nss']):.4f} "
          f"+/- {np.nanstd(targets['ioc_nss']):.4f}")
    print(f"split-half ceiling (Spearman-Brown): {ceiling:.4f}")
    print()
    print("GATE 1 — is there signal beyond sampling noise?")
    print(f"  {'PASS' if ratio > 0.15 and frac_positive > 0.8 else 'FAIL'} "
          f"(want excess/observed > 0.15 and >80% of images positive)")
    print("GATE 2 — is the target measurable enough to predict?")
    print(f"  {'PASS' if ceiling > 0.4 else 'FAIL'} (want ceiling > 0.4)")
    print("=" * 66)

    return {"excess_ratio": ratio, "frac_positive": frac_positive, "ceiling": ceiling}


def main():
    parser = argparse.ArgumentParser(description="Compute and cache observer-agreement targets.")
    parser.add_argument("--dataset", choices=["mit1003", "salicon"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--n-observers", type=int, default=None,
                        help="subsample every image to this many observers")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading observers from {args.root} ...")
    records = load_observers(args.dataset, args.root, limit=args.limit)
    print(f"Loaded {len(records)} images.")

    targets = compute_dataset(
        records, n_observers=args.n_observers, seed=args.seed,
        n_permutations=args.n_permutations, grid=args.grid,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **targets)
    print(f"Wrote {out_path}\n")
    report(targets)


if __name__ == "__main__":
    main()

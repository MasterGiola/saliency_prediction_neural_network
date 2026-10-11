"""
Evaluation metrics for saliency prediction, matching the suite used in the
Wang & Shen (DVA) reference paper's Table II: CC, NSS, SIM, KLDiv, AUC-Judd.

All functions take numpy arrays for a SINGLE image (not batched), since NSS
and AUC-Judd need the discrete fixation locations, not just the continuous
map, and batching that cleanly is more trouble than it's worth. Call these
in a loop over the batch inside evaluate.py.
"""
import numpy as np

EPS = 1e-8


def _normalize01(x):
    x = x - x.min()
    return x / (x.max() + EPS)


def _normalize_to_distribution(x):
    x = x.astype(np.float64)
    return x / (x.sum() + EPS)


def cc(pred, target):
    pred = pred.flatten().astype(np.float64)
    target = target.flatten().astype(np.float64)
    pred = pred - pred.mean()
    target = target - target.mean()
    num = (pred * target).sum()
    den = np.sqrt((pred ** 2).sum() * (target ** 2).sum()) + EPS
    return num / den


def sim(pred, target):
    p = _normalize_to_distribution(_normalize01(pred))
    g = _normalize_to_distribution(_normalize01(target))
    return np.minimum(p, g).sum()


def kldiv(pred, target):
    p = _normalize_to_distribution(_normalize01(pred))
    g = _normalize_to_distribution(_normalize01(target))
    return (g * np.log(g / (p + EPS) + EPS)).sum()


def nss(pred, fixation_map):
    """
    fixation_map: binary map, 1 at fixation/high-attention locations.
    If you only have a continuous ground-truth map (as in SALICON), threshold
    it first, e.g. fixation_map = (target > target.mean() + target.std()).

    Returns NaN when the fixation map is empty, so the sample is excluded
    from the batch average rather than scored. Returning 0.0 there would be
    wrong in a way that's easy to miss: 0.0 is a legitimate NSS value meaning
    "predictions at fixations are exactly average", so an undefined sample
    would become indistinguishable from a genuinely mediocre one and drag
    the reported mean toward zero. Empty masks are not rare — the
    mean+1*sigma threshold in evaluate_batch produces them on near-uniform maps.
    """
    if fixation_map.sum() == 0:
        return np.nan
    pred = (pred - pred.mean()) / (pred.std() + EPS)
    return float(pred[fixation_map > 0].mean())


def auc_judd(pred, fixation_map, n_thresholds=100):
    """Simplified AUC-Judd: non-fixation points drawn uniformly, not shuffled (sAUC)."""
    pred = _normalize01(pred)
    pos = pred[fixation_map > 0]
    neg = pred[fixation_map == 0]
    # NaN (excluded from the average), not 0.5: 0.5 is the meaningful "chance
    # performance" value, so using it for undefined samples would silently pull
    # the reported AUC toward chance. See the nss() docstring.
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    thresholds = np.linspace(0, 1, n_thresholds)
    tpr = np.array([(pos >= t).mean() for t in thresholds])
    fpr = np.array([(neg >= t).mean() for t in thresholds])
    # integrate ROC curve; np.trapz was renamed np.trapezoid in numpy>=2.0
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return np.abs(trapezoid(tpr, fpr))


def evaluate_batch(preds, targets, fixation_threshold_std=1.0, return_counts=False, masks=None):
    """
    preds, targets: numpy arrays [B, H, W] (already squeezed from [B,1,H,W]).
    Returns a dict of metric -> mean value over the batch.

    NSS and AUC-Judd return NaN for samples with an empty derived fixation map
    (see nss()), so aggregation uses nanmean: undefined samples are excluded
    from the average instead of being scored as 0.0 / 0.5. A metric that is NaN
    for every sample in the batch stays NaN rather than raising, which surfaces
    the problem instead of hiding it.

    return_counts=True additionally returns a dict of metric -> number of valid
    (non-NaN) samples, so the caller can weight batch means correctly and report
    how many samples each metric was actually computed over.

    masks: optional [B, H, W] array, 1.0 = real content, 0.0 = letterbox padding
    (see SaliencyDataset's valid_mask). When given, each sample's pred/target are
    restricted to the valid pixels before any metric is computed — fixation-map
    thresholding, normalization, and z-scoring inside cc/sim/kldiv/nss/auc_judd
    all then operate purely on real content, so a flat gray input region and its
    identically-zero target never contribute their trivially-easy agreement to
    the score. None of those functions assume a 2D shape — they reduce over the
    whole array or index with a boolean mask — so passing them the flattened
    valid-pixel vector works unchanged; masking happens here, once, rather than
    inside each metric.
    """
    results = {"cc": [], "sim": [], "kldiv": [], "nss": [], "auc_judd": []}
    for i, (p, g) in enumerate(zip(preds, targets)):
        if masks is not None:
            valid = masks[i] > 0
            p, g = p[valid], g[valid]
        fixation_map = (g > g.mean() + fixation_threshold_std * g.std()).astype(np.float32)
        results["cc"].append(cc(p, g))
        results["sim"].append(sim(p, g))
        results["kldiv"].append(kldiv(p, g))
        results["nss"].append(nss(p, fixation_map))
        results["auc_judd"].append(auc_judd(p, fixation_map))

    means, counts = {}, {}
    for k, v in results.items():
        arr = np.asarray(v, dtype=np.float64)
        valid = ~np.isnan(arr)
        counts[k] = int(valid.sum())
        means[k] = float(arr[valid].mean()) if valid.any() else float("nan")

    return (means, counts) if return_counts else means
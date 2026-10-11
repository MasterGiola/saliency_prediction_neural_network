"""
Stage-2 training: the agreement head, cross-validated on MIT1003.

Follows the same shape as train.py — get_optimizer / train_one_epoch /
train loop with early stopping — but the reported quantity is a
cross-validated rank correlation rather than a validation loss, since that
is the claim this stage actually needs to support.

Three numbers come out, and reporting fewer than three makes the result
uninterpretable:

  1. Cross-validated Spearman between predicted and true js_excess.
  2. The split-half noise ceiling from agreement_targets, since a raw 0.55
     against a ceiling of 0.65 is a strong result while the same 0.55
     reported bare looks mediocre.
  3. A three-feature baseline (edge density, colour dispersion, face area).
     Faces produce consensus, clutter produces disagreement, so if the model
     doesn't clearly beat a linear regression on those three numbers it has
     learned nothing beyond counting things.

Run:
    python -m src.train_agreement --config config.yaml --all-folds
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from src.agreement_data import get_agreement_loaders
from src.agreement_losses import AgreementLoss
from src.agreement_model import build_two_head_model
from src.agreement_targets import load_targets
from src.utils import get_device, load_config, set_seed


def get_agreement_optimizer(model, config: dict):
    """
    Optimizes only parameters with requires_grad — which, after freeze_base(),
    is just the agreement head.

    Deliberately not reusing train.get_optimizer: that function splits encoder
    and decoder into two learning-rate groups, which is meaningless here
    because the encoder has no trainable parameters at this stage. Passing a
    frozen parameter group to Adam is harmless but misleading in the logs.
    """
    agr_cfg = config["agreement"]
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters — did you freeze everything?")
    return torch.optim.Adam(
        params,
        lr=agr_cfg.get("lr", 1e-3),
        weight_decay=agr_cfg.get("weight_decay", 1e-4),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for images, _, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out["agreement"], out["agreement_log_var"], targets)
        loss.backward()
        optimizer.step()

        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_fold(model, loader, criterion, device):
    """-> dict with loss, spearman, pearson and the raw prediction/target arrays."""
    model.eval()
    total, n = 0.0, 0
    preds, trues = [], []

    for images, _, targets in tqdm(loader, desc="eval", leave=False):
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        loss = criterion(out["agreement"], out["agreement_log_var"], targets)

        total += loss.item() * images.size(0)
        n += images.size(0)
        preds.append(out["agreement"].detach().cpu().numpy())
        trues.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    return {
        "loss": total / max(n, 1),
        "spearman": float(spearmanr(preds, trues).statistic),
        "pearson": float(np.corrcoef(preds, trues)[0, 1]),
        "preds": preds,
        "targets": trues,
    }


def train_fold(config: dict, fold: int, device, verbose: bool = True) -> dict:
    agr_cfg = config["agreement"]
    train_loader, val_loader, _ = get_agreement_loaders(config, fold=fold)

    model = build_two_head_model(
        config, base_checkpoint=agr_cfg.get("base_checkpoint"), device=device,
    )
    criterion = AgreementLoss(
        loss_type=agr_cfg.get("loss", "beta_nll"), beta=agr_cfg.get("beta", 0.5),
    )
    optimizer = get_agreement_optimizer(model, config)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n--- fold {fold} --- train {len(train_loader.dataset)} / "
              f"val {len(val_loader.dataset)} images, {n_trainable} trainable params")

    best = {"spearman": -1.0, "epoch": 0}
    best_state = None
    patience = agr_cfg.get("early_stopping_patience", 10)
    stale = 0

    for epoch in range(1, agr_cfg.get("epochs", 40) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate_fold(model, val_loader, criterion, device)

        # Model selection on rank correlation, not on loss. The NLL is dominated
        # by how well the variance head is calibrated, which is not the quantity
        # being reported — selecting on it would optimize the wrong thing.
        if metrics["spearman"] > best["spearman"]:
            best = {"spearman": metrics["spearman"], "pearson": metrics["pearson"],
                    "epoch": epoch, "preds": metrics["preds"], "targets": metrics["targets"]}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.agreement_head.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:>3}  train_loss {train_loss:.4f}  "
                  f"val_loss {metrics['loss']:.4f}  spearman {metrics['spearman']:.4f}")

        if patience and stale >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best {best['spearman']:.4f} "
                      f"@ epoch {best['epoch']})")
            break

    if best_state is not None:
        ckpt_dir = Path(agr_cfg.get("checkpoint_dir", "checkpoints/agreement"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / f"agreement_head_fold{fold}.pt")

    return best


# ---------------------------------------------------------------------------
# Baselines and ceiling — the context that makes the headline number readable
# ---------------------------------------------------------------------------

def image_baseline_features(image_path: str) -> np.ndarray:
    """
    Three cheap image statistics, computed with no learned model.

      edge_density  : mean Sobel magnitude. Proxy for clutter.
      colour_spread : mean per-channel standard deviation. Proxy for scene variety.
      face_fraction : fraction of image area covered by detected faces, via
                      OpenCV's Haar cascade. Falls back to 0.0 if cv2 is absent,
                      which weakens the baseline — install opencv-python so the
                      comparison is actually adversarial.
    """
    from PIL import Image
    from scipy import ndimage

    img = Image.open(image_path).convert("RGB").resize((256, 256))
    arr = np.asarray(img, dtype=np.float64) / 255.0
    gray = arr.mean(axis=2)

    gx = ndimage.sobel(gray, axis=0)
    gy = ndimage.sobel(gray, axis=1)
    edge_density = float(np.hypot(gx, gy).mean())
    colour_spread = float(arr.std(axis=(0, 1)).mean())

    face_fraction = 0.0
    try:
        import cv2
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray_u8 = (gray * 255).astype(np.uint8)
        faces = cascade.detectMultiScale(gray_u8, scaleFactor=1.1, minNeighbors=5)
        if len(faces):
            face_fraction = float(sum(w * h for _, _, w, h in faces) / (256 * 256))
    except Exception:
        pass

    return np.array([edge_density, colour_spread, face_fraction])


def run_baseline(config: dict, n_folds: int = 5, seed: int = 42) -> float:
    """
    Cross-validated Spearman of a linear regression on the three features above.
    Uses the same folds as the model so the comparison is like for like.
    """
    from sklearn.linear_model import Ridge
    from src.agreement_data import AgreementDataset, make_cv_folds

    agr_cfg = config["agreement"]
    dataset = AgreementDataset(
        image_dir=agr_cfg["image_dir"], map_dir=agr_cfg["map_dir"],
        targets_path=agr_cfg["targets_path"], img_size=config["data"]["img_size"],
        target_key=agr_cfg.get("target_key", "js_excess"),
    )

    print("Computing baseline image features ...")
    features = np.stack([image_baseline_features(p) for p, _, _ in
                         tqdm(dataset.samples, leave=False)])
    targets = dataset.raw_targets()

    preds = np.zeros_like(targets)
    for train_idx, val_idx in make_cv_folds(len(dataset), n_folds,
                                            agr_cfg.get("fold_seed", seed)):
        model = Ridge(alpha=1.0).fit(features[train_idx], targets[train_idx])
        preds[val_idx] = model.predict(features[val_idx])

    return float(spearmanr(preds, targets).statistic)


def noise_ceiling(targets_path: str) -> float:
    """Spearman-Brown-corrected split-half reliability of the target itself."""
    cache = load_targets(targets_path)
    half_a, half_b = cache.get("js_excess_half_a"), cache.get("js_excess_half_b")
    if half_a is None or half_b is None:
        return float("nan")
    ok = np.isfinite(half_a) & np.isfinite(half_b)
    if ok.sum() < 10:
        return float("nan")
    r = float(spearmanr(half_a[ok], half_b[ok]).statistic)
    return 2 * r / (1 + r) if r > -1 else float("nan")


def run_cross_validation(config: dict, device=None, folds=None,
                         skip_baseline: bool = False, out_path: str = None) -> dict:
    """
    Full cross-validated run: every fold, plus the ceiling and the baseline.

    This is the notebook/`experiments.py` entry point — main() is a thin CLI
    wrapper around it. Returns the same dict that gets written to JSON, so
    phase 4 can compare tiers without re-parsing anything off disk.
    """
    device = device or get_device()
    agr_cfg = config["agreement"]
    n_folds = agr_cfg.get("n_folds", 5)
    folds = range(n_folds) if folds is None else folds

    per_fold, all_preds, all_targets = [], [], []
    for fold in folds:
        best = train_fold(config, fold, device)
        per_fold.append(best["spearman"])
        all_preds.append(best["preds"])
        all_targets.append(best["targets"])

    pooled = float(spearmanr(np.concatenate(all_preds),
                             np.concatenate(all_targets)).statistic)
    ceiling = noise_ceiling(agr_cfg["targets_path"])
    baseline = float("nan") if skip_baseline else run_baseline(config, n_folds)

    print("\n" + "=" * 62)
    print("AGREEMENT HEAD — CROSS-VALIDATED RESULTS")
    print("=" * 62)
    print(f"per-fold Spearman     : {[round(s, 4) for s in per_fold]}")
    print(f"mean +/- sd           : {np.mean(per_fold):.4f} +/- {np.std(per_fold):.4f}")
    print(f"pooled Spearman       : {pooled:.4f}")
    print(f"3-feature baseline    : {baseline:.4f}")
    print(f"split-half ceiling    : {ceiling:.4f}")
    if np.isfinite(ceiling) and ceiling > 0:
        print(f"fraction of ceiling   : {pooled / ceiling:.1%}")
    print("=" * 62)
    if np.isfinite(baseline):
        print("VERDICT:", "model beats the trivial baseline" if pooled - baseline > 0.05
              else "NOT clearly better than three image statistics — report as such")

    results = {"per_fold_spearman": per_fold, "pooled_spearman": pooled,
               "baseline_spearman": baseline, "noise_ceiling": ceiling}

    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train the observer-agreement head.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--out", default="results/agreement_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    folds = None if args.all_folds else [args.fold]
    run_cross_validation(config, device=device, folds=folds,
                         skip_baseline=args.skip_baseline, out_path=args.out)


if __name__ == "__main__":
    main()

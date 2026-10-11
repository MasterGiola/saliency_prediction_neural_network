"""
Evaluation loop.
"""
import torch
from tqdm import tqdm

from src.losses import deep_supervision_loss
from src.metrics import evaluate_batch
from src.model import unwrap_prediction


@torch.no_grad()
def evaluate(model, loader, criterion, device, aux_weight=1.0):
    """
    Returns a dict with the CC/SIM/KLDiv/NSS/AUC-Judd metrics plus two losses:

      "loss"        - the full deep-supervision objective (fused + averaged aux),
                      i.e. the exact quantity train_one_epoch minimizes. This is
                      what the LR scheduler and early stopping monitor, so that
                      they track the objective actually being optimized.

      "fused_loss"  - the fused output scored alone. The auxiliary heads are
                      discarded at inference, so this is the loss on what the
                      model actually predicts at test time. Used to select the
                      "_best.pt" checkpoint, and it's the only one of the two
                      that's comparable across tiers — base_resnet has no aux
                      heads, so its "loss" is on a different scale from the
                      fusion models' by construction.

    For base_resnet (aux is None) the two are identical.

    Padding mask: if `loader`'s dataset was built with return_mask=True (val_set
    in get_dataloaders, and get_mit1003_dataset — both eval-only), each batch is
    a 3-tuple (images, targets, masks) instead of 2, and the mask is passed to
    evaluate_batch so CC/SIM/KLDiv/NSS/AUC-Judd are computed on real content only,
    excluding letterbox padding (see SaliencyDataset's docstring). Handled
    defensively per-batch (2-tuple falls back to masks=None) so evaluate() works
    unchanged on a loader that doesn't carry masks — e.g. train_loader, which
    intentionally never does, since masking is an evaluation-only concern.
    """
    model.eval()
    total_loss = 0.0
    total_fused_loss = 0.0
    n = 0
    metric_sums = {"cc": 0.0, "sim": 0.0, "kldiv": 0.0, "nss": 0.0, "auc_judd": 0.0}
    metric_counts = {k: 0 for k in metric_sums}

    for batch in tqdm(loader, desc="eval", leave=False):
        if len(batch) == 3:
            images, targets, masks = batch
            masks_np = masks.squeeze(1).numpy()  # [B, H, W], stays on CPU — numpy-only from here
        else:
            images, targets = batch
            masks_np = None
        images, targets = images.to(device), targets.to(device)

        pred = model(images)
        fused, aux = unwrap_prediction(pred)

        loss = deep_supervision_loss(criterion, fused, aux, targets, aux_weight=aux_weight)
        fused_loss = criterion(fused, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_fused_loss += fused_loss.item() * bs

        preds_np = fused.squeeze(1).cpu().numpy()
        targets_np = targets.squeeze(1).cpu().numpy()
        batch_metrics, batch_counts = evaluate_batch(preds_np, targets_np, masks=masks_np,
                                                       return_counts=True)

        # Accumulate per-metric rather than weighting everything by batch size:
        # NSS and AUC-Judd skip samples whose derived fixation map is empty, so
        # their valid count can be < bs. Dividing those by the full sample count
        # would understate them in proportion to how many samples were skipped.
        for k, v in batch_metrics.items():
            c = batch_counts[k]
            if c:
                metric_sums[k] += v * c
                metric_counts[k] += c
        n += bs

    results = {k: (metric_sums[k] / metric_counts[k] if metric_counts[k] else float("nan"))
               for k in metric_sums}
    results["loss"] = total_loss / n
    results["fused_loss"] = total_fused_loss / n

    # Surface any metric that couldn't be computed on every sample, so a
    # partially-undefined metric doesn't get reported as if it were complete.
    for k, c in metric_counts.items():
        if c < n:
            print(f"  NOTE: {k} computed on {c}/{n} samples "
                  f"({n - c} skipped: empty derived fixation map)")

    return results


def evaluate_mit1003(model, config: dict, device, loss_type: str = None):
    """
    Convenience wrapper: builds the MIT1003 loader from config and runs the
    same evaluate() used for SALICON validation, so you get directly
    comparable metrics on a dataset the model never trained on. Returns None
    (with a printed warning) if config["data"]["mit1003_root"] isn't set.

    loss_type: defaults to whatever's in config["train"]["loss"] — override
    if you want to score with a different loss than the one used for
    training (the metrics dict itself, CC/SIM/KLDiv/NSS/AUC-Judd, doesn't
    depend on this; only the reported "loss" value does).
    """
    from src.data import get_mit1003_loader
    from src.losses import SaliencyLoss

    loader = get_mit1003_loader(config)
    if loader is None:
        print("config['data']['mit1003_root'] is not set — skipping MIT1003 evaluation.")
        return None

    train_cfg = config.get("train", {})
    criterion = SaliencyLoss(
        loss_type=loss_type or train_cfg.get("loss", "kldiv_cc"),
        kldiv_weight=train_cfg.get("kldiv_weight", 1.0),
        cc_weight=train_cfg.get("cc_weight", 0.5),
    )
    return evaluate(model, loader, criterion, device, aux_weight=train_cfg.get("aux_weight", 1.0))

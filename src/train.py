"""
Training loop.
"""
import math

import torch
from tqdm import tqdm

from src.losses import SaliencyLoss, deep_supervision_loss
from src.model import unwrap_prediction
from src.utils import save_checkpoint, save_checkpoint_named, load_checkpoint, build_run_name, get_device


def get_optimizer(model, config: dict):
    """
    Builds the optimizer with two parameter groups when the model has a
    pretrained `.encoder` (both BaseSaliencyNet and FusionSaliencyNet do):
    the encoder gets `lr * encoder_lr_mult`, everything else (decoder,
    fusion/final head, aux heads) gets the full `lr`.

    Fine-tuning a pretrained backbone at the same LR as a randomly
    initialized decoder risks distorting features that were already good —
    giving it a smaller LR (encoder_lr_mult < 1, e.g. 0.1) is the standard
    fix. Set encoder_lr_mult=1.0 (or omit it) to fall back to a single LR
    for every parameter, matching the old behavior.
    """
    train_cfg = config["train"]
    base_lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.0)
    encoder_lr_mult = train_cfg.get("encoder_lr_mult", 1.0)

    if hasattr(model, "encoder") and encoder_lr_mult != 1.0:
        encoder_param_ids = {id(p) for p in model.encoder.parameters()}
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]
        params = [
            {"params": encoder_params, "lr": base_lr * encoder_lr_mult},
            {"params": other_params, "lr": base_lr},
        ]
    else:
        # filtered for consistency with the two-group branch above: without this,
        # a frozen encoder (requires_grad=False) would still have its params handed
        # to Adam, which silently allocates momentum/variance state for tensors that
        # never receive a gradient. Harmless today (nothing in this codebase freezes
        # the encoder), but a real bug the moment a freeze option is added — and the
        # two branches should filter identically regardless.
        params = [p for p in model.parameters() if p.requires_grad]

    if train_cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")


def get_scheduler(optimizer, config: dict):
    """
    Builds an LR scheduler from config["train"]["scheduler"], stepped once
    per epoch (see the bottom of the training loop in `train()`).

    config["train"]["scheduler"]:
      type: "none" (default) | "cosine" | "plateau"

      "cosine":
        warmup_epochs: linearly ramps each param group's LR from 0 up to its
          base LR over this many epochs (default 0, i.e. no warmup).
        min_lr_ratio: floor the decay at this fraction of each group's base
          LR rather than 0 (default 0.01) — expressed as a ratio (not an
          absolute LR) so it applies correctly to both the encoder's and the
          decoder's param group even though they have different base LRs.

      "plateau": wraps ReduceLROnPlateau, watching val_loss (see train()).
        plateau_factor, plateau_patience, min_lr as usual.

    Returns None if type is "none" (or unset), and train() just skips
    stepping it — this keeps a from-scratch config with no "scheduler" key
    behaving exactly like before (flat LR for the whole run).
    """
    train_cfg = config["train"]
    sched_cfg = train_cfg.get("scheduler") or {}
    sched_type = sched_cfg.get("type", "none")

    if sched_type in (None, "none"):
        return None

    if sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=sched_cfg.get("plateau_factor", 0.5),
            patience=sched_cfg.get("plateau_patience", 3),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )

    if sched_type == "cosine":
        total_epochs = train_cfg["epochs"]
        warmup_epochs = sched_cfg.get("warmup_epochs", 0)
        min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.01)

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            span = max(total_epochs - warmup_epochs, 1)
            progress = min(max((epoch - warmup_epochs) / span, 0.0), 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown scheduler type: {sched_type}")


def _is_plateau_scheduler(scheduler) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def train_one_epoch(model, loader, optimizer, criterion, device, aux_weight=1.0):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        pred = model(images)
        fused, aux = unwrap_prediction(pred)

        loss = deep_supervision_loss(criterion, fused, aux, targets, aux_weight=aux_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def train(model, train_loader, val_loader, config: dict):
    """
    Returns (model, history). history is a dict of per-epoch lists:
    train_loss, val_loss, fused_val_loss, cc, sim, kldiv, nss, auc_judd —
    feed this straight into src.visualize.plot_training_curves().

    Two validation losses are tracked (see evaluate() for the full rationale):
    val_loss is the deep-supervision objective being minimized (fused +
    averaged aux, per DVA Eq. 9) and drives the LR scheduler and early
    stopping; fused_val_loss scores the fused output alone and selects
    "<run_name>_best.pt", since the aux heads are discarded at inference.
    For base_resnet the two are identical.

    Checkpoint naming: files are named "<run_name>_epoch<N>.pt", where
    run_name is built from model name + backbone + loss (see
    src.utils.build_run_name), so different configs sharing a
    checkpoint_dir don't collide or get confused during resume.

    Resume: set config["train"]["resume_from"] to a checkpoint path (or use
    src.utils.find_latest_checkpoint(checkpoint_dir, name=run_name) to find
    it automatically). Resuming restores model/optimizer/scheduler state, the
    loss/metric history (so plots continue seamlessly), AND early-stopping
    state (best_val_loss, epochs_without_improvement) — so patience and the
    LR schedule both continue correctly across a Colab disconnect instead of
    resetting. Checkpoints saved before the scheduler existed just won't have
    scheduler state to restore — the scheduler resumes fresh from the
    restored epoch instead of erroring.

    Early stopping: set config["train"]["early_stopping_patience"] (0/null
    disables). Monitors val_loss. The checkpoint at the epoch that triggers
    early stopping is always saved even if it doesn't land on the regular
    checkpoint_every cadence.
    """
    from src.evaluate import evaluate  # local import avoids circular import

    device = get_device()
    model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    train_cfg = config["train"]
    run_name = build_run_name(config)

    criterion = SaliencyLoss(
        loss_type=train_cfg.get("loss", "kldiv_cc"),
        kldiv_weight=train_cfg.get("kldiv_weight", 1.0),
        cc_weight=train_cfg.get("cc_weight", 0.5),
    )
    aux_weight = train_cfg.get("aux_weight", 1.0)
    patience = train_cfg.get("early_stopping_patience", 0)
    min_delta = train_cfg.get("early_stopping_min_delta", 1e-4)

    history = {"train_loss": [], "val_loss": [], "fused_val_loss": [], "cc": [], "sim": [],
               "kldiv": [], "nss": [], "auc_judd": []}
    start_epoch = 1
    best_val_loss = float("inf")
    best_fused_loss = float("inf")
    epochs_without_improvement = 0

    # Resume from a checkpoint if one is configured. Order matters here: model is already on
    # `device` and optimizer already built from its (device-resident) parameters, so passing
    # device as load_checkpoint's map_location lands the restored optimizer state on the same
    # device too — load it before model.to(device)/optimizer creation and you'd get a
    # CPU/GPU mismatch on the first optimizer.step().
    resume_path = train_cfg.get("resume_from")
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        last_epoch, loaded_history, extra_state = load_checkpoint(
            resume_path, model, optimizer, scheduler=scheduler, device=device
        )
        start_epoch = last_epoch + 1
        if loaded_history is not None:
            history = loaded_history
        best_val_loss = extra_state.get("best_val_loss", float("inf"))
        # checkpoints written before fused-loss selection existed won't carry this;
        # falling back to best_val_loss is the right approximation (they were equal
        # for base_resnet, and an overestimate for fusion models, which just means
        # the first post-resume epoch is slightly more likely to be saved as "best")
        best_fused_loss = extra_state.get("best_fused_loss", best_val_loss)
        epochs_without_improvement = extra_state.get("epochs_without_improvement", 0)
        print(f"Resuming training at epoch {start_epoch} (checkpoint was saved after epoch {last_epoch}, "
              f"best_val_loss so far: {best_val_loss:.4f}, "
              f"epochs_without_improvement: {epochs_without_improvement})")

    if start_epoch > train_cfg["epochs"]:
        print(f"Checkpoint is already at epoch {start_epoch - 1}, which is >= train.epochs "
              f"({train_cfg['epochs']}) in the config. Nothing to train — raise train.epochs "
              f"if you want to keep going.")
        return model, history

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        print(f"Training epoch {epoch}/{train_cfg['epochs']}...")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, aux_weight)
        print("Evaluating on validation set...")
        val_metrics = evaluate(model, val_loader, criterion, device, aux_weight=aux_weight)

        history["train_loss"].append(train_loss)
        for key in ["val_loss", "fused_val_loss", "cc", "sim", "kldiv", "nss", "auc_judd"]:
            metric_key = {"val_loss": "loss", "fused_val_loss": "fused_loss"}.get(key, key)
            history[key].append(val_metrics[metric_key])

        print(f"Epoch {epoch}/{train_cfg['epochs']} - train_loss: {train_loss:.4f} "
              f"- val_loss: {val_metrics['loss']:.4f} "
              f"(fused: {val_metrics['fused_loss']:.4f}) "
              f"- CC: {val_metrics['cc']:.4f} - NSS: {val_metrics['nss']:.4f} "
              f"- KLDiv: {val_metrics['kldiv']:.4f} - AUC: {val_metrics['auc_judd']:.4f}")

        # Two different quantities, deliberately:
        #   val_loss   — the full deep-supervision objective, i.e. what training
        #                 actually minimizes. Early stopping and the LR scheduler
        #                 watch this, so they respond to the objective in use.
        #   fused_loss — the fused output alone. The aux heads are thrown away at
        #                 inference, so this is the loss on what the model really
        #                 predicts; "_best.pt" is selected on it. Picking the best
        #                 checkpoint by the joint objective would optimize for
        #                 auxiliary-head quality that never ships.
        # For base_resnet (no aux heads) the two are equal and this collapses to
        # the previous single-criterion behavior.
        val_loss = val_metrics["loss"]
        fused_loss = val_metrics["fused_loss"]

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        best_fused_improved = fused_loss < best_fused_loss - min_delta
        if best_fused_improved:
            best_fused_loss = fused_loss

        # Step the LR scheduler once per epoch, after validation (ReduceLROnPlateau needs
        # this epoch's val_loss to decide whether to decay; the cosine/warmup schedule
        # doesn't need it but stepping both the same way here keeps this branch simple).
        if scheduler is not None:
            if _is_plateau_scheduler(scheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lrs = [round(g["lr"], 8) for g in optimizer.param_groups]
            print(f"  Current LR(s): {current_lrs}")

        should_stop = bool(patience) and epochs_without_improvement >= patience
        extra_state = {"best_val_loss": best_val_loss,
                       "best_fused_loss": best_fused_loss,
                       "epochs_without_improvement": epochs_without_improvement}

        # Save on the configured cadence, always save the final epoch (otherwise a crash right
        # after the last epoch loses it entirely), and always save the epoch that triggers early
        # stopping even if it doesn't land on the cadence.
        if epoch % train_cfg["checkpoint_every"] == 0 or epoch == train_cfg["epochs"] or should_stop:
            path = save_checkpoint(model, optimizer, epoch, train_cfg["checkpoint_dir"],
                                    name=run_name, history=history, extra_state=extra_state,
                                    scheduler=scheduler)
            print(f"Saved checkpoint: {path}")

        if best_fused_improved:
            # convenience copy that's always the best epoch, so you don't have to hunt
            # through history to figure out which numbered checkpoint to use for evaluation.
            # Uses save_checkpoint_named (exact filename, overwritten in place) rather than
            # save_checkpoint (which would append _epoch{N} and accumulate one file per
            # improvement instead of staying a single "best" file).
            best_path = save_checkpoint_named(model, optimizer, epoch, train_cfg["checkpoint_dir"],
                                               f"{run_name}_best.pt", history=history, extra_state=extra_state,
                                               scheduler=scheduler)
            print(f"  New best fused val_loss: {best_fused_loss:.4f} — saved {best_path}")

        if should_stop:
            print(f"Early stopping: no val_loss improvement for {patience} epochs "
                  f"(best was {best_val_loss:.4f}). Stopping at epoch {epoch}.")
            break

    return model, history

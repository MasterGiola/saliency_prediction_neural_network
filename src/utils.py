"""Shared helpers: config loading, reproducibility, checkpointing."""
import random
import yaml
import numpy as np
import torch
import os
from pathlib import Path


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load the YAML config file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seeds everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_run_name(config: dict) -> str:
    """
    Builds a descriptive identifier from the run's config, e.g.
    "intermediate_fusion_resnet34_kldiv_cc" — used to name checkpoints so
    runs with different models/backbones/losses don't overwrite or get
    confused with each other (including when resuming: find_latest_checkpoint
    and pull_latest_checkpoint_from_hf both filter by this name).

    Ablation switches that change the parameter shape or the architecture are
    appended as suffixes. Without this, an A/B pair differing only in e.g.
    use_skip_connections shares one filename: the second run resumes from the
    first run's checkpoint and dies on a state_dict shape mismatch, or
    overwrites it and the baseline being compared against is lost. Suffixes
    are only added for the non-default setting, so an existing config that
    leaves them all on produces exactly the same name as before and old
    checkpoints stay discoverable.

    Deliberately not included: img_size, lr, batch_size, aux_weight, seed.
    Those don't change the state_dict shape, so sharing a name across them is
    only a bookkeeping question, and folding every hyperparameter in would
    produce unreadable filenames. Use separate checkpoint_dir values if you
    need to keep such runs apart.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    parts = [
        model_cfg.get("name", "model"),
        model_cfg.get("backbone", "backbone"),
        train_cfg.get("loss", "loss"),
    ]

    if not model_cfg.get("use_skip_connections", True):
        parts.append("noskip")
    if not model_cfg.get("use_center_bias", True):
        parts.append("nocb")
    if not model_cfg.get("use_output_smoothing", True):
        parts.append("nosmooth")
    if not config.get("data", {}).get("preserve_aspect_ratio", True):
        parts.append("squash")

    # out_indices changes both the number of decoder branches and the fusion
    # head's input channels, so it must not share a name with a different setting
    out_indices = model_cfg.get("out_indices")
    if out_indices is not None and model_cfg.get("name") != "base_resnet":
        default = (1, 2, 3) if model_cfg.get("name") == "advanced_transformer" else (2, 3, 4)
        if tuple(out_indices) != default:
            parts.append("oi" + "".join(str(i) for i in out_indices))

    return "_".join(str(p) for p in parts)


def save_checkpoint(model, optimizer, epoch: int, checkpoint_dir: str, name: str = "model",
                     history: dict = None, extra_state: dict = None, scheduler=None) -> str:
    """Save model + optimizer (+ scheduler) state (+ training history + early-stopping state)
    so far. Returns the path written to.

    `extra_state` holds small scalar training-loop state that needs to survive a resume
    alongside `history` — currently best_val_loss and epochs_without_improvement, so early
    stopping's patience counter picks up where it left off instead of resetting to 0 after
    a Colab disconnect.

    `scheduler` is optional (pass None if you're not using one, e.g. the old flat-LR setup);
    when given, its state_dict is saved too so resuming continues the LR schedule (warmup
    position, plateau's patience counter, etc.) instead of restarting it.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "extra_state": extra_state or {},
    }, path)
    return path


def save_checkpoint_named(model, optimizer, epoch: int, checkpoint_dir: str, filename: str,
                          history: dict = None, extra_state: dict = None, scheduler=None) -> str:
    """
    Like save_checkpoint, but writes to an EXACT filename (no auto epoch
    suffix) — so calling this repeatedly with the same filename OVERWRITES
    in place instead of accumulating one file per epoch. Used for the
    "<run_name>_best.pt" convenience copy in src.train.train, so you don't
    have to hunt through the per-epoch accumulating checkpoints (which exist
    for resume purposes) to find the best one for final evaluation.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "extra_state": extra_state or {},
        }, path)
    return path


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device=None):
    """Load model (and optionally optimizer/scheduler) state from a checkpoint file.

    Returns (epoch, history, extra_state):
      epoch       - the epoch this checkpoint was saved after (0 if missing/old-format file)
      history     - the training-curve dict saved alongside it, or None if the checkpoint
                    predates that field (in which case the caller should start a fresh history)
      extra_state - dict of small scalar state (e.g. best_val_loss, epochs_without_improvement),
                    or {} if the checkpoint predates it

    `device` should be the device you're about to train on (e.g. from get_device()). Passing
    it as the map_location means the optimizer's internal state (e.g. Adam's momentum buffers)
    lands on the same device as the model params it's about to update — load onto CPU and
    the very first optimizer.step() will error on a CPU/GPU tensor mismatch.

    `scheduler`, if given, is only restored when the checkpoint actually has scheduler state
    (checkpoints saved before the scheduler existed won't) — in that case the scheduler just
    starts fresh from the restored epoch instead of raising.
    """
    checkpoint = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("history"), checkpoint.get("extra_state", {})


def find_latest_checkpoint(checkpoint_dir: str, name: str = "model"):
    """Return the path to the highest-epoch checkpoint in checkpoint_dir, or None if there isn't one.

    Meant for the common Colab failure mode: the runtime disconnects mid-training and you don't
    want to go hunting through Drive for the exact filename you last saved — just point
    config["train"]["resume_from"] at find_latest_checkpoint(checkpoint_dir).
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = list(checkpoint_dir.glob(f"{name}_epoch*.pt"))
    if not candidates:
        return None

    def _epoch_num(p):
        return int(p.stem.split("epoch")[-1])

    return str(max(candidates, key=_epoch_num))

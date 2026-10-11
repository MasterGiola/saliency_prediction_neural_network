"""
Visualization utilities: training curves, final test metrics, and qualitative
comparisons of predicted vs. ground-truth saliency maps.

Meant to be called from the Colab launcher notebook after training finishes.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.model import unwrap_prediction


def plot_training_curves(history: dict, figsize=(14, 4)):
    """
    history: the dict returned by src.train.train(), with keys
    train_loss, val_loss, cc, sim, kldiv, nss, auc_judd (each a list, one
    value per epoch).

    Produces two panels: loss curves, and the CC/NSS/AUC metric curves
    (the ones where higher = better; KLDiv and SIM are left off this panel
    since they're on different scales — see plot_final_metrics for those).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(epochs, history["train_loss"], label="train loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="val loss (fused + aux)", marker="o", markersize=3)
    # fused-only curve: absent from checkpoints predating the two-loss split, and
    # identical to val_loss for base_resnet, so only plot it when it adds something
    fused = history.get("fused_val_loss")
    if fused and not np.allclose(fused, history["val_loss"]):
        axes[0].plot(epochs, fused, label="val loss (fused only)",
                     marker="o", markersize=3, linestyle="--")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training / validation loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key, label in [("cc", "CC"), ("nss", "NSS"), ("auc_judd", "AUC-Judd")]:
        axes[1].plot(epochs, history[key], label=label, marker="o", markersize=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric value (higher = better)")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def print_metrics_table(metrics: dict, title: str = "Test set results"):
    """Simple aligned text table — readable directly in Colab output."""
    print(f"\n{title}")
    print("-" * len(title))
    order = ["cc", "sim", "kldiv", "nss", "auc_judd", "loss"]
    labels = {"cc": "CC", "sim": "SIM", "kldiv": "KLDiv", "nss": "NSS",
              "auc_judd": "AUC-Judd", "loss": "Loss"}
    for key in order:
        if key in metrics:
            print(f"  {labels[key]:<10s}: {metrics[key]:.4f}")


def plot_final_metrics(metrics: dict, figsize=(6, 4)):
    """Bar chart of final test-set metrics, for a quick visual summary."""
    order = ["cc", "sim", "nss", "auc_judd"]  # kldiv excluded: lower-is-better, different scale
    labels = {"cc": "CC", "sim": "SIM", "nss": "NSS", "auc_judd": "AUC-Judd"}
    values = [metrics[k] for k in order if k in metrics]
    names = [labels[k] for k in order if k in metrics]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(values)])
    ax.set_ylabel("value")
    ax.set_title("Final test metrics (higher = better)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def _denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display. img_tensor: [3, H, W]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


@torch.no_grad()
def show_predictions(model, dataset, device, n_samples: int = 4, indices=None, figsize_per_row=(12, 3)):
    """
    Qualitative comparison grid: original image | ground truth | predicted
    saliency map | prediction overlaid on the image. One row per sample.

    dataset: a SaliencyDataset instance (not a DataLoader — indexes directly
    so specific samples can be requested).
    indices: optional list of dataset indices to show; if None, picks
    `n_samples` evenly spaced samples so you're not just looking at the
    first few (which can be unrepresentative if the dataset isn't shuffled).
    """
    model.to(device)
    model.eval()

    if indices is None:
        indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    n_samples = len(indices)

    fig, axes = plt.subplots(n_samples, 4, figsize=(figsize_per_row[0], figsize_per_row[1] * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground truth", "Prediction", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    for row, idx in enumerate(indices):
        # dataset[idx] is a 3-tuple when built with return_mask=True (e.g. the
        # MIT1003 dataset from get_mit1003_dataset); the mask isn't needed for
        # display, just unpacked and discarded.
        item = dataset[idx]
        image, target = item[0], item[1]
        pred = model(image.unsqueeze(0).to(device))
        fused, _ = unwrap_prediction(pred)
        # index the batch (0) and channel (0) dims explicitly rather than squeeze():
        # squeeze() also collapses the spatial dims if either happens to be 1
        # (e.g. a 1x1 output from a misconfigured decoder), turning the map into
        # a 0-d scalar that plt.imshow then rejects outright.
        pred_map = fused[0, 0].cpu().numpy()

        img_np = _denormalize_image(image)
        target_np = target.squeeze().numpy()

        axes[row, 0].imshow(img_np)
        axes[row, 1].imshow(target_np, cmap="hot")
        axes[row, 2].imshow(pred_map, cmap="hot")

        axes[row, 3].imshow(img_np)
        axes[row, 3].imshow(pred_map, cmap="jet", alpha=0.5)

        for col in range(4):
            axes[row, col].axis("off")

    fig.tight_layout()
    return fig

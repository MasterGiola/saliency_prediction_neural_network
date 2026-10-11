"""
Dataset and loaders for agreement training.

Extends data.SaliencyDataset, which yields (image, saliency_map). This
yields (image, saliency_map, agreement_target), joining the cached .npz from
agreement_targets.py onto the images by filename stem.

The join is strict on purpose: any image without a cached target is dropped
with a warning rather than filled with a default, since a zero-filled target
looks identical to "observers agreed perfectly".

Targets are z-scored using statistics computed on the training fold only (see
fit_normalizer / apply), so no information about the validation fold's
distribution leaks into training.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD, AspectPreservingResize, _PAD_FILL_RGB
from src.agreement_targets import load_targets


class TargetNormalizer:
    """z-scores the agreement target. Fit on train folds, applied to both."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std) if std > 1e-8 else 1.0

    @classmethod
    def fit(cls, values: np.ndarray) -> "TargetNormalizer":
        finite = values[np.isfinite(values)]
        return cls(mean=finite.mean(), std=finite.std())

    def apply(self, x):
        return (x - self.mean) / self.std

    def invert(self, x):
        return x * self.std + self.mean


class AgreementDataset(Dataset):
    """
    Returns (image, saliency_map, agreement_target).

      image            FloatTensor [3, H, W], ImageNet-normalized
      saliency_map     FloatTensor [1, H, W] in [0, 1]
      agreement_target FloatTensor [] (scalar mode) or [1, grid, grid] (dense mode)

    `target_key` selects which cached column is the label. "js_excess" is the
    intended target; "ioc_nss" and "js_observed" are available so the
    null-correction ablation can be run without recomputing anything.
    """

    def __init__(self, image_dir: str, map_dir: str, targets_path: str,
                 img_size: int = 256, target_key: str = "js_excess",
                 normalizer: TargetNormalizer = None,
                 preserve_aspect_ratio: bool = True, augment: bool = False,
                 indices: np.ndarray = None):
        self.img_size = img_size
        self.target_key = target_key
        self.normalizer = normalizer or TargetNormalizer()
        self.augment = augment

        cache = load_targets(targets_path)
        ids = [str(i) for i in cache["image_id"]]
        values = np.asarray(cache[target_key], dtype=np.float64)

        by_id = {image_id: value for image_id, value in zip(ids, values)
                 if np.isfinite(value)}

        image_paths = sorted(p for p in Path(image_dir).iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

        # MIT1003 fixation maps carry a "_fixMap"/"_fixPts" suffix the source image
        # stem doesn't have (e.g. "img001_fixMap.jpeg" for "img001.jpeg"), same as
        # data.SaliencyDataset._pair_by_stem already accounts for. Strip it here too,
        # otherwise every map fails to match its image by stem and the dataset
        # ends up empty.
        map_lookup = {}
        for p in sorted(Path(map_dir).iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = p.stem
            for suffix in ("_fixMap", "_fixPts", "_fixmap"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            # first match wins, so a stray "x_fixPts.jpg" can't displace "x_fixMap.jpg"
            map_lookup.setdefault(stem, p)

        self.samples = []
        missing_target = missing_map = 0
        for path in image_paths:
            if path.stem not in by_id:
                missing_target += 1
                continue
            if path.stem not in map_lookup:
                missing_map += 1
                continue
            self.samples.append((path, map_lookup[path.stem], by_id[path.stem]))

        if missing_target or missing_map:
            print(f"[AgreementDataset] dropped {missing_target} images with no cached "
                  f"target and {missing_map} with no saliency map "
                  f"({len(self.samples)} kept)")
        if not self.samples:
            raise RuntimeError(
                f"No images matched between {image_dir} and {targets_path}. "
                f"Filename stems must agree with the image_id written by "
                f"agreement_targets.py."
            )

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        if preserve_aspect_ratio:
            image_resize = AspectPreservingResize(img_size, fill=_PAD_FILL_RGB)
            self.map_resize = AspectPreservingResize(img_size, fill=0)
        else:
            image_resize = transforms.Resize((img_size, img_size))
            self.map_resize = transforms.Resize((img_size, img_size))

        self.img_transform = transforms.Compose([
            image_resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def raw_targets(self) -> np.ndarray:
        """Un-normalized target values, in sample order. Used to fit the normalizer."""
        return np.array([t for _, _, t in self.samples], dtype=np.float64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, map_path, target = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        sal_map = Image.open(map_path).convert("L")

        # Horizontal flip is safe for the density map and a no-op for the
        # agreement scalar: mirroring an image doesn't change how much the
        # observers disagreed with each other.
        if self.augment and torch.rand(1).item() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            sal_map = sal_map.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.img_transform(image)
        sal_map = self.map_resize(sal_map)
        sal_map = torch.from_numpy(np.array(sal_map, dtype=np.float32) / 255.0).unsqueeze(0)

        value = torch.tensor(self.normalizer.apply(target), dtype=torch.float32)
        return image, sal_map, value


def make_cv_folds(n_samples: int, n_folds: int = 5, seed: int = 42):
    """
    -> list of (train_idx, val_idx) index arrays.

    Plain k-fold over images. MIT1003 has no scene-level grouping metadata, so
    there is no group structure to respect here, though the dataset does
    contain visually similar images from the same capture session, which
    means the cross-validated correlation is very slightly optimistic.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    folds = np.array_split(order, n_folds)
    out = []
    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        out.append((train_idx, val_idx))
    return out


def get_agreement_loaders(config: dict, fold: int = 0):
    """
    Build (train_loader, val_loader, normalizer) for one cross-validation fold.

    Reads config["agreement"]:
        image_dir, map_dir, targets_path, target_key, n_folds, fold_seed
    and reuses config["data"] for img_size / batch_size / num_workers.
    """
    data_cfg = config["data"]
    agr_cfg = config["agreement"]

    common = dict(
        image_dir=agr_cfg["image_dir"],
        map_dir=agr_cfg["map_dir"],
        targets_path=agr_cfg["targets_path"],
        img_size=data_cfg["img_size"],
        target_key=agr_cfg.get("target_key", "js_excess"),
        preserve_aspect_ratio=data_cfg.get("preserve_aspect_ratio", True),
    )

    # One pass with no normalizer to learn the fold split and the train statistics.
    probe = AgreementDataset(**common)
    folds = make_cv_folds(len(probe), agr_cfg.get("n_folds", 5),
                          agr_cfg.get("fold_seed", 42))
    train_idx, val_idx = folds[fold]

    normalizer = TargetNormalizer.fit(probe.raw_targets()[train_idx])

    train_set = AgreementDataset(**common, normalizer=normalizer,
                                 augment=data_cfg.get("augment", True), indices=train_idx)
    val_set = AgreementDataset(**common, normalizer=normalizer,
                               augment=False, indices=val_idx)

    train_loader = DataLoader(
        train_set, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg.get("num_workers", 2), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg.get("num_workers", 2), pin_memory=True,
    )
    return train_loader, val_loader, normalizer

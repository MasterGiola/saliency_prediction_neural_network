"""
Data pipeline for saliency prediction on SALICON (+ optional MIT1003 test set).

Expected folder layout after downloading SALICON (e.g. from Kaggle):

data/
├── salicon/
│   ├── images/
│   │   ├── train/*.jpg
│   │   └── val/*.jpg
│   └── maps/
│       ├── train/*.png       # continuous saliency maps, single channel
│       └── val/*.png
└── mit1003/                   # optional, for extra held-out evaluation
    ├── images/*.jpeg
    └── maps/*.jpeg
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ImageNet mean converted to 0-255 for letterbox padding, so the padded border
# is a neutral gray close to the average pixel the encoder already expects,
# rather than pure black, which once ImageNet-normalized becomes a large
# constant value that reads as a strong, fake high-contrast edge right where
# real content stops.
_PAD_FILL_RGB = tuple(round(m * 255) for m in IMAGENET_MEAN)


class AspectPreservingResize:
    """
    Resize the long side to `size` and pad the short side to make a square,
    instead of squashing both dimensions to (size, size).

    Plain `transforms.Resize((size, size))` distorts every non-square image
    (which is most of them — MIT1003/MIT300/SALICON are mostly ~4:3 or
    16:9), warping exactly the spatial composition (horizon lines, rule of
    thirds, object layout) that saliency prediction depends on. Letterboxing
    keeps every pixel undistorted at the cost of some blank border.

    Use `fill=_PAD_FILL_RGB` for RGB images and `fill=0` for single-channel
    saliency maps (no saliency in the padding, since there's no real scene
    content there). A single instance can't be reused across both because the
    fill value differs, so construct one per (image, map) pair; they must use
    the same `size` and agree on the resulting box so the two line up, which
    SaliencyDataset handles by calling both with identical `size`.
    """

    def __init__(self, size: int, fill=0, resample=Image.BILINEAR):
        self.size = size
        self.fill = fill
        self.resample = resample

    def compute_box(self, orig_size):
        """
        Given the original (pre-resize) (w, h), returns (left, top, new_w, new_h):
        where the resized (undistorted) content lands within the square canvas.
        Pulled out of __call__ so SaliencyDataset can build a matching validity
        mask (content vs. padding) without duplicating this arithmetic; the box
        used to build the mask must be exactly the box actually pasted, or the
        mask drifts out of sync with the real padding.
        """
        w, h = orig_size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        left = (self.size - new_w) // 2
        top = (self.size - new_h) // 2
        return left, top, new_w, new_h

    def __call__(self, img: Image.Image) -> Image.Image:
        left, top, new_w, new_h = self.compute_box(img.size)
        resized = img.resize((new_w, new_h), resample=self.resample)

        canvas = Image.new(img.mode, (self.size, self.size), self.fill)
        canvas.paste(resized, (left, top))
        return canvas


class SaliencyDataset(Dataset):
    """
    Returns (image, saliency_map) pairs, or (image, saliency_map, valid_mask)
    when return_mask=True.
      image:      FloatTensor [3, H, W], ImageNet-normalized
      map:        FloatTensor [1, H, W], values in [0, 1], NOT normalized to sum 1
                  (do that inside the loss if you need a proper distribution, since
                  different losses want different normalizations)
      valid_mask: FloatTensor [1, H, W], 1.0 where the canvas is real (resized)
                  image content, 0.0 where AspectPreservingResize added letterbox
                  padding. All 1.0 when preserve_aspect_ratio=False, since squashing
                  leaves no padding at all — every canvas pixel is real (distorted)
                  content there.

                  Meant for evaluation: metrics computed over the full canvas
                  include the padding border, where the image is a flat gray patch
                  and the target is identically 0 — trivially "easy" pixels that
                  perturb CC/SIM/KLDiv/NSS/AUC in ways that don't reflect real
                  prediction quality. Restricting evaluation to valid_mask removes
                  that. Left off training by default (this flag defaults to False)
                  since letting the network see and be trained against the padding
                  region is a separate question from how you score it, and changing
                  it would mean re-running training, not just evaluation — see
                  src.evaluate.evaluate for how the mask is used.
    """

    def __init__(self, root: str, split: str, img_size: int = 256, augment: bool = False,
                 preserve_aspect_ratio: bool = True, return_mask: bool = False):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.return_mask = return_mask

        # split="" means "flat layout" (MIT1003: images/*.jpeg, maps/*.jpeg) as
        # opposed to SALICON's images/<split>/*.jpg. Path / "" would silently
        # collapse to the same thing, but being explicit keeps the two contracts
        # visible instead of relying on that.
        img_dir = self.root / "images" / split if split else self.root / "images"
        map_dir = self.root / "maps" / split if split else self.root / "maps"

        self.image_paths, self.map_paths = self._pair_by_stem(img_dir, map_dir)

        assert len(self.image_paths) > 0, (
            f"No image/map pairs found in {img_dir} / {map_dir}. Check that both "
            f"directories exist and that filenames correspond (ignoring the map "
            f"suffix and extension)."
        )

        if preserve_aspect_ratio:
            image_resize = AspectPreservingResize(img_size, fill=_PAD_FILL_RGB)
            self.map_resize = AspectPreservingResize(img_size, fill=0)
        else:
            # old behavior, kept for A/B comparison against the new default
            image_resize = transforms.Resize((img_size, img_size))
            self.map_resize = transforms.Resize((img_size, img_size))
        # self.map_resize doubles as the mask's box source when letterboxing is on
        # (both self.map_resize.compute_box(...) calls below need this same instance,
        # since it's built with this dataset's img_size and its box IS the box that
        # was actually pasted; computing it separately risks drift).

        self.img_transform = transforms.Compose([
            image_resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def _pair_by_stem(img_dir: Path, map_dir: Path):
        """
        Pair images to saliency maps by filename stem rather than by position in
        two independently sorted lists.

        Positional pairing (the previous behavior) is only correct when both
        directories contain exactly the same names in the same order. That holds
        for SALICON but is fragile for MIT1003, where maps carry a "_fixMap"
        suffix and the fixation-map zip contains other map types besides the one
        we want. A single extra or missing file there shifts every subsequent
        pair by one, silently scoring each prediction against the wrong ground
        truth — the kind of failure that produces plausible-looking but
        meaningless metrics rather than a crash.

        Map stems are normalized by stripping known suffixes so that
        "img001.jpeg" pairs with "img001_fixMap.jpg". Images with no matching map
        are dropped (with a warning) rather than shifting the alignment.
        """
        map_by_stem = {}
        for p in sorted(map_dir.glob("*.*")):
            stem = p.stem
            for suffix in ("_fixMap", "_fixPts", "_fixmap"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            # first match wins, so a stray "x_fixPts.jpg" can't displace "x_fixMap.jpg"
            map_by_stem.setdefault(stem, p)

        image_paths, map_paths, unmatched = [], [], []
        for img in sorted(img_dir.glob("*.*")):
            if img.stem in map_by_stem:
                image_paths.append(img)
                map_paths.append(map_by_stem[img.stem])
            else:
                unmatched.append(img.name)

        if unmatched:
            preview = ", ".join(unmatched[:5])
            more = f" (+{len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
            print(f"WARNING: {len(unmatched)} image(s) in {img_dir} have no matching "
                  f"saliency map and were skipped: {preview}{more}")

        return image_paths, map_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        sal_map = Image.open(self.map_paths[idx]).convert("L")
        orig_size = image.size  # (w, h), BEFORE any resize — what compute_box needs

        # random horizontal flip — safe for saliency (mirrors both together),
        # avoid vertical flip (breaks the top=sky/bottom=ground prior)
        flip = self.augment and torch.rand(1).item() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            sal_map = sal_map.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.img_transform(image)

        sal_map = self.map_resize(sal_map)
        sal_map = torch.from_numpy(np.array(sal_map, dtype=np.float32) / 255.0).unsqueeze(0)

        if self.return_mask:
            valid_mask = self._build_valid_mask(orig_size, flip)
            return image, sal_map, valid_mask
        return image, sal_map

    def _build_valid_mask(self, orig_size, flip: bool):
        """
        [1, img_size, img_size] float tensor: 1.0 = real content, 0.0 = letterbox
        padding. Built from the SAME box math AspectPreservingResize.__call__ uses
        (via compute_box), so it can't drift out of sync with the actual padding —
        if that math changes, the mask changes with it.

        Under preserve_aspect_ratio=False there's no padding at all (squashing
        distorts every pixel but doesn't add any blank border), so the mask is
        unconditionally all-ones there.

        `flip` mirrors whatever flip was already applied to image/map, for
        correctness in general — currently a no-op in practice since
        AspectPreservingResize always horizontally centers the box, so a
        horizontal-only flip leaves a centered mask unchanged either way.
        """
        if not self.preserve_aspect_ratio:
            mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
        else:
            left, top, new_w, new_h = self.map_resize.compute_box(orig_size)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            mask[top:top + new_h, left:left + new_w] = 1.0
            if flip:
                mask = np.fliplr(mask).copy()
        return torch.from_numpy(mask).unsqueeze(0)


def get_dataloaders(config: dict):
    data_cfg = config["data"]

    preserve_aspect_ratio = data_cfg.get("preserve_aspect_ratio", True)
    train_set = SaliencyDataset(
        root=data_cfg["salicon_root"], split="train",
        img_size=data_cfg["img_size"], augment=data_cfg.get("augment", True),
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
    # return_mask=True on val only: masking is an evaluation concern (see
    # SaliencyDataset's docstring and src.evaluate.evaluate) — training still
    # sees the padding border same as before. This intentionally means
    # train_loader yields 2-tuples and val_loader yields 3-tuples; train_one_epoch
    # and evaluate() are separate functions so that's fine, and evaluate() handles
    # both arities defensively regardless of which loader it's given.
    val_set = SaliencyDataset(
        root=data_cfg["salicon_root"], split="val",
        img_size=data_cfg["img_size"], augment=False,
        preserve_aspect_ratio=preserve_aspect_ratio, return_mask=True,
    )

    train_loader = DataLoader(
        train_set, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg.get("num_workers", 2), pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg.get("num_workers", 2), pin_memory=True,
    )
    return train_loader, val_loader


def get_mit1003_dataset(config: dict):
    """Raw Dataset (not DataLoader) for MIT1003 — used directly by
    src.visualize.show_predictions, which needs indexable samples rather
    than a batched/shuffled DataLoader. Returns None if mit1003_root isn't
    configured.

    return_mask=True: MIT1003 is exclusively a held-out evaluation set (never
    trained on), so masking the letterbox padding out of its metrics is always
    appropriate here — see SaliencyDataset's docstring.
    """
    data_cfg = config["data"]
    root = data_cfg.get("mit1003_root")
    if not root:  # catches both "key missing" and "key present but null/None/empty"
        return None
    return SaliencyDataset(
        root=root, split="", img_size=data_cfg["img_size"], augment=False,
        preserve_aspect_ratio=data_cfg.get("preserve_aspect_ratio", True),
        return_mask=True,
    )


def get_mit1003_loader(config: dict):
    """Optional extra held-out test set."""
    dataset = get_mit1003_dataset(config)
    if dataset is None:
        return None
    data_cfg = config["data"]
    return DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=False)

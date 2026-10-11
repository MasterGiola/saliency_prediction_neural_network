"""
Per-observer fixation data for MIT1003 and SALICON.

The existing data.py consumes *aggregate* saliency maps: one PNG per image,
already averaged over everybody who viewed it. That averaging is exactly the
step this module undoes. Nothing downstream of data.py knows that observers
exist, so agreement targets cannot be computed anywhere in the current
pipeline without this layer.

Both datasets are reduced to the same in-memory record:

    ObserverRecord(
        image_id   = "i05june05_static_street_boston_p1010764",
        width, height = original image size in pixels,
        fixations  = [ ndarray[n_i, 2], ... ]   one array per observer,
                     each row (x, y) in NORMALIZED [0, 1] image coordinates
    )

Normalized coordinates mean the rest of the pipeline never has to care that
MIT1003 is 1024-wide and SALICON is 640-wide, and that agreement statistics
computed on one are dimensionally comparable to the other.

--------------------------------------------------------------------------
Verifying the on-disk layout
--------------------------------------------------------------------------
The on-disk layouts below are what the standard distributions ship, but both
have variants in circulation. Run

    python -m src.observers --dataset mit1003 --root /path/to/mit1003 --inspect

which prints observer counts, fixation counts and coordinate ranges for a
handful of images. If x/y look transposed (ranges hitting 1.0 on the wrong
axis for a landscape image) or counts look implausible (<3 or >40 fixations
for 3 s of viewing), fix the adapter before computing any targets. Silently
transposed coordinates produce agreement numbers that look plausible and
mean nothing.

Expected layouts:

  MIT1003 (Judd et al. 2009) — eye tracking, 15 observers, 3 s free viewing
    mit1003/
    ├── ALLSTIMULI/*.jpeg          # the images
    └── DATA/
        ├── ajs/*.mat              # one folder per observer, one .mat per image
        ├── CNG/*.mat
        └── ...                    # 15 observer folders

  SALICON — mouse-contingent tracking, MS-COCO images
    salicon/
    └── fixations/
        ├── fixations_train2014.json
        └── fixations_val2014.json
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# One degree of visual angle as a fraction of image width.
#
# Judd et al. recorded MIT1003 at 1024 px wide on a 19" display at 48 cm, which
# puts one degree of visual angle at roughly 35 px — so ~0.034 of the image
# width. That is the standard smoothing width for turning discrete fixations
# into a density (it approximates the size of the fovea, i.e. the region a
# single fixation actually samples at high acuity).
#
# Expressed as a fraction rather than pixels so the same constant applies at
# whatever grid resolution the targets are computed on, and so SALICON (a
# different display geometry, and a mouse rather than an eye) uses a defensible
# default instead of an accidental one. The SALICON value should be treated
# as a proxy.
SIGMA_FRAC_1DVA = 0.034

# This MIT1003 redistribution's DATA/<observer>/<image>.mat files carry only
# `DATA.eyeData` — a raw ~240 Hz gaze trace (measured: ~725 rows for a
# nominal 3 s trial, 725/3 ~= 242 Hz — consistent with the ETL 400 eye
# tracker Judd et al. (2009) used), not pre-clustered fixations. Treating each
# raw sample as its own fixation produces plausible-looking but meaningless
# numbers: `--inspect` showed a median of ~722 "fixations" per observer
# against an expected ~10. `_detect_fixations` below runs I-DT
# (dispersion-threshold identification; Salvucci & Goldberg, 2000) to collapse
# the trace into genuine fixations before anything downstream sees it.
GAZE_SAMPLE_RATE_HZ = 240.0

# Minimum fixation duration — standard I-DT default. Below this, a
# low-dispersion window is more likely a slow saccade than a real fixation.
FIXATION_MIN_DURATION_S = 0.10

# Distinguishes an unprocessed ~240 Hz trace (hundreds of rows for a 3 s
# trial) from an already-clustered fixation list (typically 3-15 rows). Only
# traces above this go through fixation detection — running I-DT on an
# already-sparse list could leave too few samples to fill even one dispersion
# window and silently drop everything, which would be worse than doing
# nothing.
RAW_TRACE_ROW_THRESHOLD = 60


def _detect_fixations(gaze: np.ndarray, disp_thresh_px: float,
                      min_duration_samples: int) -> np.ndarray:
    """
    Dispersion-threshold (I-DT) fixation detection on a raw gaze trace.

    Grows a window from each start sample while the bounding-box dispersion
    (width + height, in pixels) stays under `disp_thresh_px`; once at least
    `min_duration_samples` fit inside that window, keeps growing it as long
    as dispersion allows, then emits the window's centroid as one fixation
    and continues from where it left off. Points that never form a
    sufficiently-long low-dispersion window (pure saccade) are dropped, same
    as this module already drops out-of-frame fixations elsewhere.

    -> ndarray [n_fixations, 2], one centroid row per detected fixation.
    """
    n = len(gaze)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n < min_duration_samples:
        # Too short a trace to apply duration thresholding meaningfully —
        # treat the whole thing as one fixation rather than discarding it.
        return gaze.mean(axis=0, keepdims=True)

    fixations = []
    start = 0
    while start <= n - min_duration_samples:
        end = start + min_duration_samples
        window = gaze[start:end]
        xmin, ymin = window.min(axis=0)
        xmax, ymax = window.max(axis=0)
        if (xmax - xmin) + (ymax - ymin) <= disp_thresh_px:
            while end < n:
                x, y = gaze[end]
                nxmin, nymin = min(xmin, x), min(ymin, y)
                nxmax, nymax = max(xmax, x), max(ymax, y)
                if (nxmax - nxmin) + (nymax - nymin) > disp_thresh_px:
                    break
                xmin, ymin, xmax, ymax = nxmin, nymin, nxmax, nymax
                end += 1
            fixations.append(gaze[start:end].mean(axis=0))
            start = end
        else:
            start += 1

    if not fixations:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(fixations, axis=0)


@dataclass
class ObserverRecord:
    """All observers for one image. `fixations[i]` is observer i's scanpath."""
    image_id: str
    width: int
    height: int
    fixations: list = field(default_factory=list)  # list of ndarray[n_i, 2], (x, y) in [0, 1]

    @property
    def n_observers(self) -> int:
        return len(self.fixations)

    @property
    def n_fixations(self) -> int:
        return int(sum(len(f) for f in self.fixations))

    def pooled(self) -> np.ndarray:
        """All observers' fixations concatenated — the sample the aggregate map is built from."""
        if not self.fixations:
            return np.zeros((0, 2), dtype=np.float64)
        return np.concatenate(self.fixations, axis=0)

    def subsample_observers(self, n: int, rng: np.random.Generator) -> "ObserverRecord":
        """
        Keep a random n of the observers.

        Agreement statistics are N-dependent: with more observers you see more
        of the true spread, so raw JS/IOC values are not comparable between an
        image viewed by 15 people and one viewed by 40. Every image must be
        reduced to a common N before its target is computed or compared. This
        is the mechanism for that — see agreement_targets.compute_dataset.
        """
        if n >= self.n_observers:
            return self
        idx = rng.choice(self.n_observers, size=n, replace=False)
        return ObserverRecord(
            image_id=self.image_id, width=self.width, height=self.height,
            fixations=[self.fixations[i] for i in idx],
        )


# ---------------------------------------------------------------------------
# Density rendering
# ---------------------------------------------------------------------------

def render_density(fixations: np.ndarray, grid: int = 64,
                   sigma_frac: float = SIGMA_FRAC_1DVA) -> np.ndarray:
    """
    Discrete fixations -> normalized density on a (grid x grid) map, summing to 1.

    Deliberately computed on a coarse fixed grid rather than at model
    resolution. Entropy — and therefore every quantity in agreement_targets —
    depends on the discretization: a finer grid mechanically raises the entropy
    of the same fixation set. Fixing the grid here is what makes the numbers
    comparable across images and across datasets; do not tie it to
    data.img_size, as that would silently change every target.

    Returns a uniform density if there are no fixations, which keeps the
    entropy arithmetic downstream well-defined (an observer contributing no
    fixations carries no information, and maximum entropy is the natural
    encoding of that).
    """
    if len(fixations) == 0:
        return np.full((grid, grid), 1.0 / (grid * grid), dtype=np.float64)

    # Bin fixations onto the grid first, then blur once. Equivalent to summing
    # one Gaussian per fixation, but O(grid^2) instead of O(n * grid^2) and it
    # lets the permutation null (which re-renders thousands of times) stay cheap.
    hist = np.zeros((grid, grid), dtype=np.float64)
    xs = np.clip((fixations[:, 0] * grid).astype(int), 0, grid - 1)
    ys = np.clip((fixations[:, 1] * grid).astype(int), 0, grid - 1)
    np.add.at(hist, (ys, xs), 1.0)  # row = y, col = x

    sigma_px = max(sigma_frac * grid, 0.5)
    density = _gaussian_blur(hist, sigma_px)

    total = density.sum()
    if total <= 0:
        return np.full((grid, grid), 1.0 / (grid * grid), dtype=np.float64)
    return density / total


def _gaussian_blur(x: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur. Uses scipy when available, falls back to a
    hand-rolled separable convolution so this module has no hard scipy dep."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(x, sigma=sigma_px, mode="constant", cval=0.0)
    except ImportError:
        radius = max(1, int(round(3 * sigma_px)))
        t = np.arange(-radius, radius + 1, dtype=np.float64)
        k = np.exp(-(t ** 2) / (2 * sigma_px ** 2))
        k /= k.sum()
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, x)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
        return out


def render_observer_densities(record: ObserverRecord, grid: int = 64,
                              sigma_frac: float = SIGMA_FRAC_1DVA) -> np.ndarray:
    """-> ndarray [n_observers, grid, grid], each slice summing to 1."""
    return np.stack([render_density(f, grid, sigma_frac) for f in record.fixations], axis=0)


# ---------------------------------------------------------------------------
# MIT1003
# ---------------------------------------------------------------------------

def load_mit1003(root: str, limit: int = None) -> list:
    """
    -> list[ObserverRecord], one per image, observers in folder-name order.

    Judd's .mat files nest the fixation array inside a MATLAB struct whose
    field names differ slightly between redistributions, so rather than
    hard-coding a path like `data.DATA[0].eyeData` we walk the struct and take
    the largest 2-column (or wider) numeric array we find. In every layout
    we've seen, that is the scanpath: columns 0 and 1 are x and y in pixels,
    and any further columns are timing/pupil data we don't need.

    Fixations falling outside the image are dropped rather than clipped —
    clipping would pile spurious mass onto the border, which inflates apparent
    agreement (everyone "agrees" on the edge).
    """
    from scipy.io import loadmat
    from PIL import Image

    root = Path(root)
    stim_dir = _first_existing(root, ["ALLSTIMULI", "images", "stimuli"])
    data_dir = _first_existing(root, ["DATA", "data"])
    if stim_dir is None or data_dir is None:
        raise FileNotFoundError(
            f"Expected ALLSTIMULI/ and DATA/ under {root}. Found: "
            f"{sorted(p.name for p in root.iterdir())[:10]}"
        )

    observer_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not observer_dirs:
        raise FileNotFoundError(f"No per-observer subfolders in {data_dir}")

    image_paths = sorted(p for p in stim_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        image_paths = image_paths[:limit]

    records = []
    for img_path in image_paths:
        image_id = img_path.stem
        with Image.open(img_path) as im:
            width, height = im.size

        fixations = []
        for obs_dir in observer_dirs:
            mat_path = obs_dir / f"{image_id}.mat"
            if not mat_path.exists():
                continue
            raw = _extract_scanpath(loadmat(mat_path, squeeze_me=True, struct_as_record=False))
            if raw is None or len(raw) == 0:
                continue
            raw = np.asarray(raw, dtype=np.float64)[:, :2]
            if raw.shape[0] > RAW_TRACE_ROW_THRESHOLD:
                disp_thresh_px = SIGMA_FRAC_1DVA * width
                min_duration_samples = max(1, round(FIXATION_MIN_DURATION_S * GAZE_SAMPLE_RATE_HZ))
                raw = _detect_fixations(raw, disp_thresh_px, min_duration_samples)
                if len(raw) == 0:
                    continue
            xy = raw / np.array([width, height], dtype=np.float64)
            inside = np.all((xy >= 0.0) & (xy <= 1.0), axis=1)
            fixations.append(xy[inside])

        if fixations:
            records.append(ObserverRecord(image_id, width, height, fixations))

    return records


def _first_existing(root: Path, names) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _extract_scanpath(mat_obj, _depth: int = 0):
    """Recursively walk a loadmat result and return the largest [n, >=2] numeric array."""
    if _depth > 6:
        return None

    best = None
    def consider(arr):
        nonlocal best
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
            if best is None or arr.shape[0] > best.shape[0]:
                best = arr

    if isinstance(mat_obj, dict):
        for key, value in mat_obj.items():
            if key.startswith("__"):
                continue
            found = _extract_scanpath(value, _depth + 1)
            if found is not None:
                consider(found)
    elif isinstance(mat_obj, np.ndarray):
        if mat_obj.dtype.kind in "fiu":
            consider(np.atleast_2d(mat_obj))
        else:  # object/struct array
            for item in mat_obj.ravel():
                found = _extract_scanpath(item, _depth + 1)
                if found is not None:
                    consider(found)
    elif hasattr(mat_obj, "_fieldnames"):  # scipy mat_struct
        for name in mat_obj._fieldnames:
            found = _extract_scanpath(getattr(mat_obj, name), _depth + 1)
            if found is not None:
                consider(found)

    return best


# ---------------------------------------------------------------------------
# SALICON
# ---------------------------------------------------------------------------

def load_salicon(fixation_json: str, limit: int = None) -> list:
    """
    -> list[ObserverRecord] from a SALICON fixations_*.json.

    SALICON's annotation file is COCO-shaped: an `images` list carrying id /
    width / height, and an `annotations` list where each entry is ONE worker's
    trace for ONE image. The per-worker grouping is the thing we need and the
    thing the released PNG maps have already thrown away.

    Coordinate convention is the trap here. SALICON stores fixations as
    [row, col] (i.e. y first), matching MATLAB indexing, not [x, y]. We swap on
    read so everything downstream is (x, y) like MIT1003. If the --inspect
    output shows the aspect ratio of the coordinate cloud fighting the image
    aspect ratio, this is the line to revisit.
    """
    with open(fixation_json, "r") as f:
        blob = json.load(f)

    sizes = {img["id"]: (img["width"], img["height"]) for img in blob.get("images", [])}

    by_image = {}
    for ann in blob.get("annotations", []):
        image_id = ann["image_id"]
        raw = ann.get("fixations")
        if raw is None:
            gaze = ann.get("gaze") or []
            raw = [entry[0] for entry in gaze if entry]  # [[x, y], t, ...] -> [x, y]
            raw = [[p[1], p[0]] for p in raw]  # keep the [row, col] convention below
        if not raw:
            continue
        by_image.setdefault(image_id, []).append(np.asarray(raw, dtype=np.float64))

    records = []
    for image_id, traces in by_image.items():
        width, height = sizes.get(image_id, (640, 480))
        fixations = []
        for trace in traces:
            xy = trace[:, [1, 0]]  # [row, col] -> [x, y]
            xy = xy / np.array([width, height], dtype=np.float64)
            inside = np.all((xy >= 0.0) & (xy <= 1.0), axis=1)
            fixations.append(xy[inside])
        records.append(ObserverRecord(str(image_id), width, height, fixations))
        if limit and len(records) >= limit:
            break

    return records


def load_observers(dataset: str, root: str, limit: int = None) -> list:
    """Dispatch on dataset name. `root` is the dataset root for mit1003, or the
    path to a fixations_*.json for salicon."""
    if dataset == "mit1003":
        return load_mit1003(root, limit=limit)
    if dataset == "salicon":
        return load_salicon(root, limit=limit)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Inspection CLI — run this before computing any targets
# ---------------------------------------------------------------------------

def inspect(records: list, n_show: int = 5) -> None:
    if not records:
        print("No records loaded.")
        return

    n_obs = np.array([r.n_observers for r in records])
    per_obs_fix = np.array([len(f) for r in records for f in r.fixations])

    print(f"images loaded          : {len(records)}")
    print(f"observers per image    : min {n_obs.min()}  median {np.median(n_obs):.0f}  max {n_obs.max()}")
    print(f"fixations per observer : min {per_obs_fix.min()}  median {np.median(per_obs_fix):.0f}  "
          f"max {per_obs_fix.max()}")
    print()
    print("Sanity checks:")
    print(f"  fixations/observer in a plausible 3-15 range?  "
          f"{'OK' if 3 <= np.median(per_obs_fix) <= 20 else 'SUSPECT — check the adapter'}")
    print(f"  observer count constant across images?         "
          f"{'yes' if n_obs.min() == n_obs.max() else 'no — subsample to a common N'}")
    print()
    for record in records[:n_show]:
        pooled = record.pooled()
        print(f"  {record.image_id[:44]:<44} {record.width}x{record.height} "
              f"obs={record.n_observers:<3} fix={record.n_fixations:<4} "
              f"x[{pooled[:, 0].min():.2f},{pooled[:, 0].max():.2f}] "
              f"y[{pooled[:, 1].min():.2f},{pooled[:, 1].max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description="Load and inspect per-observer fixation data.")
    parser.add_argument("--dataset", choices=["mit1003", "salicon"], required=True)
    parser.add_argument("--root", required=True,
                        help="mit1003: dataset root. salicon: path to fixations_*.json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    records = load_observers(args.dataset, args.root, limit=args.limit)
    if args.inspect or True:
        inspect(records)


if __name__ == "__main__":
    main()

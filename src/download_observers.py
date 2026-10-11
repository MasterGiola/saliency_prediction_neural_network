"""
Per-observer fixation data for MIT1003 — the third archive.

download.download_mit1003 fetches ALLSTIMULI.zip and ALLFIXATIONMAPS.zip,
which give you images/ and maps/, the aggregate fixation maps averaged over
all 15 observers. That is exactly what the density pipeline needs and
exactly what the agreement work cannot use.

The individual scanpaths live in a third archive, DATA.zip, on the same page.
This module fetches it. It is a separate file rather than an edit to
download.py so that the agreement work cannot conflict with or clobber code
that already works.

Two behavioural differences from download_mit1003, both intentional:

  - It does not early-return when dest_root already exists. This normally
    runs after download_mit1003 has already created that directory, so an
    existence check on the root would skip the download every time. It checks
    for DATA/ specifically instead.
  - It does not delete _raw afterwards. DATA.zip is small and the site is
    flaky; keeping the archive means a re-extract does not need a re-download.

Resulting layout:

    mit1003/
    ├── images/          <- from download_mit1003
    ├── maps/            <- from download_mit1003
    └── DATA/            <- from here
        ├── ajs/*.mat    (one folder per observer, one .mat per image)
        ├── CNG/*.mat
        └── ...          (15 observer folders)

src/observers.py finds images via _first_existing(root, ["ALLSTIMULI", "images",
"stimuli"]), so the images/ folder download_mit1003 already produced is picked up
without any further work.
"""
import shutil
import zipfile
from pathlib import Path

from src.download import MIT1003_BASE_URL, _download_file

# The per-observer archive. If this 404s, open the index page and check the
# current filename — this is an old MIT lab page and the archive names have
# drifted over the years.
MIT1003_DATA_URL = f"{MIT1003_BASE_URL}/DATA.zip"
MIT1003_INDEX_URL = f"{MIT1003_BASE_URL}/index.html"

# 3 s of free viewing at ~3-4 fixations/s. Used only for the sanity check
# below — counts far outside this suggest the .mat layout was misread.
PLAUSIBLE_FIXATIONS = (3, 25)


def download_mit1003_observers(dest_root: str = "data/mit1003", force: bool = False) -> str:
    """
    Downloads DATA.zip and extracts it to dest_root/DATA/.

    Returns dest_root. Safe to call repeatedly — skips if DATA/ already has
    observer subfolders, unless force=True.
    """
    dest_root = Path(dest_root)
    data_dir = dest_root / "DATA"

    if data_dir.exists() and not force:
        existing = [d for d in data_dir.iterdir() if d.is_dir()]
        if existing:
            print(f"{data_dir} already has {len(existing)} observer folders, skipping "
                  f"download (pass force=True to re-download).")
            return str(dest_root)

    raw_dir = dest_root / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_zip = raw_dir / "DATA.zip"

    print("Downloading MIT1003 per-observer fixation data...")
    try:
        _download_file(MIT1003_DATA_URL, data_zip)
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch {MIT1003_DATA_URL} ({exc}).\n"
            f"The MIT page is frequently down and its filenames have changed over "
            f"the years. Open {MIT1003_INDEX_URL} in a browser, find the archive of "
            f"per-subject eye-tracking data, and either update MIT1003_DATA_URL in "
            f"this file or download it manually and unzip it to {data_dir}."
        ) from exc

    print("Extracting per-observer data...")
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_zip) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".mat") or "__MACOSX" in name:
                continue
            parts = Path(name).parts
            if len(parts) < 2:
                # A .mat sitting at the archive root has no observer folder to
                # belong to; skipping it is safer than inventing a grouping,
                # since the observer grouping is the entire point of this data.
                continue
            observer = parts[-2]
            target_dir = data_dir / observer
            target_dir.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target_dir / Path(name).name, "wb") as dst:
                shutil.copyfileobj(src, dst)

    observers = sorted(d for d in data_dir.iterdir() if d.is_dir())
    n_files = sum(len(list(d.glob("*.mat"))) for d in observers)
    print(f"MIT1003 observers ready: {len(observers)} observers, {n_files} .mat files "
          f"in {data_dir}")
    if len(observers) != 15:
        print(f"WARNING: expected 15 observers, found {len(observers)}. The agreement "
              f"targets are N-dependent — pass n_observers explicitly to "
              f"agreement_targets.compute_dataset so every image uses the same count.")
    return str(dest_root)


def verify_observers(dest_root: str = "data/mit1003", n_check: int = 20) -> bool:
    """
    Parse a sample of images and check the numbers are physically plausible.

    This is the check that catches a misread .mat layout. Transposed x/y or a
    wrong field pulled out of the struct produces agreement values that look
    entirely reasonable and mean nothing at all, so it is worth failing loudly
    here rather than discovering it after training.
    """
    from src.observers import load_observers

    records = load_observers("mit1003", str(dest_root), limit=n_check)
    if not records:
        print("FAIL: no records parsed. Check that DATA/<observer>/<image>.mat exists "
              "and that image filenames match between images/ and DATA/.")
        return False

    n_obs = [r.n_observers for r in records]
    per_obs = [len(f) for r in records for f in r.fixations]
    pooled = [r.pooled() for r in records]

    x_max = max(float(p[:, 0].max()) for p in pooled if len(p))
    y_max = max(float(p[:, 1].max()) for p in pooled if len(p))

    checks = [
        (f"parsed {len(records)} images", True),
        (f"observers per image between 10 and 20 (got {min(n_obs)}-{max(n_obs)})",
         10 <= min(n_obs) and max(n_obs) <= 20),
        (f"median fixations per observer in {PLAUSIBLE_FIXATIONS} "
         f"(got {sorted(per_obs)[len(per_obs) // 2]})",
         PLAUSIBLE_FIXATIONS[0] <= sorted(per_obs)[len(per_obs) // 2] <= PLAUSIBLE_FIXATIONS[1]),
        (f"normalized coordinates stay within [0,1] (max x {x_max:.2f}, y {y_max:.2f})",
         x_max <= 1.0 and y_max <= 1.0),
        ("coordinates use a meaningful fraction of the frame",
         x_max > 0.5 and y_max > 0.5),
    ]

    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    passed = all(ok for _, ok in checks)
    if not passed:
        print("\n  >> Do NOT compute targets until these pass. Inspect a single .mat "
              "by hand and adjust observers._extract_scanpath / the x-y ordering.")
    return passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch MIT1003 per-observer fixation data.")
    parser.add_argument("--root", default="data/mit1003")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download_mit1003_observers(args.root, force=args.force)
    print()
    verify_observers(args.root)

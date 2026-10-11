"""
Dataset download helpers for SALICON and MIT1003.

Both functions leave data in the exact images/<split>/*.jpg + maps/<split>/*.png
layout that src/data.py's SaliencyDataset expects, and both download to local
Colab disk (not a mounted Drive path) — no drive.mount() needed for either.

Usage in the launcher notebook:

    from src.download import download_salicon, download_mit1003, verify_dataset

    salicon_root = download_salicon(
        file_id="YOUR_SALICON_ZIP_FILE_ID",
        local_root="/content/data_local/salicon",
    )
    mit1003_root = download_mit1003("/content/data_local/mit1003")

    verify_dataset(salicon_root, "train")
    verify_dataset(salicon_root, "val")
"""
import shutil
import time
import zipfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20):
    """Stream-download a (possibly large) file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))


# ---------------------------------------------------------------------------
# SALICON — downloaded from a zip uploaded to Google Drive and shared as
# "Anyone with the link". No drive.mount() required: gdown pulls it directly
# over HTTP using the file ID from the share link.
# ---------------------------------------------------------------------------

def download_salicon(file_id: str, local_root: str = "/content/data_local/salicon", force: bool = False) -> str:
    """
    Downloads a single publicly-shared Google Drive zip file directly over
    HTTP and extracts it to local Colab disk.

    Setup (do this once):
      1. Upload your SALICON zip to Google Drive, already laid out as:
           salicon.zip
           ├── images/
           │   ├── train/*.jpg
           │   └── val/*.jpg
           └── maps/
               ├── train/*.png
               └── val/*.png
      2. Right-click it -> Share -> General access -> "Anyone with the link"
      3. Copy the share link, e.g.:
           https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
         The FILE_ID is the segment between /d/ and /view.

    Drive enforces an anonymous download quota on publicly shared files. One
    download per session is fine; many rapid downloads of the same file (e.g.
    a whole class hitting the same link) can trigger a temporary quota error,
    in which case wait and retry.

    Since there's no mount, this only gets you the dataset. Checkpoints you
    want to persist across sessions still need a separate plan (e.g. mount
    Drive just for writing checkpoints, or download them manually at the
    end of a session).
    """
    import gdown

    local_root = Path(local_root)

    if local_root.exists() and not force:
        print(f"{local_root} already exists locally, skipping download "
              f"(pass force=True to re-download).")
        return str(local_root)

    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True)

    local_zip = local_root.parent / f"{local_root.name}.zip"

    t0 = time.time()
    print(f"Downloading file {file_id} from public Drive link...")
    gdown.download(id=file_id, output=str(local_zip), quiet=False)
    print(f"Download done in {time.time() - t0:.1f}s, extracting...")

    shutil.unpack_archive(str(local_zip), extract_dir=local_root)
    local_zip.unlink()

    print(f"SALICON ready at {local_root} in {time.time() - t0:.1f}s total")
    return str(local_root)


# ---------------------------------------------------------------------------
# MIT1003 — official site, no login required, direct zip downloads.
# ---------------------------------------------------------------------------

MIT1003_BASE_URL = "http://people.csail.mit.edu/tjudd/WherePeopleLook"
MIT1003_STIMULI_URL = f"{MIT1003_BASE_URL}/ALLSTIMULI.zip"
MIT1003_FIXMAPS_URL = f"{MIT1003_BASE_URL}/ALLFIXATIONMAPS.zip"


def download_mit1003(dest_root: str = "/content/data_local/mit1003", force: bool = False) -> str:
    """
    Downloads and reorganizes MIT1003 into:
      dest_root/images/*.jpeg
      dest_root/maps/*.jpeg

    The official site is an old MIT lab page and occasionally flaky — if the
    request times out or 404s, retry, or check
    http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html manually
    for the current filenames/URLs.
    """
    dest_root = Path(dest_root)

    if dest_root.exists() and not force:
        print(f"{dest_root} already exists locally, skipping download "
              f"(pass force=True to re-download).")
        return str(dest_root)

    raw_dir = dest_root / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    stimuli_zip = raw_dir / "ALLSTIMULI.zip"
    fixmaps_zip = raw_dir / "ALLFIXATIONMAPS.zip"

    print("Downloading MIT1003 stimuli...")
    _download_file(MIT1003_STIMULI_URL, stimuli_zip)
    print("Downloading MIT1003 fixation maps...")
    _download_file(MIT1003_FIXMAPS_URL, fixmaps_zip)

    images_dir = dest_root / "images"
    maps_dir = dest_root / "maps"
    images_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting images...")
    with zipfile.ZipFile(stimuli_zip) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".jpg", ".jpeg")) and "__MACOSX" not in name:
                target = images_dir / Path(name).name
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    print("Extracting fixation maps...")
    with zipfile.ZipFile(fixmaps_zip) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".jpg", ".jpeg")) and "__MACOSX" not in name and "_fixMap" in name:
                target = maps_dir / Path(name).name
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    shutil.rmtree(raw_dir)

    n_images = len(list(images_dir.glob("*")))
    n_maps = len(list(maps_dir.glob("*")))
    print(f"MIT1003 ready: {n_images} images, {n_maps} maps in {dest_root}")
    if n_images != n_maps:
        print(f"WARNING: image/map count mismatch ({n_images} vs {n_maps}) — "
              f"check filenames match between the two zips before training.")
    return str(dest_root)


def verify_dataset(root: str, split: str = ""):
    """
    Quick sanity check that mirrors how SaliencyDataset actually pairs files.

    Pass split="" for a flat layout (MIT1003). Reports raw counts, how many
    pairs survive stem matching, and prints the first few pairs so you can
    eyeball that names correspond — a count check alone can't catch
    misalignment, only missing files.
    """
    from src.data import SaliencyDataset  # local import keeps download.py standalone

    root = Path(root)
    img_dir = root / "images" / split if split else root / "images"
    map_dir = root / "maps" / split if split else root / "maps"

    n_img = len(list(img_dir.glob("*.*")))
    n_map = len(list(map_dir.glob("*.*")))
    images, maps = SaliencyDataset._pair_by_stem(img_dir, map_dir)

    label = split or "(flat)"
    print(f"{label}: {n_img} image files, {n_map} map files -> {len(images)} matched pairs")
    assert images, f"No matched pairs in {img_dir} / {map_dir}"

    print("  First pairs:")
    for i, m in list(zip(images, maps))[:3]:
        print(f"    {i.name}  <->  {m.name}")

    Image.open(images[0]).verify()
    Image.open(maps[0]).verify()
    print("  Sample files open correctly.")

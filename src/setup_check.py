"""
Local environment checks.

The Colab launcher can assume a known environment: CUDA present, dependencies
pinned by the image, data downloaded into /content. Running locally none of
that holds, and the failure modes are slow and confusing — a missing scipy
surfaces 30 minutes into a target computation, a CPU-only torch build
surfaces as training that appears to work but takes 40x longer.

check_all() front-loads every one of those into a few seconds.
"""
import importlib
import shutil
import sys
from pathlib import Path

# (module, pip name, why it is needed, hard requirement)
REQUIREMENTS = [
    ("torch", "torch", "everything", True),
    ("torchvision", "torchvision", "data transforms", True),
    ("PIL", "pillow", "image loading", True),
    ("timm", "timm", "encoder backbones", True),
    ("numpy", "numpy", "everything", True),
    ("yaml", "pyyaml", "config loading", True),
    ("tqdm", "tqdm", "progress bars", True),
    ("scipy", "scipy", "loadmat, spearmanr, gaussian_filter", True),
    ("sklearn", "scikit-learn", "the three-feature baseline", True),
    ("matplotlib", "matplotlib", "training curves", False),
    ("cv2", "opencv-python-headless", "face term in the baseline", False),
]


def check_python() -> bool:
    ok = sys.version_info >= (3, 9)
    print(f"  [{'PASS' if ok else 'FAIL'}] python {sys.version_info.major}."
          f"{sys.version_info.minor} (need >= 3.9)")
    return ok


def check_packages() -> bool:
    missing_hard, missing_soft = [], []
    for module, pip_name, why, hard in REQUIREMENTS:
        try:
            importlib.import_module(module)
            status, mark = "PASS", ""
        except ImportError:
            status = "FAIL" if hard else "WARN"
            mark = f"  <- pip install {pip_name}"
            (missing_hard if hard else missing_soft).append(pip_name)
        print(f"  [{status}] {module:<13} {why}{mark}")

    if missing_soft:
        print(f"\n  Optional missing: {' '.join(missing_soft)}")
        if "opencv-python-headless" in missing_soft:
            print("  Without opencv the baseline loses its face feature, which makes it "
                  "weaker and therefore flatters the model. Install it so the "
                  "comparison is actually adversarial.")
    if missing_hard:
        print(f"\n  Install the required ones:\n    pip install {' '.join(missing_hard)}")
    return not missing_hard


def check_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("  [FAIL] torch not installed")
        return False

    if not torch.cuda.is_available():
        print(f"  [FAIL] CUDA unavailable (torch {torch.__version__}, "
              f"built for CUDA {torch.version.cuda})")
        print("         Phase 2 runs fine on CPU. Phases 1/3/4 will be impractically slow.")
        if shutil.which("nvidia-smi"):
            print("         nvidia-smi exists, so a driver is present — you likely have a "
                  "CPU-only torch build. Reinstall from pytorch.org with the CUDA index URL.")
        else:
            print("         No nvidia-smi found either — no NVIDIA driver on this machine.")
        return False

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    print(f"  [PASS] CUDA {torch.version.cuda}, torch {torch.__version__}")
    print(f"  [PASS] {props.name}, {vram:.1f} GB VRAM, {torch.cuda.device_count()} device(s)")
    if vram < 6:
        print(f"         Under 6 GB — lower data.batch_size (try 8) if you hit OOM "
              f"on the fusion tiers, which hold three decoders at once.")
    return True


def check_data(config: dict) -> bool:
    """Verify the layout every phase depends on, and say which phase each blocks."""
    data_cfg = config.get("data", {})
    salicon = Path(data_cfg.get("salicon_root", ""))
    mit = Path(data_cfg.get("mit1003_root", ""))

    checks = []
    for split in ("train", "val"):
        images, maps = salicon / "images" / split, salicon / "maps" / split
        n = len(list(images.glob("*.*"))) if images.exists() else 0
        m = len(list(maps.glob("*.*"))) if maps.exists() else 0
        checks.append((f"SALICON {split}: {n} images / {m} maps", n > 0 and n == m,
                       "phase 1"))

    n_img = len(list((mit / "images").glob("*.*"))) if (mit / "images").exists() else 0
    n_map = len(list((mit / "maps").glob("*.*"))) if (mit / "maps").exists() else 0
    checks.append((f"MIT1003: {n_img} images / {n_map} maps", n_img > 0 and n_img == n_map,
                   "phases 3-4"))

    data_dir = mit / "DATA"
    observers = [d for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []
    n_mat = sum(len(list(d.glob("*.mat"))) for d in observers)
    checks.append((f"MIT1003 observers: {len(observers)} folders / {n_mat} .mat files",
                   len(observers) >= 10 and n_mat > 0, "phase 2 — run download_observers"))

    ok = True
    for label, passed, blocks in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
              + ("" if passed else f"   blocks {blocks}"))
        ok = ok and passed
    return ok


def check_all(config: dict = None) -> bool:
    """Run every check. Returns True only if nothing required is missing."""
    print("=" * 66)
    print("ENVIRONMENT")
    print("=" * 66)
    results = [check_python(), check_packages()]
    print("\nCUDA")
    print("-" * 66)
    cuda_ok = check_cuda()

    if config is not None:
        print("\nDATA")
        print("-" * 66)
        results.append(check_data(config))

    print("\n" + "=" * 66)
    if all(results) and cuda_ok:
        print("Ready. Start with phase 0.")
    elif all(results):
        print("Dependencies and data fine, but no CUDA — phase 2 only.")
    else:
        print("Fix the FAIL lines above before running anything.")
    print("=" * 66)
    return all(results)


if __name__ == "__main__":
    try:
        from src.utils import load_config
        cfg = load_config("configs/config.yaml")
    except Exception:
        cfg = None
        print("(no config found — skipping data checks)\n")
    check_all(cfg)

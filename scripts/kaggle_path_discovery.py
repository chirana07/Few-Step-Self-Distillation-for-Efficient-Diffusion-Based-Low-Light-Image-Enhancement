"""
kaggle_path_discovery.py — bulletproof dataset auto-discovery for Kaggle.

Walks /kaggle/input/ and finds paired (low, high) image folders for:
  - LOL-v2 Real (Train + Test)
  - LOL eval15 (Test only)

Tolerates naming variations:
  * "Low" / "low" / "input" / "lq" / "darken" / "dark"
  * "High" / "high" / "Normal" / "normal" / "GT" / "gt" / "target"
  * "Train" / "train" / "training"
  * "Test" / "test" / "eval" / "eval15" / "testing"
  * arbitrary extra wrapper directories (e.g. /kaggle/input/lol-v2/LOL-v2/Real_captured/...)

Validates each candidate by counting .png/.jpg files and checking that the low
and high directories have the SAME number of files (paired dataset).

Then symlinks the best matches into a canonical layout under
/kaggle/working/data/ that train.py and dataset.py expect:

    /kaggle/working/data/
        Real_captured/
            Train/
                Low/    -> [discovered]
                Normal/ -> [discovered]
            Test/
                Low/    -> [discovered]
                Normal/ -> [discovered]
        eval15/
            low/  -> [discovered]
            high/ -> [discovered]

Usage from a notebook cell:

    from kaggle_path_discovery import (
        discover_all, setup_canonical_symlinks, diagnose, validate_canonical
    )
    diagnose()                                # prints what's available
    discoveries = discover_all()              # returns dict of best matches
    data_root = setup_canonical_symlinks(discoveries)
    validate_canonical(data_root)             # fails loudly if anything wrong
    # then pass --dataset-root data_root to train.py
"""
import os
import sys


# ----- naming aliases (case-insensitive) -----
LOW_NAMES  = {"low", "input", "lq", "darken", "dark"}
HIGH_NAMES = {"high", "normal", "gt", "ground_truth", "groundtruth", "target"}
TRAIN_PARENT = {"train", "training", "trainset"}
TEST_PARENT  = {"test", "eval", "eval15", "testing", "testset"}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _count_images(d):
    """Count image files directly inside d (non-recursive)."""
    if not os.path.isdir(d):
        return 0
    try:
        return sum(1 for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
    except (OSError, PermissionError):
        return 0


def _classify_kind(parent_path):
    """Determine train/test kind from the parent directory name and the full path."""
    parent_name = os.path.basename(parent_path).lower()
    if parent_name in TRAIN_PARENT:
        return "train"
    if parent_name in TEST_PARENT:
        return "test"
    full = parent_path.lower()
    # eval15 is always test
    if "eval15" in full:
        return "test"
    if "/train" in full or full.endswith("/train"):
        return "train"
    if "/test" in full or full.endswith("/test"):
        return "test"
    return "unknown"


def find_candidates(input_root="/kaggle/input"):
    """Walk input_root finding paired (low, high) directory candidates.

    Returns list of dicts:
        {"low_dir": ..., "high_dir": ..., "n_images": N, "kind": "train|test|unknown",
         "parent": ..., "is_real": bool, "is_synthetic": bool, "is_eval15": bool}
    """
    candidates = []
    if not os.path.isdir(input_root):
        return candidates

    for dirpath, dirnames, _ in os.walk(input_root, followlinks=True):
        # Build a case-insensitive lookup for child names
        children_lower = {d.lower(): d for d in dirnames}

        # Find any low-ish child and any high-ish child
        low_child  = next((children_lower[n] for n in LOW_NAMES  if n in children_lower), None)
        high_child = next((children_lower[n] for n in HIGH_NAMES if n in children_lower), None)
        if not (low_child and high_child):
            continue

        low_dir  = os.path.join(dirpath, low_child)
        high_dir = os.path.join(dirpath, high_child)
        n_low  = _count_images(low_dir)
        n_high = _count_images(high_dir)

        # Strict pairing: must have same number of images, and at least 1
        if n_low == 0 or n_high == 0:
            continue
        if n_low != n_high:
            # not strictly paired but record anyway (might be useful for diagnosis)
            continue

        full_lower = dirpath.lower()
        candidates.append({
            "low_dir":  low_dir,
            "high_dir": high_dir,
            "n_images": n_low,
            "kind":     _classify_kind(dirpath),
            "parent":   dirpath,
            "is_real":      ("real" in full_lower),
            "is_synthetic": ("synth" in full_lower),
            "is_eval15":    ("eval15" in full_lower),
        })
    return candidates


def discover_all(input_root="/kaggle/input"):
    """Pick best (lolv2_real_train, lolv2_real_test, eval15_test) from candidates.

    Returns a dict like:
        {"lolv2_real_train": {...},  # or None
         "lolv2_real_test":  {...},
         "eval15_test":      {...}}
    """
    cands = find_candidates(input_root)

    # LOL-v2 Real: prefer paths containing 'real', exclude synthetic
    real = [c for c in cands if c["is_real"] and not c["is_synthetic"]]
    if not real:
        # Fallback: any non-synthetic, non-eval15 candidate
        real = [c for c in cands if not c["is_synthetic"] and not c["is_eval15"]]

    train = max((c for c in real if c["kind"] == "train"),
                key=lambda c: c["n_images"], default=None)
    test  = max((c for c in real if c["kind"] == "test"),
                key=lambda c: c["n_images"], default=None)

    # eval15: any candidate with 'eval15' in path
    eval15_cands = [c for c in cands if c["is_eval15"]]
    if eval15_cands:
        eval15 = max(eval15_cands, key=lambda c: c["n_images"])
    else:
        # Fallback: a small (10-30 image) test pair that's not LOL-v2
        small = [c for c in cands if c["kind"] == "test" and 10 <= c["n_images"] <= 30
                 and not c["is_real"] and not c["is_synthetic"]]
        eval15 = small[0] if small else None

    return {
        "lolv2_real_train": train,
        "lolv2_real_test":  test,
        "eval15_test":      eval15,
    }


def _safe_symlink(src, dst):
    """Make a symlink dst -> src, replacing any existing link/file at dst."""
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Symlink source does not exist: {src}")
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        try:
            if os.path.islink(dst):
                os.remove(dst)
            elif os.path.isdir(dst):
                # only remove if empty; otherwise leave it
                if not os.listdir(dst):
                    os.rmdir(dst)
        except OSError:
            pass
    os.symlink(src, dst)


def setup_canonical_symlinks(discoveries, working_root="/kaggle/working"):
    """Wire the discovered paths into a canonical layout under <working_root>/data/."""
    data_root = os.path.join(working_root, "data")
    os.makedirs(data_root, exist_ok=True)

    train = discoveries.get("lolv2_real_train")
    test  = discoveries.get("lolv2_real_test")
    e15   = discoveries.get("eval15_test")

    if train:
        _safe_symlink(train["low_dir"],  os.path.join(data_root, "Real_captured/Train/Low"))
        _safe_symlink(train["high_dir"], os.path.join(data_root, "Real_captured/Train/Normal"))
    if test:
        _safe_symlink(test["low_dir"],   os.path.join(data_root, "Real_captured/Test/Low"))
        _safe_symlink(test["high_dir"],  os.path.join(data_root, "Real_captured/Test/Normal"))
    if e15:
        _safe_symlink(e15["low_dir"],    os.path.join(data_root, "eval15/low"))
        _safe_symlink(e15["high_dir"],   os.path.join(data_root, "eval15/high"))

    return data_root


def validate_canonical(data_root, require_train=True, require_test=True,
                       require_eval15=False, min_train_images=100, min_test_images=15):
    """Hard-fail if the canonical layout under data_root is missing or malformed.

    Returns a dict of {name: count} on success.
    """
    counts = {}

    def check(rel, label, min_n, required):
        full = os.path.join(data_root, rel)
        n = _count_images(full)
        counts[label] = n
        if required and n < min_n:
            raise RuntimeError(
                f"[validate_canonical] {label} has only {n} images at {full} "
                f"(need >= {min_n}). Re-check Cell 3's discovery output."
            )
        return n

    check("Real_captured/Train/Low",    "lolv2_real_train_low",    min_train_images, require_train)
    check("Real_captured/Train/Normal", "lolv2_real_train_normal", min_train_images, require_train)
    check("Real_captured/Test/Low",     "lolv2_real_test_low",     min_test_images,  require_test)
    check("Real_captured/Test/Normal",  "lolv2_real_test_normal",  min_test_images,  require_test)
    if require_eval15:
        check("eval15/low",  "eval15_low",  10, True)
        check("eval15/high", "eval15_high", 10, True)

    return counts


def diagnose(input_root="/kaggle/input"):
    """Print a human-readable listing of what's available under input_root.

    Always safe to call; never raises.
    """
    print(f"\n=== Contents of {input_root} ===")
    if not os.path.isdir(input_root):
        print(f"  (directory does not exist!)")
        return

    for entry in sorted(os.listdir(input_root)):
        full = os.path.join(input_root, entry)
        if not os.path.isdir(full):
            continue
        n_files = 0
        for _, _, fs in os.walk(full, followlinks=True):
            n_files += len(fs)
            if n_files > 100000:
                break
        print(f"  {entry}/  ({n_files} files total)")

    cands = find_candidates(input_root)
    print(f"\n=== {len(cands)} paired (low, high) directory candidates discovered ===")
    if not cands:
        print("  (none found — your dataset folder structure is not recognized.)")
        print("  Expected: a directory containing two subdirs named like Low+Normal,")
        print("            low+high, input+target, etc., with equal image counts.")
    for c in cands:
        labels = []
        if c["is_real"]:      labels.append("REAL")
        if c["is_synthetic"]: labels.append("SYNTH")
        if c["is_eval15"]:    labels.append("EVAL15")
        tag = ",".join(labels) or "—"
        print(f"  [{c['kind']:<7s}] [{tag:<14s}] n={c['n_images']:<4d}")
        print(f"             low : {c['low_dir']}")
        print(f"             high: {c['high_dir']}")


def print_picks(discoveries):
    """Print which candidates were chosen as the best for each role."""
    labels = {
        "lolv2_real_train": "LOL-v2 Real (Train)",
        "lolv2_real_test":  "LOL-v2 Real (Test)",
        "eval15_test":      "LOL eval15 (Test)",
    }
    print("\n=== Discovery picks ===")
    for key, label in labels.items():
        d = discoveries.get(key)
        if d:
            print(f"  {label:<25s}  n={d['n_images']:<4d}  {d['low_dir']}")
            print(f"  {'':<25s}              {d['high_dir']}")
        else:
            print(f"  {label:<25s}  (not found)")


if __name__ == "__main__":
    # Manual diagnostic mode: `python kaggle_path_discovery.py` from a Kaggle terminal
    diagnose()
    discoveries = discover_all()
    print_picks(discoveries)
    if all(discoveries.get(k) is None for k in discoveries):
        sys.exit("\nNo datasets found. Attach your LOL-v2 / eval15 datasets via the Kaggle right panel.")

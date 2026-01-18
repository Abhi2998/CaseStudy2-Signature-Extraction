"""
make_manifest_and_splits.py

What it does:
1) Scans dataset root that contains folders like: 0001, 0002, ...
2) Inside each folder, finds paired images:
      c-<id>-<k>.jpg  (input / cluttered)
   and
      cf-<id>-<k>.jpg (target / cleaner)
3) Writes:
   - manifest_pairs.csv
   - splits.json (group-based split to avoid leakage)

How to run (recommended):
  python make_manifest_and_splits.py

Optional:
  python make_manifest_and_splits.py --raw_dir "D:\\signature_dataset"

"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

# -----------------------------
# EDIT THIS (hardcode your path)
# -----------------------------
DATASET_ROOT = Path(r"C:\\Users\\Dell\\Desktop\\Case Study 2\\data_gen_signet")  # <-- CHANGE THIS

# Output files (created next to this script, unless you change them)
OUT_MANIFEST = Path("manifest_pairs.csv")
OUT_SPLITS = Path("splits.json")

# File extensions allowed
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Expected filename patterns (edit only if your dataset differs)
C_RE = re.compile(r"^c-(?P<doc_id>\d+)-(?P<k>\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
CF_RE = re.compile(r"^cf-(?P<doc_id>\d+)-(?P<k>\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)


def scan_group(group_dir: Path):
    """Return list of rows for one group folder."""
    inputs = {}
    targets = {}

    for p in group_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue

        m = C_RE.match(p.name)
        if m:
            key = (m.group("doc_id"), m.group("k"))
            inputs[key] = p
            continue

        m = CF_RE.match(p.name)
        if m:
            key = (m.group("doc_id"), m.group("k"))
            targets[key] = p
            continue

    all_keys = sorted(set(inputs.keys()) | set(targets.keys()))
    rows = []
    for (doc_id, k) in all_keys:
        in_path = str(inputs.get((doc_id, k), "")) if (doc_id, k) in inputs else ""
        tgt_path = str(targets.get((doc_id, k), "")) if (doc_id, k) in targets else ""

        rows.append(
            {
                "group_id": group_dir.name,   # e.g., 0001
                "doc_id": doc_id,             # extracted from filename
                "pair_k": k,                  # extracted from filename
                "input_path": in_path,        # c-...
                "target_path": tgt_path,      # cf-...
                "has_input": bool(in_path),
                "has_target": bool(tgt_path),
            }
        )
    return rows


def make_manifest(raw_dir: Path, out_csv: Path):
    group_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    all_rows = []
    for gd in group_dirs:
        all_rows.extend(scan_group(gd))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "doc_id",
                "pair_k",
                "input_path",
                "target_path",
                "has_input",
                "has_target",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    missing_input = sum(1 for r in all_rows if (not r["has_input"]) and r["has_target"])
    missing_target = sum(1 for r in all_rows if r["has_input"] and (not r["has_target"]))
    complete_pairs = sum(1 for r in all_rows if r["has_input"] and r["has_target"])

    print(f"[manifest] groups scanned: {len(group_dirs)}")
    print(f"[manifest] rows: {len(all_rows)}")
    print(f"[manifest] complete pairs: {complete_pairs}")
    print(f"[manifest] missing input (have target only): {missing_input}")
    print(f"[manifest] missing target (have input only): {missing_target}")
    print(f"[manifest] wrote: {out_csv.resolve()}")


def make_splits(raw_dir: Path, out_json: Path, seed: int = 42, train=0.80, val=0.10, test=0.10):
    assert abs((train + val + test) - 1.0) < 1e-9, "train+val+test must sum to 1.0"

    groups = sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])
    rnd = random.Random(seed)
    rnd.shuffle(groups)

    n = len(groups)
    n_train = int(n * train)
    n_val = int(n * val)

    train_groups = sorted(groups[:n_train])
    val_groups = sorted(groups[n_train : n_train + n_val])
    test_groups = sorted(groups[n_train + n_val :])

    out = {
        "seed": seed,
        "ratios": {"train": train, "val": val, "test": test},
        "counts": {
            "groups_total": n,
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
        },
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[splits] wrote: {out_json.resolve()}")
    print(f"[splits] train/val/test groups: {len(train_groups)}/{len(val_groups)}/{len(test_groups)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(DATASET_ROOT), help="Dataset root containing 0001, 0002, ...")
    ap.add_argument("--out_manifest", default=str(OUT_MANIFEST))
    ap.add_argument("--out_splits", default=str(OUT_SPLITS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {raw_dir}")

    make_manifest(raw_dir, Path(args.out_manifest))
    make_splits(raw_dir, Path(args.out_splits), seed=args.seed, train=args.train, val=args.val, test=args.test)


if __name__ == "__main__":
    main()

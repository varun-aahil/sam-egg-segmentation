"""Check splits.json for split leakage across near-duplicate perceptual hash groups.

Recomputes the near-duplicate groups exactly as split.py does:
  - same source-name rule: re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem)
  - phash distance <= 8 on grayscale image
  - transitive union-find grouping
Then reads splits.json and identifies any groups whose images straddle dev and eval splits.

Usage:
  python src/check_split_leakage.py --images data --splits splits.json
"""
import argparse
import csv
import json
import re
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}


def source_name(path):
    return re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem)


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def check_leakage(images_dir="data", splits_path="splits.json", eval_csv_path="out/eval/per_image_eval.csv", hash_dist=8):
    root = Path(images_dir)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    n = len(paths)
    print(f"Loaded {n} dataset images from {root}")

    # Compute perceptual hashes
    hashes = [imagehash.phash(Image.open(p).convert("L")) for p in paths]

    # Transitive union-find grouping exactly matching split.py
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            same_src = source_name(paths[i]) == source_name(paths[j])
            dist = hashes[i] - hashes[j]
            if same_src or dist <= hash_dist:
                parent[find(parent, i)] = find(parent, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(parent, i), []).append(i)
    groups = list(groups.values())
    print(f"Computed {len(groups)} transitive groups from {n} images (phash <= {hash_dist} or same source name)")

    # Read splits.json
    splits_file = Path(splits_path)
    if not splits_file.exists():
        raise FileNotFoundError(f"Missing {splits_path}")
    splits = json.loads(splits_file.read_text(encoding="utf-8"))

    dev_set = set(splits["dev"])
    eval_set = set(splits["eval"])
    print(f"Read {splits_path}: dev={len(dev_set)}, eval={len(eval_set)}")

    # Check each group
    straddling_groups = []
    affected_eval_images = set()

    for g_idx, g in enumerate(groups):
        g_rels = [paths[i].relative_to(root).as_posix() for i in g]
        in_dev = [r for r in g_rels if r in dev_set]
        in_eval = [r for r in g_rels if r in eval_set]

        if in_dev and in_eval:
            straddling_groups.append({
                "group_id": g_idx,
                "group_size": len(g),
                "dev_images": in_dev,
                "eval_images": in_eval,
                "all_images": g_rels,
            })
            affected_eval_images.update(in_eval)

    print("\n" + "=" * 80)
    print("SPLIT LEAKAGE AUDIT RESULTS:")
    print("=" * 80)

    if not straddling_groups:
        print("[PASS] ZERO groups straddle dev and eval splits!")
        print(f"Total groups: {len(groups)}")
        multi_img_groups = [g for g in groups if len(g) > 1]
        print(f"Groups with > 1 image: {len(multi_img_groups)}")
        for idx, g in enumerate(multi_img_groups, 1):
            names = [paths[i].name for i in g]
            split_loc = "dev" if paths[g[0]].relative_to(root).as_posix() in dev_set else "eval"
            print(f"  Group {idx} (size {len(g)}, allocated entirely to {split_loc}): {names}")
        print(f"Affected eval images: 0 / {len(eval_set)}")
    else:
        print(f"[FAIL] Found {len(straddling_groups)} straddling groups!")
        for sg in straddling_groups:
            print(f"\nGroup {sg['group_id']} (size {sg['group_size']}):")
            print(f"  Dev images ({len(sg['dev_images'])}): {sg['dev_images']}")
            print(f"  Eval images ({len(sg['eval_images'])}): {sg['eval_images']}")
        print(f"\nTotal affected eval images: {len(affected_eval_images)} / {len(eval_set)}")

    # Recompute metrics on non-straddling eval images from existing CSV
    eval_csv_p = Path(eval_csv_path)
    if eval_csv_p.exists():
        eval_rows = list(csv.DictReader(open(eval_csv_p, encoding="utf-8")))
        all_eval_ious = [float(r["iou"]) for r in eval_rows]
        all_eval_dices = [float(r["dice"]) for r in eval_rows]
        all_eval_f1s = [float(r["det_f1"]) for r in eval_rows]

        clean_eval_rows = [
            r for r in eval_rows
            if not any(r["image"] in aff for aff in affected_eval_images)
        ]
        clean_eval_ious = [float(r["iou"]) for r in clean_eval_rows]
        clean_eval_dices = [float(r["dice"]) for r in clean_eval_rows]
        clean_eval_f1s = [float(r["det_f1"]) for r in clean_eval_rows]

        print("\n" + "-" * 80)
        print("EVAL METRIC COMPARISON (FULL vs LEAK-FREE EVAL):")
        print("-" * 80)
        print(f"{'Split':<25} {'N':>4} {'Mean IoU':>10} {'Mean Dice':>11} {'Mean Det F1':>12}")
        print("-" * 80)
        print(f"{'Full Eval Split':<25} {len(eval_rows):>4d} {np.mean(all_eval_ious):>10.4f} {np.mean(all_eval_dices):>11.4f} {np.mean(all_eval_f1s):>12.4f}")
        print(f"{'Leak-Free Eval Subset':<25} {len(clean_eval_rows):>4d} {np.mean(clean_eval_ious):>10.4f} {np.mean(clean_eval_dices):>11.4f} {np.mean(clean_eval_f1s):>12.4f}")
        print("-" * 80)

    return straddling_groups, affected_eval_images


def main():
    ap = argparse.ArgumentParser(description="Check splits.json for perceptual hash group leakage")
    ap.add_argument("--images", default="data", help="dataset folder")
    ap.add_argument("--splits", default="splits.json", help="path to splits.json")
    ap.add_argument("--eval-csv", default="out/eval/per_image_eval.csv", help="eval per-image results CSV")
    ap.add_argument("--hash-dist", type=int, default=8, help="max phash distance")
    args = ap.parse_args()

    check_leakage(args.images, args.splits, args.eval_csv, args.hash_dist)


if __name__ == "__main__":
    main()


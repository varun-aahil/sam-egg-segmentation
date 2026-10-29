import argparse
import json
import random
import re
from pathlib import Path

import imagehash
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}


def source_name(path):
    # roboflow adds .rf.<hash> to the original file name
    return re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem)


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="root folder of the downloaded dataset")
    ap.add_argument("--out", default="splits.json")
    ap.add_argument("--hash-dist", type=int, default=8, help="max phash distance to call two images the same scene")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.images)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    n = len(paths)
    hashes = [imagehash.phash(Image.open(p).convert("L")) for p in paths]

    # link images that share a source file name or look near-identical
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            same_src = source_name(paths[i]) == source_name(paths[j])
            if same_src or hashes[i] - hashes[j] <= args.hash_dist:
                parent[find(parent, i)] = find(parent, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(parent, i), []).append(i)
    groups = list(groups.values())

    # split whole groups so near-duplicates never straddle dev and eval
    random.Random(args.seed).shuffle(groups)
    target = round(n * args.eval_frac)
    dev, ev = [], []
    for g in groups:
        if len(ev) + len(g) <= target:
            ev += g
        else:
            dev += g

    rel = lambda idx: sorted(paths[i].relative_to(root).as_posix() for i in idx)
    Path(args.out).write_text(json.dumps({"dev": rel(dev), "eval": rel(ev)}, indent=2))

    print(f"{n} images, {len(groups)} groups -> dev {len(dev)}, eval {len(ev)}")
    big = sorted((g for g in groups if len(g) > 1), key=len, reverse=True)[:5]
    for g in big:
        print(f"group of {len(g)}:", [paths[i].name for i in g[:4]])


if __name__ == "__main__":
    main()

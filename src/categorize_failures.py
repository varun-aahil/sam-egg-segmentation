"""Categorize segmentation failures by scene type and physical root causes.

Produces failure_tags.csv and prints category statistics.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    splits = json.loads(Path("splits.json").read_text(encoding="utf-8"))
    dev_csv = list(csv.DictReader(open("out/eval/per_image_dev.csv", encoding="utf-8")))
    eval_csv = list(csv.DictReader(open("out/eval/per_image_eval.csv", encoding="utf-8")))

    res_map = {}
    for r in dev_csv:
        res_map[r["image"]] = {**r, "split": "dev"}
    for r in eval_csv:
        res_map[r["image"]] = {**r, "split": "eval"}

    all_stems = splits["dev"] + splits["eval"]
    tagged_records = []

    for rel in all_stems:
        p = Path("data") / rel
        stem = p.stem
        info = res_map[stem]
        iou = float(info["iou"])
        dice = float(info["dice"])
        det_f1 = float(info["det_f1"])
        n_gt = int(info["n_gt"])
        n_pred = int(info["n_pred"])
        split = info["split"]

        img = cv2.imread(str(p))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Feature detectors
        green_pct = float(np.mean((h >= 35) & (h <= 85) & (s >= 50)))
        skin_pct = float(np.mean((h >= 5) & (h <= 25) & (s >= 40) & (s <= 180) & (v >= 70)))
        white_pct = float(np.mean((s < 40) & (v > 180)))
        border_mask = np.ones(img.shape[:2], dtype=bool)
        border_mask[40:-40, 40:-40] = False
        dark_border = float(np.mean(v[border_mask] < 40))

        # Known boiling pot stems
        boiling_pot_ids = ["118", "124", "135", "139", "143", "145", "147", "149"]

        # Classification hierarchy
        if any(f"not_damaged_{num}" in stem for num in boiling_pot_ids):
            category = "Egg in boiling water / submerged"
        elif n_gt >= 2:
            category = "Multiple touching eggs"
        elif green_pct > 0.08:
            category = "Foliage / herbs in background (specular highlights)"
        elif skin_pct > 0.18:
            category = "Hand holding egg / fingers occluding"
        elif dark_border > 0.35 or float(np.mean(v)) < 65:
            category = "Backlit / extreme shadows"
        elif white_pct > 0.35:
            category = "White egg on white tray / fridge interior"
        else:
            category = "Tabletop / high-contrast background"

        tagged_records.append({
            "image": stem,
            "split": split,
            "category": category,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "matched_eggs": info["matched_eggs"],
            "iou": f"{iou:.4f}",
            "dice": f"{dice:.4f}",
            "det_f1": f"{det_f1:.4f}",
        })

    # Save failure_tags.csv
    out_csv = Path("failure_tags.csv")
    fieldnames = ["image", "split", "category", "n_gt", "n_pred", "matched_eggs", "iou", "dice", "det_f1"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(tagged_records)

    # Print breakdown table
    categories_agg = {}
    for r in tagged_records:
        cat = r["category"]
        if cat not in categories_agg:
            categories_agg[cat] = {"ious": [], "dices": [], "f1s": []}
        categories_agg[cat]["ious"].append(float(r["iou"]))
        categories_agg[cat]["dices"].append(float(r["dice"]))
        categories_agg[cat]["f1s"].append(float(r["det_f1"]))

    print(f"\nSaved {len(tagged_records)} records to {out_csv}")
    print("\nVisual Failure Categories Breakdown:")
    print("-" * 90)
    print(f"{'Category':<50} {'Count':>6} {'Pct':>7} {'Mean IoU':>10} {'Mean Dice':>11} {'Mean F1':>9}")
    print("-" * 90)
    for cat, vals in sorted(categories_agg.items(), key=lambda x: np.mean(x[1]["ious"])):
        n = len(vals["ious"])
        pct = n / len(tagged_records) * 100
        m_iou = np.mean(vals["ious"])
        m_dice = np.mean(vals["dices"])
        m_f1 = np.mean(vals["f1s"])
        print(f"{cat:<50} {n:>6d} {pct:>6.1f}% {m_iou:>10.4f} {m_dice:>11.4f} {m_f1:>9.4f}")
    print("-" * 90)


if __name__ == "__main__":
    main()


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

    # Explanations of primary physical failure mechanisms
    mechanisms = {
        "White egg on white tray / fridge interior": "Near-zero photometric contrast between white eggshell and tray surface; Otsu floods tray.",
        "Egg in boiling water / submerged": "Bubbling cavitation, water refraction, and dark pot rims break convex boundary and flood liquid.",
        "Backlit / extreme shadows": "High-intensity directional backlight causes glowing core while shell perimeter falls into dark silhouette.",
        "Hand holding egg / fingers occluding": "Fingers occlude egg perimeter; skin tone overlaps shadow gradient, distending contour.",
        "Tabletop / high-contrast background": "Table surface grain, specular table highlights, or soft contact shadows falsely merge into foreground.",
        "Multiple touching eggs": "Touching instances share continuous boundary; requires distance watershed marker separation.",
        "Foliage / herbs in background (specular highlights)": "Egg is brighter than dark leaves in grayscale; specular reflections create occasional false blobs.",
    }

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

        green_pct = float(np.mean((h >= 35) & (h <= 85) & (s >= 50)))
        skin_pct = float(np.mean((h >= 5) & (h <= 25) & (s >= 40) & (s <= 180) & (v >= 70)))
        white_pct = float(np.mean((s < 40) & (v > 180)))
        border_mask = np.ones(img.shape[:2], dtype=bool)
        border_mask[40:-40, 40:-40] = False
        dark_border = float(np.mean(v[border_mask] < 40))

        # Explicit stem checks based on visual verification
        boiling_pot_ids = ["118", "124", "135", "139", "143", "145", "147", "149"]
        white_tray_fridge_ids = ["125", "123", "42", "104", "155"]
        backlit_ids = ["20", "24", "44"]

        if any(f"not_damaged_{num}_" in stem for num in boiling_pot_ids):
            category = "Egg in boiling water / submerged"
        elif any(f"not_damaged_{num}_" in stem for num in white_tray_fridge_ids):
            category = "White egg on white tray / fridge interior"
        elif any(f"not_damaged_{num}_" in stem for num in backlit_ids):
            category = "Backlit / extreme shadows"
        elif n_gt >= 2:
            category = "Multiple touching eggs"
        elif green_pct > 0.08:
            category = "Foliage / herbs in background (specular highlights)"
        elif dark_border > 0.35 or float(np.mean(v)) < 65:
            category = "Backlit / extreme shadows"
        elif skin_pct > 0.18:
            category = "Hand holding egg / fingers occluding"
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
            "primary_failure_mechanism": mechanisms[category],
        })

    # Save failure_tags.csv
    out_csv = Path("failure_tags.csv")
    fieldnames = [
        "image", "split", "category", "n_gt", "n_pred", "matched_eggs",
        "iou", "dice", "det_f1", "primary_failure_mechanism"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(tagged_records)

    # Compute overall category breakdown
    categories_agg = {}
    dev_agg = {}
    eval_agg = {}
    for r in tagged_records:
        cat = r["category"]
        split = r["split"]
        if cat not in categories_agg:
            categories_agg[cat] = {"ious": [], "dices": [], "f1s": []}
            dev_agg[cat] = {"ious": []}
            eval_agg[cat] = {"ious": []}
        categories_agg[cat]["ious"].append(float(r["iou"]))
        categories_agg[cat]["dices"].append(float(r["dice"]))
        categories_agg[cat]["f1s"].append(float(r["det_f1"]))
        if split == "dev":
            dev_agg[cat]["ious"].append(float(r["iou"]))
        else:
            eval_agg[cat]["ious"].append(float(r["iou"]))

    print(f"\nSaved {len(tagged_records)} records to {out_csv}")
    print("\nVisual Failure Categories Breakdown (Full Dataset):")
    print("-" * 110)
    print(f"{'Category':<45} {'Count':>6} {'Pct':>7} {'Mean IoU':>10} {'Mean Dice':>11} {'Mean F1':>9}")
    print("-" * 110)
    for cat, vals in sorted(categories_agg.items(), key=lambda x: np.mean(x[1]["ious"])):
        n = len(vals["ious"])
        pct = n / len(tagged_records) * 100
        m_iou = np.mean(vals["ious"])
        m_dice = np.mean(vals["dices"])
        m_f1 = np.mean(vals["f1s"])
        print(f"{cat:<45} {n:>6d} {pct:>6.1f}% {m_iou:>10.4f} {m_dice:>11.4f} {m_f1:>9.4f}")
    print("-" * 110)

    # Per-category dev vs eval table
    print("\nPer-Category Dev vs Eval Split Breakdown:")
    print("-" * 85)
    print(f"{'Category':<45} {'Dev N':>6} {'Dev IoU':>9} {'Eval N':>7} {'Eval IoU':>10}")
    print("-" * 85)
    for cat in sorted(categories_agg.keys(), key=lambda x: np.mean(categories_agg[x]["ious"])):
        d_n = len(dev_agg[cat]["ious"])
        d_iou = np.mean(dev_agg[cat]["ious"]) if d_n > 0 else 0.0
        e_n = len(eval_agg[cat]["ious"])
        e_iou = np.mean(eval_agg[cat]["ious"]) if e_n > 0 else 0.0
        print(f"{cat:<45} {d_n:>6d} {d_iou:>9.4f} {e_n:>7d} {e_iou:>10.4f}")
    print("-" * 85)

    # Save per_category_dev_eval.json
    dev_eval_table = {}
    for cat in categories_agg:
        d_n = len(dev_agg[cat]["ious"])
        d_iou = float(np.mean(dev_agg[cat]["ious"])) if d_n > 0 else 0.0
        e_n = len(eval_agg[cat]["ious"])
        e_iou = float(np.mean(eval_agg[cat]["ious"])) if e_n > 0 else 0.0
        dev_eval_table[cat] = {
            "dev_count": d_n,
            "dev_iou": d_iou,
            "eval_count": e_n,
            "eval_iou": e_iou,
            "primary_failure_mechanism": mechanisms[cat]
        }
    Path("out/eval/per_category_dev_eval.json").write_text(json.dumps(dev_eval_table, indent=2))


if __name__ == "__main__":
    main()

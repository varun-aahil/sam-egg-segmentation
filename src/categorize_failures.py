"""Categorize segmentation failures by scene type and physical root causes.

Produces failure_tags.csv and prints category statistics.
"""
import csv
import json
from pathlib import Path

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
        "White egg on white tray / fridge interior": "Near-zero photometric contrast between white eggshell and white plastic tray or paper bag; Otsu floods background.",
        "Egg in boiling water / submerged": "Bubbling cavitation, water refraction, and dark pot rims break convex boundary and flood liquid.",
        "Backlit / extreme shadows": "High-intensity directional backlight causes glowing core while shell perimeter falls into dark silhouette.",
        "Hand holding egg / fingers occluding": "Fingers or palm occlude egg perimeter; skin tone overlaps shadow gradient, distending contour into hand.",
        "Multiple touching eggs": "Touching or adjacent instances share continuous boundary; requires distance watershed marker separation.",
        "Foliage / herbs in background (specular highlights)": "Egg contrasts against dark foliage in grayscale; specular leaf reflections create occasional false blobs.",
        "Egg on container / lid / plain surface": "Plastic food-container lids, bowls, metal pans, or plain packaging surfaces create perimeter reflections and false components.",
        "Tabletop / high-contrast background": "Tabletop grain, hearth ash, patterned cloth tablecloths, or laptop keyboard keys introduce background clutter.",
    }

    # Verified Visual Taxonomy mapping by image ID:
    category_map = {
        "Egg in boiling water / submerged": [124, 135, 145, 113, 129, 114, 128, 147],
        "Foliage / herbs in background (specular highlights)": [10, 11, 12, 13, 49],
        "Backlit / extreme shadows": [20, 24, 44],
        "White egg on white tray / fridge interior": [42, 116, 117, 104, 106, 111, 112, 123, 125, 143],
        "Hand holding egg / fingers occluding": [1, 2, 25, 102, 155, 157, 158, 160, 161, 151, 152],
        "Multiple touching eggs": [103, 105, 115, 121, 132, 133, 134, 136, 140, 142, 148, 144],
        "Egg on container / lid / plain surface": [110, 120, 139, 141, 108, 127, 130, 21, 22, 31, 35, 40, 34, 39, 156, 163, 150, 159, 149, 30, 107, 109, 118, 131, 138, 153, 32, 119, 122, 126],
        "Tabletop / high-contrast background": [28, 33, 45, 100, 101, 146, 29, 14, 27, 36, 41, 50, 78, 3, 4, 5, 6, 16, 37, 52, 62, 95, 154, 162, 15, 55, 164, 17, 18, 19, 43]
    }

    id_to_cat = {}
    for cat, ids in category_map.items():
        for i in ids:
            id_to_cat[i] = cat

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

        short_stem = stem.split("_jpg")[0]
        img_id = int(short_stem.replace("not_damaged_", ""))
        category = id_to_cat[img_id]

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
    print(f"{'Category':<52} {'Count':>6} {'Pct':>7} {'Mean IoU':>10} {'Mean Dice':>11} {'Mean F1':>9}")
    print("-" * 110)
    for cat, vals in sorted(categories_agg.items(), key=lambda x: np.mean(x[1]["ious"])):
        n = len(vals["ious"])
        pct = n / len(tagged_records) * 100
        m_iou = np.mean(vals["ious"])
        m_dice = np.mean(vals["dices"])
        m_f1 = np.mean(vals["f1s"])
        print(f"{cat:<52} {n:>6d} {pct:>6.1f}% {m_iou:>10.4f} {m_dice:>11.4f} {m_f1:>9.4f}")
    print("-" * 110)

    # Per-category dev vs eval table
    print("\nPer-Category Dev vs Eval Split Breakdown:")
    print("-" * 92)
    print(f"{'Category':<52} {'Dev N':>6} {'Dev IoU':>9} {'Eval N':>7} {'Eval IoU':>10}")
    print("-" * 92)
    for cat in sorted(categories_agg.keys(), key=lambda x: np.mean(categories_agg[x]["ious"])):
        d_n = len(dev_agg[cat]["ious"])
        d_iou = np.mean(dev_agg[cat]["ious"]) if d_n > 0 else 0.0
        e_n = len(eval_agg[cat]["ious"])
        e_iou = np.mean(eval_agg[cat]["ious"]) if e_n > 0 else 0.0
        print(f"{cat:<52} {d_n:>6d} {d_iou:>9.4f} {e_n:>7d} {e_iou:>10.4f}")
    print("-" * 92)

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

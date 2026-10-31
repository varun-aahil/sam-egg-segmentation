"""Compute composition check re-weighting dev per-category mean IoU by eval category counts."""
import json
import csv
from pathlib import Path


def main():
    tags_path = Path("failure_tags.csv")
    per_cat_path = Path("out/eval/per_category_dev_eval.json")

    with open(per_cat_path, "r", encoding="utf-8") as f:
        per_cat = json.load(f)

    # Verify counts from failure_tags.csv
    with open(tags_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    eval_rows = [r for r in rows if r["split"] == "eval"]
    dev_rows = [r for r in rows if r["split"] == "dev"]

    actual_dev_iou = sum(float(r["iou"]) for r in dev_rows) / len(dev_rows)
    actual_eval_iou = sum(float(r["iou"]) for r in eval_rows) / len(eval_rows)

    tot_eval = sum(v["eval_count"] for v in per_cat.values())
    assert tot_eval == len(eval_rows) == 22, f"Expected 22 eval rows, got {tot_eval}"

    # Re-weight dev per-category mean IoUs by eval counts
    weighted_dev_sum = 0.0
    category_weights = {}

    for cat_name, stats in per_cat.items():
        n_eval = stats["eval_count"]
        dev_iou = stats["dev_iou"]
        weight = n_eval / tot_eval
        weighted_dev_sum += n_eval * dev_iou
        category_weights[cat_name] = {
            "eval_count": n_eval,
            "eval_weight": round(weight, 4),
            "dev_mean_iou": round(dev_iou, 4) if stats["dev_count"] > 0 else None,
            "eval_mean_iou": round(stats["eval_iou"], 4) if stats["eval_count"] > 0 else None,
            "weighted_dev_contribution": round(weight * dev_iou, 4)
        }

    predicted_eval_iou = weighted_dev_sum / tot_eval
    total_gap = actual_dev_iou - actual_eval_iou
    explained_gap = actual_dev_iou - predicted_eval_iou
    unexplained_gap = predicted_eval_iou - actual_eval_iou
    share_explained_pct = (explained_gap / total_gap) * 100.0

    result = {
        "description": "Composition check re-weighting dev per-category mean IoUs by eval category counts to quantify the impact of domain shift.",
        "note": "This is a descriptive check on 22 images, not a significance test.",
        "dataset_split_sizes": {
            "dev_images": len(dev_rows),
            "eval_images": len(eval_rows),
            "total_images": len(rows)
        },
        "metrics": {
            "dev_mean_iou": round(actual_dev_iou, 4),
            "eval_mean_iou": round(actual_eval_iou, 4),
            "predicted_eval_iou_from_reweighting": round(predicted_eval_iou, 4),
            "total_dev_eval_gap": round(total_gap, 4),
            "gap_explained_by_category_composition": round(explained_gap, 4),
            "gap_unexplained_residual": round(unexplained_gap, 4),
            "share_of_gap_explained_pct": round(share_explained_pct, 2)
        },
        "per_category_weights": category_weights
    }

    out_json = Path("out/eval/composition_check.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out_json} successfully.")
    print(f"Dev IoU: {actual_dev_iou:.4f}, Eval IoU: {actual_eval_iou:.4f}")
    print(f"Predicted Eval IoU: {predicted_eval_iou:.4f}")
    print(f"Total Gap: {total_gap:.4f}, Explained: {explained_gap:.4f} ({share_explained_pct:.1f}%)")


if __name__ == "__main__":
    main()

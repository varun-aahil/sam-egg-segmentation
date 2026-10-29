"""Evaluation: compare classical segmentation masks against SAM ground truth.

Reports pixel-level IoU, Dice, precision, recall, and instance-level metrics.
Generates plots and a failure gallery.

Usage:
  python src/evaluate.py --pred out/classical/final/dev/masks --gt out/sam/masks --split dev --data data
  python src/evaluate.py --pred out/classical/final/eval/masks --gt out/sam/masks --split eval --data data
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def load_label_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.astype(np.uint16)


def pixel_metrics(pred, gt):
    """Pixel-level binary metrics."""
    p = pred > 0
    g = gt > 0
    tp = (p & g).sum()
    fp = (p & ~g).sum()
    fn = (~p & g).sum()
    inter = tp
    union = tp + fp + fn
    iou = inter / union if union > 0 else 1.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec}


def instance_metrics(pred, gt, iou_thresh=0.5):
    """Instance-level metrics using Hungarian matching."""
    pred_ids = sorted(set(pred.flat) - {0})
    gt_ids = sorted(set(gt.flat) - {0})

    if not pred_ids and not gt_ids:
        return {"det_tp": 0, "det_fp": 0, "det_fn": 0,
                "det_prec": 1.0, "det_rec": 1.0, "det_f1": 1.0,
                "count_error": 0, "pred_count": 0, "gt_count": 0,
                "matched_ious": []}

    if not pred_ids:
        return {"det_tp": 0, "det_fp": 0, "det_fn": len(gt_ids),
                "det_prec": 0.0, "det_rec": 0.0, "det_f1": 0.0,
                "count_error": -len(gt_ids), "pred_count": 0, "gt_count": len(gt_ids),
                "matched_ious": []}

    if not gt_ids:
        return {"det_tp": 0, "det_fp": len(pred_ids), "det_fn": 0,
                "det_prec": 0.0, "det_rec": 1.0, "det_f1": 0.0,
                "count_error": len(pred_ids), "pred_count": len(pred_ids), "gt_count": 0,
                "matched_ious": []}

    # build cost matrix (negative IoU for Hungarian min-cost)
    cost = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pi in enumerate(pred_ids):
        pm = pred == pi
        for j, gi in enumerate(gt_ids):
            gm = gt == gi
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            cost[i, j] = -(inter / union) if union > 0 else 0

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_ious = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        iou = -cost[r, c]
        if iou >= iou_thresh:
            tp += 1
            matched_ious.append(iou)

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"det_tp": tp, "det_fp": fp, "det_fn": fn,
            "det_prec": prec, "det_rec": rec, "det_f1": f1,
            "count_error": len(pred_ids) - len(gt_ids),
            "pred_count": len(pred_ids), "gt_count": len(gt_ids),
            "matched_ious": matched_ious}


def make_diff_image(img_bgr, pred, gt):
    """Color-coded difference: green=TP, red=FP, blue=FN."""
    h, w = pred.shape
    diff = np.zeros((h, w, 3), np.uint8)
    p = pred > 0
    g = gt > 0
    diff[p & g] = [0, 180, 0]       # true positive: green
    diff[p & ~g] = [0, 0, 200]      # false positive: red
    diff[~p & g] = [200, 0, 0]      # false negative: blue
    # blend with original
    base = (img_bgr * 0.3).astype(np.uint8)
    mask = p | g
    out = base.copy()
    out[mask] = (0.5 * base[mask] + 0.5 * diff[mask]).astype(np.uint8)
    out[~mask] = base[~mask]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="directory of predicted label PNGs")
    ap.add_argument("--gt", required=True, help="directory of ground truth (SAM) label PNGs")
    ap.add_argument("--split", required=True, choices=["dev", "eval"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits-json", default="splits.json")
    ap.add_argument("--out", default="out/eval")
    args = ap.parse_args()

    splits = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
    image_paths = splits[args.split]
    data_root = Path(args.data)
    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)

    out_dir = Path(args.out) / args.split
    (out_dir / "failures").mkdir(parents=True, exist_ok=True)

    results = []
    for rel in image_paths:
        img_path = data_root / rel
        stem = img_path.stem

        pred_path = pred_dir / f"{stem}.png"
        gt_path = gt_dir / f"{stem}.png"

        if not pred_path.exists() or not gt_path.exists():
            continue

        pred = load_label_image(pred_path)
        gt = load_label_image(gt_path)

        px = pixel_metrics(pred, gt)
        inst = instance_metrics(pred, gt)

        results.append({
            "image": stem,
            "rel_path": rel,
            **px,
            **inst
        })

    if not results:
        print("no images to evaluate")
        return

    # per-image results
    ious = [r["iou"] for r in results]
    dices = [r["dice"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    det_precs = [r["det_prec"] for r in results]
    det_recs = [r["det_rec"] for r in results]
    det_f1s = [r["det_f1"] for r in results]
    count_errs = [r["count_error"] for r in results]

    print(f"\n=== {args.split} set: {len(results)} images ===")
    print(f"Pixel IoU:       mean={np.mean(ious):.4f}  median={np.median(ious):.4f}")
    print(f"Pixel Dice:      mean={np.mean(dices):.4f}  median={np.median(dices):.4f}")
    print(f"Pixel Precision: mean={np.mean(precs):.4f}")
    print(f"Pixel Recall:    mean={np.mean(recs):.4f}")
    print(f"Det Precision:   mean={np.mean(det_precs):.4f}")
    print(f"Det Recall:      mean={np.mean(det_recs):.4f}")
    print(f"Det F1:          mean={np.mean(det_f1s):.4f}")
    print(f"Count Error:     mean={np.mean(count_errs):.2f}  "
          f"abs mean={np.mean(np.abs(count_errs)):.2f}")

    # save per-image csv
    import csv
    csv_path = out_dir / "per_image.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "iou", "dice", "precision", "recall",
                                          "det_prec", "det_rec", "det_f1",
                                          "pred_count", "gt_count", "count_error"])
        w.writeheader()
        for r in results:
            w.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]
                        for k in w.fieldnames})

    # summary json
    summary = {
        "split": args.split,
        "n_images": len(results),
        "pixel_iou_mean": float(np.mean(ious)),
        "pixel_iou_median": float(np.median(ious)),
        "pixel_dice_mean": float(np.mean(dices)),
        "pixel_dice_median": float(np.median(dices)),
        "pixel_precision_mean": float(np.mean(precs)),
        "pixel_recall_mean": float(np.mean(recs)),
        "det_precision_mean": float(np.mean(det_precs)),
        "det_recall_mean": float(np.mean(det_recs)),
        "det_f1_mean": float(np.mean(det_f1s)),
        "count_error_mean": float(np.mean(count_errs)),
        "count_error_abs_mean": float(np.mean(np.abs(count_errs))),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- plots ---
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # IoU distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ious, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(ious), color="red", linestyle="--", label=f"mean={np.mean(ious):.3f}")
    ax.set_xlabel("Pixel IoU")
    ax.set_ylabel("Count")
    ax.set_title(f"IoU Distribution ({args.split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fig_dir / "iou_dist.png"), dpi=150)
    plt.close(fig)

    # Dice distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dices, bins=20, edgecolor="black", alpha=0.7, color="orange")
    ax.axvline(np.mean(dices), color="red", linestyle="--", label=f"mean={np.mean(dices):.3f}")
    ax.set_xlabel("Pixel Dice")
    ax.set_ylabel("Count")
    ax.set_title(f"Dice Distribution ({args.split})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fig_dir / "dice_dist.png"), dpi=150)
    plt.close(fig)

    # predicted vs true egg count
    pred_counts = [r["pred_count"] for r in results]
    gt_counts = [r["gt_count"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_counts, pred_counts, alpha=0.6)
    lim = max(max(gt_counts), max(pred_counts)) + 1
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5)
    ax.set_xlabel("True egg count (SAM)")
    ax.set_ylabel("Predicted egg count")
    ax.set_title(f"Egg Count: Predicted vs True ({args.split})")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "count_scatter.png"), dpi=150)
    plt.close(fig)

    # --- failure gallery: worst 8 by IoU ---
    sorted_by_iou = sorted(results, key=lambda r: r["iou"])
    n_worst = min(8, len(sorted_by_iou))
    worst = sorted_by_iou[:n_worst]

    fig, axes = plt.subplots(n_worst, 4, figsize=(16, 4 * n_worst))
    if n_worst == 1:
        axes = [axes]
    for idx, r in enumerate(worst):
        img_path = data_root / r["rel_path"]
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pred = load_label_image(pred_dir / f"{r['image']}.png")
        gt = load_label_image(gt_dir / f"{r['image']}.png")
        diff = make_diff_image(bgr, pred, gt)
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

        axes[idx][0].imshow(rgb)
        axes[idx][0].set_title(f"Input: {r['image'][:18]}", fontsize=8)
        axes[idx][1].imshow(gt > 0, cmap="gray")
        axes[idx][1].set_title(f"SAM GT ({r['gt_count']} eggs)", fontsize=8)
        axes[idx][2].imshow(pred > 0, cmap="gray")
        axes[idx][2].set_title(f"Predicted ({r['pred_count']} eggs)", fontsize=8)
        axes[idx][3].imshow(diff_rgb)
        axes[idx][3].set_title(f"Diff (IoU={r['iou']:.3f})", fontsize=8)
        for a in axes[idx]:
            a.axis("off")

    fig.suptitle(f"Worst {n_worst} failures by pixel IoU ({args.split})", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "failure_gallery.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {fig_dir}")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()


"""Evaluation: compare classical segmentation masks against SAM ground truth.

Reports pixel-level IoU, Dice, precision, recall, and instance-level metrics.
Generates plots, failure gallery, success gallery, and random gallery.

Usage:
  python src/evaluate.py --pred out/classical/final/dev/masks --gt out/sam/masks --split dev --data data
  python src/evaluate.py --pred out/classical/final/eval/masks --gt out/sam/masks --split eval --data data
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


def load_label_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.astype(np.uint16)


def pixel_metrics(pred, gt):
    p = pred > 0
    g = gt > 0
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    inter = tp
    union = tp + fp + fn
    iou = inter / union if union > 0 else 1.0
    dice = 2.0 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec}


def instance_metrics(pred, gt, iou_thresh=0.5):
    pred_ids = sorted(set(pred.flat) - {0})
    gt_ids = sorted(set(gt.flat) - {0})

    if not pred_ids and not gt_ids:
        return {
            "det_tp": 0, "det_fp": 0, "det_fn": 0,
            "det_prec": 1.0, "det_rec": 1.0, "det_f1": 1.0,
            "count_error": 0, "pred_count": 0, "gt_count": 0,
            "matched_ious": []
        }

    if not pred_ids:
        return {
            "det_tp": 0, "det_fp": 0, "det_fn": len(gt_ids),
            "det_prec": 0.0, "det_rec": 0.0, "det_f1": 0.0,
            "count_error": -len(gt_ids), "pred_count": 0, "gt_count": len(gt_ids),
            "matched_ious": []
        }

    if not gt_ids:
        return {
            "det_tp": 0, "det_fp": len(pred_ids), "det_fn": 0,
            "det_prec": 0.0, "det_rec": 1.0, "det_f1": 0.0,
            "count_error": len(pred_ids), "pred_count": len(pred_ids), "gt_count": 0,
            "matched_ious": []
        }

    cost = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pi in enumerate(pred_ids):
        pm = pred == pi
        for j, gi in enumerate(gt_ids):
            gm = gt == gi
            inter = int((pm & gm).sum())
            union = int((pm | gm).sum())
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

    return {
        "det_tp": tp, "det_fp": fp, "det_fn": fn,
        "det_prec": prec, "det_rec": rec, "det_f1": f1,
        "count_error": len(pred_ids) - len(gt_ids),
        "pred_count": len(pred_ids), "gt_count": len(gt_ids),
        "matched_ious": matched_ious
    }


def make_diff_image(img_bgr, pred, gt):
    h, w = pred.shape
    diff = np.zeros((h, w, 3), np.uint8)
    p = pred > 0
    g = gt > 0
    diff[p & g] = [0, 180, 0]       # green: true positive
    diff[p & ~g] = [0, 0, 200]      # red: false positive
    diff[~p & g] = [200, 0, 0]      # blue: false negative

    base = (img_bgr * 0.35).astype(np.uint8)
    mask = p | g
    out = base.copy()
    out[mask] = (0.45 * base[mask] + 0.55 * diff[mask]).astype(np.uint8)
    out[~mask] = base[~mask]
    return out


def plot_gallery(cases, data_root, pred_dir, gt_dir, title, out_path):
    n = len(cases)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.8 * n))
    if n == 1:
        axes = [axes]
    for idx, r in enumerate(cases):
        img_path = data_root / r["rel_path"]
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pred = load_label_image(pred_dir / f"{r['image']}.png")
        gt = load_label_image(gt_dir / f"{r['image']}.png")
        diff = make_diff_image(bgr, pred, gt)
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

        axes[idx][0].imshow(rgb)
        axes[idx][0].set_title(f"{r['image'][:24]}", fontsize=9)
        axes[idx][1].imshow(gt > 0, cmap="gray")
        axes[idx][1].set_title(f"SAM GT ({r['gt_count']} egg{'s' if r['gt_count']!=1 else ''})", fontsize=9)
        axes[idx][2].imshow(pred > 0, cmap="gray")
        axes[idx][2].set_title(f"Classical ({r['pred_count']} egg{'s' if r['pred_count']!=1 else ''})", fontsize=9)
        axes[idx][3].imshow(diff_rgb)
        axes[idx][3].set_title(f"Diff: IoU={r['iou']:.3f}, Dice={r['dice']:.3f}", fontsize=9)
        for a in axes[idx]:
            a.axis("off")

    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


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
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_fig_dir = Path("report/figures")
    report_fig_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rel in image_paths:
        img_path = data_root / rel
        stem = img_path.stem

        pred_path = pred_dir / f"{stem}.png"
        gt_path = gt_dir / f"{stem}.png"

        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction for {stem}: {pred_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT for {stem}: {gt_path}")

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

    ious = [r["iou"] for r in results]
    dices = [r["dice"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    det_precs = [r["det_prec"] for r in results]
    det_recs = [r["det_rec"] for r in results]
    det_f1s = [r["det_f1"] for r in results]
    count_errs = [r["count_error"] for r in results]

    print(f"\n=== {args.split.upper()} Set Evaluation (n={len(results)}) ===")
    print(f"Pixel IoU:       mean={np.mean(ious):.4f}  median={np.median(ious):.4f}")
    print(f"Pixel Dice:      mean={np.mean(dices):.4f}  median={np.median(dices):.4f}")
    print(f"Pixel Precision: mean={np.mean(precs):.4f}")
    print(f"Pixel Recall:    mean={np.mean(recs):.4f}")
    print(f"Det Precision:   mean={np.mean(det_precs):.4f}")
    print(f"Det Recall:      mean={np.mean(det_recs):.4f}")
    print(f"Det F1:          mean={np.mean(det_f1s):.4f}")
    print(f"Count Error:     mean={np.mean(count_errs):.2f}  abs mean={np.mean(np.abs(count_errs)):.2f}")

    # 1. Per-image CSVs
    per_image_rows = []
    for r in results:
        per_image_rows.append({
            "image": r["image"],
            "n_gt": r["gt_count"],
            "n_pred": r["pred_count"],
            "matched_eggs": r["det_tp"],
            "iou": f"{r['iou']:.4f}",
            "dice": f"{r['dice']:.4f}",
            "det_prec": f"{r['det_prec']:.4f}",
            "det_rec": f"{r['det_rec']:.4f}",
            "det_f1": f"{r['det_f1']:.4f}",
        })

    fieldnames = ["image", "n_gt", "n_pred", "matched_eggs", "iou", "dice", "det_prec", "det_rec", "det_f1"]
    for csv_dest in [out_dir / "per_image.csv", Path(args.out) / f"per_image_{args.split}.csv"]:
        with open(csv_dest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_image_rows)

    # 2. Summary JSON
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
    (Path(args.out) / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))

    # 3. Plots: IoU distribution with KDE / reference lines
    fig, ax = plt.subplots(figsize=(7, 4.2))
    n_bins = 15
    counts, bins, patches = ax.hist(ious, bins=n_bins, edgecolor="black", color="#4C72B0", alpha=0.75, density=False)
    ax.axvline(np.mean(ious), color="#C44E52", linestyle="--", linewidth=2, label=f"Mean: {np.mean(ious):.3f}")
    ax.axvline(np.median(ious), color="#55A868", linestyle=":", linewidth=2, label=f"Median: {np.median(ious):.3f}")
    ax.set_xlabel("Pixel IoU", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_title(f"IoU Distribution ({args.split.upper()} Split, n={len(results)})", fontsize=12)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    iou_dist_path = fig_dir / "iou_dist.png"
    fig.savefig(str(iou_dist_path), dpi=150)
    fig.savefig(str(report_fig_dir / f"iou_dist_{args.split}.png"), dpi=150)
    plt.close(fig)

    # 4. Plots: Predicted vs True count scatter with jitter and y=x line
    np.random.seed(42)
    gt_arr = np.array([r["gt_count"] for r in results], dtype=float)
    pred_arr = np.array([r["pred_count"] for r in results], dtype=float)
    jitter_gt = gt_arr + np.random.normal(0, 0.06, size=len(gt_arr))
    jitter_pred = pred_arr + np.random.normal(0, 0.06, size=len(pred_arr))

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(jitter_gt, jitter_pred, alpha=0.65, color="#2B5C8F", s=50, edgecolors="white", linewidths=0.5)
    max_c = max(int(max(gt_arr)), int(max(pred_arr)), 3) + 1
    ax.plot([0, max_c], [0, max_c], "r--", alpha=0.6, label="Ideal (y = x)")
    ax.set_xlabel("Ground Truth Egg Count (SAM)", fontsize=11)
    ax.set_ylabel("Predicted Egg Count (Classical)", fontsize=11)
    ax.set_title(f"Predicted vs True Egg Count ({args.split.upper()})", fontsize=12)
    ax.set_xlim(-0.3, max_c)
    ax.set_ylim(-0.3, max_c)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    scatter_path = fig_dir / "count_scatter.png"
    fig.savefig(str(scatter_path), dpi=150)
    fig.savefig(str(report_fig_dir / f"count_scatter_{args.split}.png"), dpi=150)
    plt.close(fig)

    # 5. Galleries: Worst-8, Best-8, Random-8
    sorted_by_iou = sorted(results, key=lambda r: r["iou"])
    worst_8 = sorted_by_iou[:8]
    best_8 = sorted_by_iou[-8:][::-1]

    rng = np.random.RandomState(42)
    rand_indices = rng.choice(len(results), size=min(8, len(results)), replace=False)
    rand_8 = [results[i] for i in rand_indices]

    plot_gallery(worst_8, data_root, pred_dir, gt_dir,
                 f"Worst 8 Failure Cases by Pixel IoU ({args.split.upper()} Split)",
                 fig_dir / "failure_gallery.png")
    shutil.copy(fig_dir / "failure_gallery.png", report_fig_dir / f"failure_gallery_{args.split}.png")

    plot_gallery(best_8, data_root, pred_dir, gt_dir,
                 f"Best 8 Success Cases by Pixel IoU ({args.split.upper()} Split)",
                 fig_dir / "success_gallery.png")
    shutil.copy(fig_dir / "success_gallery.png", report_fig_dir / f"success_gallery_{args.split}.png")

    plot_gallery(rand_8, data_root, pred_dir, gt_dir,
                 f"Random 8 Representative Cases ({args.split.upper()} Split)",
                 fig_dir / "random_gallery.png")
    shutil.copy(fig_dir / "random_gallery.png", report_fig_dir / f"random_gallery_{args.split}.png")

    print(f"\nArtifacts saved for {args.split.upper()}:")
    print(f"  CSVs:    {out_dir / 'per_image.csv'} and out/eval/per_image_{args.split}.csv")
    print(f"  Summary: {out_dir / 'summary.json'}")
    print(f"  Figures: {fig_dir} and report/figures/")


if __name__ == "__main__":
    main()

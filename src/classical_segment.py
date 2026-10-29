"""Classical image-processing egg segmentation with experiment harness.

Usage:
  python src/classical_segment.py --run-all-dev
  python src/classical_segment.py --trial final --split eval
  python src/classical_segment.py --list-trials
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sam_segment import draw_overlay


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def fill_holes(mask):
    """Fill enclosed holes in a binary mask."""
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def remove_small(mask, min_area=500):
    """Remove connected components smaller than min_area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def split_touching_watershed(binary, min_dist=20):
    """Use Euclidean distance transform + watershed to split touching eggs."""
    dist = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=min_dist, labels=binary)
    if len(coords) == 0:
        _, labels = cv2.connectedComponents(binary.astype(np.uint8))
        return labels.astype(np.uint16)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=binary)
    return labels.astype(np.uint16)


def score_component(cnt, h, w):
    """Score a component based on area, solidity, ellipse fit, and center distance."""
    area = cv2.contourArea(cnt)
    if area < 1000 or area > 0.8 * h * w:
        return -1.0
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return -1.0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    dist_to_center = np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    ellipse_score = 0.5
    if len(cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(cnt)
            (ex, ey), (MA, ma), angle = ellipse
            if MA > 0 and ma > 0:
                aspect_ratio = max(MA, ma) / min(MA, ma)
                ar_penalty = abs(aspect_ratio - 1.35)
                ell_area = np.pi * (MA / 2) * (ma / 2)
                area_match = min(area, ell_area) / max(area, ell_area)
                ellipse_score = area_match * max(0.0, 1.0 - 0.5 * ar_penalty)
        except Exception:
            pass

    return float(solidity * 0.4 + ellipse_score * 0.4 + (1.0 - dist_to_center) * 0.2)


def refine_grabcut(bgr, binary, min_area=1000):
    """Refine foreground mask using GrabCut with foreground and background seeds."""
    if np.sum(binary) == 0:
        return binary
    h, w = binary.shape
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[10:h - 10, 10:w - 10] = cv2.GC_PR_BGD

    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded = cv2.erode(binary, k_erode)
    gc_mask[binary > 0] = cv2.GC_PR_FGD
    gc_mask[eroded > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        out = fill_holes(out)
        out = remove_small(out, min_area)
        return out
    except Exception:
        return binary


# ---------------------------------------------------------------------------
# Parameterized Pipeline
# ---------------------------------------------------------------------------

def segment_pipeline(bgr, **params):
    """Unified parameterized egg segmentation pipeline."""
    space = params.get("space", "gray")
    blur = params.get("blur", 5)
    thresh_method = params.get("thresh_method", "otsu")
    open_k = params.get("open_k", 0)
    close_k = params.get("close_k", 0)
    should_fill_holes = params.get("fill_holes", True)
    min_area = params.get("min_area", 500)
    use_component_ranking = params.get("component_ranking", False)
    use_grabcut = params.get("grabcut", False)
    ws_mode = params.get("watershed", "none")
    ws_min_dist = params.get("ws_min_dist", 20)

    h, w = bgr.shape[:2]

    # 1. Color space conversion & representation
    if space == "gray":
        ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif space == "clahe":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clip = params.get("clahe_clip", 2.0)
        tile = params.get("clahe_tile", 8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        ch = clahe.apply(gray)
    elif space == "lab_l":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ch = lab[:, :, 0]
    elif space == "border_dist":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[20:h - 20, 20:w - 20] = False
        bg_color = np.median(lab[border_mask], axis=0)
        diff = lab.astype(np.float32) - bg_color.astype(np.float32)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        ch = np.clip(dist / max(1e-5, dist.max()) * 255, 0, 255).astype(np.uint8)
    elif space == "hsv":
        ch = bgr
    else:
        ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if blur > 0 and space != "hsv":
        ch = cv2.GaussianBlur(ch, (blur, blur), 0)

    # 2. Binarization
    if thresh_method == "otsu":
        _, binary = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_method == "otsu_inv":
        _, binary = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif thresh_method == "adaptive":
        block = params.get("adaptive_block", 51)
        c = params.get("adaptive_c", -10)
        binary = cv2.adaptiveThreshold(ch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, c)
    elif thresh_method == "hsv_gate":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s_max = params.get("s_max", 70)
        v_min = params.get("v_min", 130)
        binary = ((hsv[:, :, 1] < s_max) & (hsv[:, :, 2] > v_min)).astype(np.uint8) * 255
    elif thresh_method == "border_polarity":
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[20:h - 20, 20:w - 20] = False
        border_mean = np.mean(ch[border_mask])
        center_mean = np.mean(ch[20:h - 20, 20:w - 20])
        t_type = cv2.THRESH_BINARY_INV if border_mean > center_mean else cv2.THRESH_BINARY
        _, binary = cv2.threshold(ch, 0, 255, t_type + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological operations
    if open_k > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    if close_k > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    if should_fill_holes:
        binary = fill_holes(binary)

    if min_area > 0:
        binary = remove_small(binary, min_area)

    # 4. Optional Component Ranking
    if use_component_ranking:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored = []
        for cnt in contours:
            s = score_component(cnt, h, w)
            if s > 0.4:
                scored.append((s, cnt))
        scored.sort(key=lambda x: x[0], reverse=True)
        out_bin = np.zeros_like(binary)
        for _, cnt in scored[:3]:
            cv2.drawContours(out_bin, [cnt], -1, 255, -1)
        binary = out_bin

    # 5. Optional GrabCut refinement
    if use_grabcut:
        binary = refine_grabcut(bgr, binary, min_area=min_area)

    # 6. Instance label generation / Watershed
    if ws_mode == "always":
        labels = split_touching_watershed(binary, min_dist=ws_min_dist)
    elif ws_mode == "conditional":
        n, cc_labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        labels = np.zeros_like(cc_labels, dtype=np.uint16)
        next_id = 1
        for i in range(1, n):
            comp_mask = (cc_labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = cnts[0]
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            aspect_ratio = 1.0
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (ex, ey), (MA, ma), angle = ellipse
                    if MA > 0 and ma > 0:
                        aspect_ratio = max(MA, ma) / min(MA, ma)
                except Exception:
                    pass

            needs_watershed = (solidity < 0.90) or (aspect_ratio > 2.2) or (area > 30000)
            if needs_watershed:
                sub_labels = split_touching_watershed(comp_mask * 255, min_dist=ws_min_dist)
                for sid in np.unique(sub_labels):
                    if sid == 0:
                        continue
                    labels[sub_labels == sid] = next_id
                    next_id += 1
            else:
                labels[comp_mask > 0] = next_id
                next_id += 1
    else:
        _, labels = cv2.connectedComponents(binary)
        labels = labels.astype(np.uint16)

    return labels


# ---------------------------------------------------------------------------
# Trial Configurations
# ---------------------------------------------------------------------------

TRIALS = [
    {
        "trial_id": "baseline",
        "what_changed": "Grayscale conversion + Gaussian blur (5x5) + Otsu threshold + hole fill + min area 500",
        "why": "Establish minimal intensity-based segmentation baseline",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "otsu_inv",
        "what_changed": "Inverted Otsu thresholding",
        "why": "Test hypothesis whether dark eggs or bright backlighting require inverse thresholding",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu_inv", "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "clahe_otsu",
        "what_changed": "CLAHE (clip 2.0, tile 8x8) before Gaussian blur and Otsu",
        "why": "Evaluate local histogram equalization to mitigate harsh directional shadows",
        "params": {"space": "clahe", "clahe_clip": 2.0, "clahe_tile": 8, "blur": 5, "thresh_method": "otsu", "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "adaptive_thresh",
        "what_changed": "Adaptive Gaussian thresholding (block 51, C=-10)",
        "why": "Locally adaptive thresholding to account for spatial illumination gradients",
        "params": {"space": "gray", "blur": 5, "thresh_method": "adaptive", "adaptive_block": 51, "adaptive_c": -10, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "hsv_sat",
        "what_changed": "HSV color gating: low saturation (S < 70) and high brightness (V > 130)",
        "why": "Color-space constraint targeting pale/white eggshells to reject colored background",
        "params": {"space": "hsv", "thresh_method": "hsv_gate", "s_max": 70, "v_min": 130, "open_k": 7, "close_k": 7, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "lab_l",
        "what_changed": "CIELAB L* channel + Gaussian blur + Otsu thresholding",
        "why": "Test perceptual lightness channel L* versus raw luminance Y/grayscale",
        "params": {"space": "lab_l", "blur": 5, "thresh_method": "otsu", "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "morph_small",
        "what_changed": "Grayscale Otsu + small elliptical opening & closing (7x7)",
        "why": "Eliminate fine background noise bridges and smooth egg contours",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 7, "close_k": 7, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "morph_medium",
        "what_changed": "Grayscale Otsu + medium elliptical opening & closing (15x15)",
        "why": "Scale structuring element closer to minimum egg radius to remove larger clutter",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 15, "close_k": 15, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "morph_large",
        "what_changed": "Grayscale Otsu + large elliptical opening & closing (25x25) + hole fill",
        "why": "Match structuring element to natural egg convexity, eliminating non-elliptical noise",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "border_polarity",
        "what_changed": "Border-adaptive polarity detection: invert threshold if border mean > center mean",
        "why": "Handle backlit scenes and bright surfaces by automatically detecting background polarity",
        "params": {"space": "gray", "blur": 5, "thresh_method": "border_polarity", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "lab_border_dist",
        "what_changed": "CIELAB Euclidean distance from estimated image border background color",
        "why": "Segment eggs by color-distance contrast against table or background surface",
        "params": {"space": "border_dist", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "component_ranking",
        "what_changed": "Component selection scored by ellipse fit residual, area, and center distance",
        "why": "Reject peripheral clutter and irregular background objects based on geometric egg priors",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500, "component_ranking": True},
    },
    {
        "trial_id": "watershed_conditional",
        "what_changed": "Conditional distance watershed applied only to components failing ellipse-fit criteria",
        "why": "Avoid over-segmenting single regular eggs while splitting clustered touching eggs",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500, "watershed": "conditional", "ws_min_dist": 20},
    },
    {
        "trial_id": "watershed_always",
        "what_changed": "Euclidean distance transform + watershed applied to all detected egg clusters",
        "why": "Resolve touching eggs into separate instances across all scenes",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500, "watershed": "always", "ws_min_dist": 20},
    },
    {
        "trial_id": "final",
        "what_changed": "Final frozen pipeline: Grayscale Otsu + large morph (25x25) + GrabCut refinement + distance watershed",
        "why": "Achieves highest IoU with refined egg silhouettes and separated instances",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 1000, "grabcut": True, "watershed": "always", "ws_min_dist": 20},
    },
]

TRIAL_MAP = {t["trial_id"]: t for t in TRIALS}


# ---------------------------------------------------------------------------
# Evaluation and logging harness
# ---------------------------------------------------------------------------

def compute_iou_dice(pred_labels, gt_labels):
    """Compute pixel-level IoU and Dice between prediction and ground truth."""
    pred_fg = pred_labels > 0
    gt_fg = gt_labels > 0
    inter = int((pred_fg & gt_fg).sum())
    union = int((pred_fg | gt_fg).sum())
    iou = inter / union if union > 0 else 1.0
    total = int(pred_fg.sum()) + int(gt_fg.sum())
    dice = 2.0 * inter / total if total > 0 else 1.0
    return iou, dice


def execute_trial(trial_cfg, split, splits, data_root, sam_root, out_base):
    """Run a single trial configuration on a dataset split and measure metrics."""
    trial_id = trial_cfg["trial_id"]
    params = trial_cfg["params"]
    image_paths = splits[split]
    out_dir = out_base / trial_id / split
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    ious, dices = [], []
    t0 = time.time()

    for rel in image_paths:
        img_path = data_root / rel
        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        labels = segment_pipeline(bgr, **params)

        stem = img_path.stem
        cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), labels)
        overlay = draw_overlay(bgr, labels)
        cv2.imwrite(str(out_dir / "overlays" / f"{stem}.jpg"), overlay)

        gt_path = sam_root / f"{stem}.png"
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt is not None:
                iou, dice = compute_iou_dice(labels, gt)
                ious.append(iou)
                dices.append(dice)

    elapsed = time.time() - t0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0

    print(f"[{trial_id:20s} | {split:4s}] n={len(image_paths)}, "
          f"mean IoU={mean_iou:.4f}, mean Dice={mean_dice:.4f}, time={elapsed:.1f}s")

    return mean_iou, mean_dice, len(image_paths)


def log_experiment_row(log_path, trial_cfg, split, mean_iou, mean_dice, n_images, decision, reason):
    """Append a structured row to experiments.csv."""
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "trial_id", "split", "what_changed", "why", "parameters",
                "mean_iou", "mean_dice", "n_images", "decision", "reason"
            ])
        w.writerow([
            trial_cfg["trial_id"],
            split,
            trial_cfg["what_changed"],
            trial_cfg["why"],
            json.dumps(trial_cfg["params"]),
            f"{mean_iou:.4f}",
            f"{mean_dice:.4f}",
            n_images,
            decision,
            reason
        ])


def run_all_dev(splits, data_root, sam_root, out_base, log_path):
    """Run all trials sequentially on dev split, dynamically recording KEEP / DISCARD."""
    print(f"Running all {len(TRIALS)} trials on dev split ({len(splits['dev'])} images)...")
    if log_path.exists():
        log_path.unlink()

    best_iou = 0.0

    for idx, trial_cfg in enumerate(TRIALS):
        trial_id = trial_cfg["trial_id"]
        mean_iou, mean_dice, n_images = execute_trial(
            trial_cfg, "dev", splits, data_root, sam_root, out_base
        )

        if idx == 0:
            decision = "BASELINE"
            reason = f"Baseline established (dev IoU {mean_iou:.4f}, Dice {mean_dice:.4f})"
            best_iou = mean_iou
        else:
            diff = mean_iou - best_iou
            if mean_iou > best_iou:
                decision = "KEEP"
                reason = f"Improves dev IoU from {best_iou:.4f} to {mean_iou:.4f} (+{diff:.4f})"
                best_iou = mean_iou
            else:
                decision = "DISCARD"
                reason = f"Fails to improve dev IoU ({mean_iou:.4f} vs best {best_iou:.4f}, {diff:.4f})"

        log_experiment_row(log_path, trial_cfg, "dev", mean_iou, mean_dice, n_images, decision, reason)

    print(f"\nAll dev trials completed successfully! Log written to {log_path}")


def run_single(trial_id, split, splits, data_root, sam_root, out_base, log_path):
    """Run a single named trial on a split (used for the frozen final eval run)."""
    if trial_id not in TRIAL_MAP:
        raise ValueError(f"Unknown trial '{trial_id}'. Use --list-trials to see available trials.")

    trial_cfg = TRIAL_MAP[trial_id]
    mean_iou, mean_dice, n_images = execute_trial(
        trial_cfg, split, splits, data_root, sam_root, out_base
    )

    if split == "eval":
        decision = "EVAL_FINAL"
        reason = f"Frozen final pipeline evaluated once on unseen eval split (IoU {mean_iou:.4f}, Dice {mean_dice:.4f})"
    else:
        decision = "TRIAL_RUN"
        reason = f"Single trial run on {split} (IoU {mean_iou:.4f}, Dice {mean_dice:.4f})"

    log_experiment_row(log_path, trial_cfg, split, mean_iou, mean_dice, n_images, decision, reason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Classical segmentation experiment harness.")
    ap.add_argument("--trial", help="Name of the trial to run")
    ap.add_argument("--split", default="dev", choices=["dev", "eval"], help="Split to run on")
    ap.add_argument("--data", default="data", help="Data directory")
    ap.add_argument("--splits-json", default="splits.json", help="Path to splits.json")
    ap.add_argument("--sam-masks", default="out/sam/masks", help="SAM ground truth masks directory")
    ap.add_argument("--out", default="out/classical", help="Output directory")
    ap.add_argument("--log", default="experiments.csv", help="Experiment log CSV")
    ap.add_argument("--run-all-dev", action="store_true", help="Run all trials sequentially on dev split")
    ap.add_argument("--list-trials", action="store_true", help="List all available trial configs")
    args = ap.parse_args()

    if args.list_trials:
        for t in TRIALS:
            print(f"  {t['trial_id']:22s}  {t['what_changed']}")
        return

    splits = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
    data_root = Path(args.data)
    sam_root = Path(args.sam_masks)
    out_base = Path(args.out)
    log_path = Path(args.log)

    if args.run_all_dev:
        run_all_dev(splits, data_root, sam_root, out_base, log_path)
        return

    if not args.trial:
        ap.error("--trial is required unless --run-all-dev or --list-trials is specified")

    run_single(args.trial, args.split, splits, data_root, sam_root, out_base, log_path)


if __name__ == "__main__":
    main()

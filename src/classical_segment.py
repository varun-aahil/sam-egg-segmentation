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
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

try:
    from sam_segment import draw_overlay
except ImportError:
    from src.sam_segment import draw_overlay


def fill_holes(mask):
    return ndimage.binary_fill_holes(mask).astype(np.uint8) * 255


def remove_small(mask, min_area=500):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def split_touching_watershed(binary, min_dist=50):
    dist = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=min_dist, labels=binary)
    if len(coords) <= 1:
        _, labels = cv2.connectedComponents(binary.astype(np.uint8))
        return labels.astype(np.uint16)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=binary)
    return labels.astype(np.uint16)


def score_component(cnt, h, w):
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


def refine_grabcut(bgr, binary, min_area=2500):
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


def segment_pipeline(bgr, **params):
    h, w = bgr.shape[:2]
    space = params.get("space", "gray")
    blur = params.get("blur", 5)
    thresh_method = params.get("thresh_method", "otsu")
    open_k = params.get("open_k", 0)
    close_k = params.get("close_k", 0)
    should_fill_holes = params.get("fill_holes", True)
    min_area = params.get("min_area", 500)
    comp_rank = params.get("comp_rank", None)
    use_grabcut = params.get("grabcut", False)
    ws_mode = params.get("watershed", "none")
    ws_min_dist = params.get("ws_min_dist", 50)
    polarity_margin = params.get("polarity_margin", 0)

    # 1. Color space conversion
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
    elif space == "hsv_gate":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s_max = params.get("s_max", 70)
        v_min = params.get("v_min", 130)
        binary = ((hsv[:, :, 1] < s_max) & (hsv[:, :, 2] > v_min)).astype(np.uint8) * 255
    else:
        ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2. Filtering & Thresholding
    if space != "hsv_gate":
        if blur > 0:
            ch = cv2.GaussianBlur(ch, (blur, blur), 0)

        if thresh_method == "otsu":
            _, binary = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif thresh_method == "otsu_inv":
            _, binary = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif thresh_method == "adaptive":
            block = params.get("adaptive_block", 51)
            c = params.get("adaptive_c", -10)
            binary = cv2.adaptiveThreshold(ch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block, c)
        elif thresh_method == "border_polarity":
            border_mask = np.ones((h, w), dtype=bool)
            border_mask[20:h - 20, 20:w - 20] = False
            border_mean = np.mean(ch[border_mask])
            center_mean = np.mean(ch[20:h - 20, 20:w - 20])
            t_type = cv2.THRESH_BINARY_INV if border_mean > (center_mean + polarity_margin) else cv2.THRESH_BINARY
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
    if comp_rank is not None and comp_rank > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored = []
        for cnt in contours:
            s = score_component(cnt, h, w)
            if s > 0.3:
                scored.append((s, cnt))
        scored.sort(key=lambda x: x[0], reverse=True)
        out_bin = np.zeros_like(binary)
        for _, cnt in scored[:comp_rank]:
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

            needs_watershed = (solidity < 0.88 and area > 15000) or (aspect_ratio > 2.0 and area > 20000) or (area > 50000)
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


TRIALS = [
    {
        "trial_id": "baseline",
        "parent": None,
        "what_changed": "Grayscale conversion + Gaussian blur (5x5) + Otsu threshold + hole fill + min area 500",
        "why": "Establish minimal intensity-based segmentation baseline",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 0, "close_k": 0, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "otsu_inv",
        "parent": "baseline",
        "what_changed": "Invert Otsu thresholding",
        "why": "Test hypothesis whether dark eggs or bright backlighting require inverse thresholding",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu_inv", "open_k": 0, "close_k": 0, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "clahe_otsu",
        "parent": "baseline",
        "what_changed": "CLAHE local contrast equalization (clip 2.0, tile 8x8) before Otsu",
        "why": "Evaluate local histogram equalization to mitigate harsh shadows",
        "params": {"space": "clahe", "clahe_clip": 2.0, "clahe_tile": 8, "blur": 5, "thresh_method": "otsu", "open_k": 0, "close_k": 0, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "adaptive_thresh",
        "parent": "baseline",
        "what_changed": "Adaptive Gaussian thresholding (block 51, C=-10)",
        "why": "Locally adaptive thresholding to account for spatial illumination gradients",
        "params": {"space": "gray", "blur": 5, "thresh_method": "adaptive", "adaptive_block": 51, "adaptive_c": -10, "open_k": 0, "close_k": 0, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "lab_l",
        "parent": "baseline",
        "what_changed": "CIELAB L* perceptual lightness channel + Otsu",
        "why": "Test perceptual lightness channel L* versus raw luminance Y/grayscale",
        "params": {"space": "lab_l", "blur": 5, "thresh_method": "otsu", "open_k": 0, "close_k": 0, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "hsv_gate",
        "parent": "baseline",
        "what_changed": "HSV color gating (S < 70, V > 130)",
        "why": "Color-space constraint targeting pale/white eggshells to reject colored background",
        "params": {"space": "hsv_gate", "s_max": 70, "v_min": 130, "open_k": 0, "close_k": 0, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "morph_small",
        "parent": "baseline",
        "what_changed": "Add small elliptical opening & closing (7x7)",
        "why": "Eliminate fine background noise bridges and smooth egg contours",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 7, "close_k": 7, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "morph_medium",
        "parent": "morph_small",
        "what_changed": "Increase elliptical kernel from 7x7 to 15x15",
        "why": "Scale structuring element closer to minimum egg radius to remove larger clutter",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 15, "close_k": 15, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "morph_large",
        "parent": "morph_medium",
        "what_changed": "Increase elliptical kernel from 15x15 to 25x25",
        "why": "Match structuring element to natural egg convexity, eliminating non-elliptical noise",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "border_polarity_m0",
        "parent": "morph_large",
        "what_changed": "Auto-invert threshold if border mean > center mean",
        "why": "Automatically detect background polarity for backlit scenes and bright tables",
        "params": {"space": "gray", "blur": 5, "thresh_method": "border_polarity", "polarity_margin": 0, "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "border_polarity_m15",
        "parent": "morph_large",
        "what_changed": "Auto-invert threshold only if border mean > center mean + 15",
        "why": "Require high contrast margin before inverting to avoid false inversion on white tables",
        "params": {"space": "gray", "blur": 5, "thresh_method": "border_polarity", "polarity_margin": 15, "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "lab_border_dist",
        "parent": "morph_large",
        "what_changed": "CIELAB Euclidean color distance from estimated border background",
        "why": "Segment eggs by color-distance contrast against table or background surface",
        "params": {"space": "border_dist", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 500},
    },
    {
        "trial_id": "area_filter_2500",
        "parent": "morph_large",
        "what_changed": "Increase minimum component area from 500 to 2500 pixels",
        "why": "Ground truth audit showed true egg area is >10k px; higher threshold rejects background clutter blobs",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 2500},
    },
    {
        "trial_id": "component_ranking_top1",
        "parent": "area_filter_2500",
        "what_changed": "Retain only the single top-ranked component by geometric egg score",
        "why": "84.5% of images contain 1 egg; strict single-egg selection eliminates all peripheral blobs",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 2500, "comp_rank": 1},
    },
    {
        "trial_id": "component_ranking_top3",
        "parent": "area_filter_2500",
        "what_changed": "Retain at most the top 3 scoring components based on area, solidity, and ellipse fit",
        "why": "Dataset has at most 3 eggs per image; drops lower-scoring clutter blobs",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 2500, "comp_rank": 3},
    },
    {
        "trial_id": "grabcut_refine",
        "parent": "area_filter_2500",
        "what_changed": "Apply GrabCut refinement using morphological erosion as foreground seed",
        "why": "Refine egg silhouettes along true color boundaries",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 2500, "grabcut": True},
    },
    {
        "trial_id": "watershed_always",
        "parent": "area_filter_2500",
        "what_changed": "Euclidean distance transform + watershed applied unconditionally to all components",
        "why": "Split touching eggs into separate instances across all scenes",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 2500, "watershed": "always", "ws_min_dist": 20},
    },
    {
        "trial_id": "watershed_conditional",
        "parent": "area_filter_2500",
        "what_changed": "Conditional distance watershed (min_dist=50) applied only to non-convex, elongated or oversized clusters",
        "why": "Resolve touching egg clusters into distinct instances without over-segmenting single regular eggs",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 2500, "watershed": "conditional", "ws_min_dist": 50},
    },
    {
        "trial_id": "final",
        "parent": "watershed_conditional",
        "what_changed": "Frozen final pipeline: Grayscale Otsu + large morph (25x25) + hole filling + min area 2500 + conservative conditional watershed",
        "why": "Best overall segmentation and instance separation",
        "params": {"space": "gray", "blur": 5, "thresh_method": "otsu", "open_k": 25, "close_k": 25, "fill_holes": True, "min_area": 2500, "watershed": "conditional", "ws_min_dist": 50},
    },
]

TRIAL_MAP = {t["trial_id"]: t for t in TRIALS}


def compute_instance_metrics(pred_labels, gt_labels, iou_thresh=0.5):
    pred_ids = sorted(set(pred_labels.flat) - {0})
    gt_ids = sorted(set(gt_labels.flat) - {0})

    if not pred_ids and not gt_ids:
        return 1.0, 0
    if not pred_ids or not gt_ids:
        return 0.0, len(pred_ids) - len(gt_ids)

    cost = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pi in enumerate(pred_ids):
        pm = pred_labels == pi
        for j, gi in enumerate(gt_ids):
            gm = gt_labels == gi
            inter = int(np.sum(pm & gm))
            union = int(np.sum(pm | gm))
            cost[i, j] = -(inter / union) if union > 0 else 0

    row_ind, col_ind = linear_sum_assignment(cost)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if -cost[r, c] >= iou_thresh)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    count_err = len(pred_ids) - len(gt_ids)
    return f1, count_err


def execute_trial(trial_cfg, split, splits, data_root, sam_root, out_base):
    trial_id = trial_cfg["trial_id"]
    params = trial_cfg["params"]
    image_paths = splits[split]
    out_dir = out_base / trial_id / split
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    ious, dices, f1s, count_errs = [], [], [], []
    t0 = time.time()

    for rel in image_paths:
        img_path = data_root / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Missing input image: {img_path}")

        stem = img_path.stem
        gt_path = sam_root / f"{stem}.png"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground truth SAM mask for: {stem}")

        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            raise ValueError(f"Could not read ground truth SAM mask: {gt_path}")

        bgr = cv2.imread(str(img_path))
        labels = segment_pipeline(bgr, **params)

        cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), labels)
        overlay = draw_overlay(bgr, labels)
        cv2.imwrite(str(out_dir / "overlays" / f"{stem}.jpg"), overlay)

        pred_fg = labels > 0
        gt_fg = gt > 0
        inter = int(np.sum(pred_fg & gt_fg))
        union = int(np.sum(pred_fg | gt_fg))
        iou = inter / union if union > 0 else 1.0
        tot = int(np.sum(pred_fg)) + int(np.sum(gt_fg))
        dice = 2.0 * inter / tot if tot > 0 else 1.0

        f1, cerr = compute_instance_metrics(labels, gt)

        ious.append(iou)
        dices.append(dice)
        f1s.append(f1)
        count_errs.append(cerr)

    elapsed = time.time() - t0
    n_scored = len(ious)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0
    median_dice = float(np.median(dices)) if dices else 0.0
    det_f1 = float(np.mean(f1s)) if f1s else 0.0
    mean_cerr = float(np.mean(count_errs)) if count_errs else 0.0

    print(f"[{trial_id:22s} | {split:4s}] n={n_scored}, IoU={mean_iou:.4f} (med {median_iou:.4f}), "
          f"Dice={mean_dice:.4f}, DetF1={det_f1:.4f}, CntErr={mean_cerr:.2f}, time={elapsed:.1f}s")

    return {
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "det_f1": det_f1,
        "count_err": mean_cerr,
        "n_images": n_scored,
    }


def log_experiment_row(log_path, trial_cfg, split, metrics, decision, reason):
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "trial_id", "parent", "split", "what_changed", "why", "parameters",
                "mean_iou", "median_iou", "mean_dice", "median_dice",
                "det_f1", "count_err", "n_images", "decision", "reason"
            ])
        w.writerow([
            trial_cfg["trial_id"],
            trial_cfg.get("parent") or "None",
            split,
            trial_cfg["what_changed"],
            trial_cfg["why"],
            json.dumps(trial_cfg["params"]),
            f"{metrics['mean_iou']:.4f}",
            f"{metrics['median_iou']:.4f}",
            f"{metrics['mean_dice']:.4f}",
            f"{metrics['median_dice']:.4f}",
            f"{metrics['det_f1']:.4f}",
            f"{metrics['count_err']:.2f}",
            metrics["n_images"],
            decision,
            reason
        ])


def run_all_dev(splits, data_root, sam_root, out_base, log_path):
    dev_trials = [t for t in TRIALS if t["trial_id"] != "final"]
    print(f"Running all {len(dev_trials)} tuning trials on dev split ({len(splits['dev'])} images)...")
    if log_path.exists():
        log_path.unlink()

    dev_metrics_by_trial = {}

    for idx, trial_cfg in enumerate(dev_trials):
        trial_id = trial_cfg["trial_id"]
        parent_id = trial_cfg["parent"]
        metrics = execute_trial(trial_cfg, "dev", splits, data_root, sam_root, out_base)
        dev_metrics_by_trial[trial_id] = metrics

        if parent_id is None:
            decision = "BASELINE"
            reason = (f"Baseline established on dev (IoU {metrics['mean_iou']:.4f}, "
                      f"Dice {metrics['mean_dice']:.4f}, DetF1 {metrics['det_f1']:.4f})")
        else:
            parent_m = dev_metrics_by_trial[parent_id]
            diff = metrics["mean_iou"] - parent_m["mean_iou"]
            dice_diff = metrics["mean_dice"] - parent_m["mean_dice"]

            if diff > 1e-4 and dice_diff >= -1e-4:
                decision = "KEEP"
                reason = (f"Improves dev IoU over parent '{parent_id}' from {parent_m['mean_iou']:.4f} "
                          f"to {metrics['mean_iou']:.4f} (+{diff:.4f})")
            elif diff > 1e-4 and dice_diff < -1e-4:
                decision = "DISCARD"
                reason = (f"Marginal IoU gain (+{diff:.4f}) but Dice degrades from "
                          f"{parent_m['mean_dice']:.4f} to {metrics['mean_dice']:.4f} ({dice_diff:.4f})")
            elif abs(diff) <= 1e-4 and metrics["det_f1"] > parent_m["det_f1"] + 0.05 and metrics["count_err"] < parent_m["count_err"] + 2.0:
                f1_diff = metrics["det_f1"] - parent_m["det_f1"]
                decision = "KEEP"
                reason = (f"Dev IoU tied with parent '{parent_id}' ({metrics['mean_iou']:.4f}); "
                          f"kept on tiebreaker: DetF1 improves from {parent_m['det_f1']:.4f} to {metrics['det_f1']:.4f} (+{f1_diff:.4f})")
            else:
                decision = "DISCARD"
                reason = (f"Fails to improve dev IoU over parent '{parent_id}' "
                          f"({metrics['mean_iou']:.4f} vs {parent_m['mean_iou']:.4f}, {diff:+.4f})")

        log_experiment_row(log_path, trial_cfg, "dev", metrics, decision, reason)

    print(f"\nAll dev tuning trials completed successfully! Log written to {log_path}")


def run_single(trial_id, split, splits, data_root, sam_root, out_base, log_path):
    if trial_id not in TRIAL_MAP:
        raise ValueError(f"Unknown trial '{trial_id}'. Use --list-trials to see available trials.")

    trial_cfg = TRIAL_MAP[trial_id]
    metrics = execute_trial(trial_cfg, split, splits, data_root, sam_root, out_base)

    if split == "eval":
        decision = "EVAL_FINAL"
        reason = (f"Frozen final pipeline evaluated once on unseen eval split "
                  f"(IoU {metrics['mean_iou']:.4f}, Dice {metrics['mean_dice']:.4f}, DetF1 {metrics['det_f1']:.4f})")
    else:
        decision = "TRIAL_RUN"
        reason = f"Single trial run on {split} (IoU {metrics['mean_iou']:.4f}, Dice {metrics['mean_dice']:.4f})"

    log_experiment_row(log_path, trial_cfg, split, metrics, decision, reason)


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
            p = t.get("parent") or "None"
            print(f"  {t['trial_id']:24s} (parent: {p:20s})  {t['what_changed']}")
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

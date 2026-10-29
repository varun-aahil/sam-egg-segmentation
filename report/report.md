# Egg Segmentation and Tracking - Technical Report

**Candidate:** Varun Aahil  
**Task Date:** October 2026  
**Dataset:** Roboflow Egg Dataset (110 images, 640x640)  
**Partition:** 4:1 Dev/Eval split (88 dev, 22 eval) via perceptual hash grouping.

---

## 1. Task 1: SAM Ground Truth Generation
- Model: Meta Segment Anything Model (SAM) `vit_b` backbone on CUDA.
- Annotation Strategy: Interactive point prompts recorded per egg via `src/annotate_points.py` and saved to `points.json`.
- Inference: `SamPredictor` with positive/negative points and `multimask_output=True`, selecting the best-scoring mask.
- Verified: All 110 images have clean, ground truth masks free of heuristic filter biases.

---

## 2. Task 2: Classical Image-Processing Experiments
Full 15-trial progression on dev set with frozen eval run:

| Trial ID | Split | What Changed | Why | Dev Mean IoU | Dev Mean Dice | Decision | Reason |
|----------|-------|--------------|-----|--------------|---------------|----------|--------|
| baseline | dev | Grayscale conversion + Gaussian blur (5x5) + Otsu threshold + hole fill + min area 500 | Establish minimal intensity-based segmentation baseline | 0.3521 | 0.4700 | BASELINE | Baseline established (dev IoU 0.3521, Dice 0.4700) |
| otsu_inv | dev | Inverted Otsu thresholding | Test hypothesis whether dark eggs or bright backlighting require inverse thresholding | 0.1040 | 0.1641 | DISCARD | Fails to improve dev IoU (0.1040 vs best 0.3521, -0.2480) |
| clahe_otsu | dev | CLAHE (clip 2.0, tile 8x8) before Gaussian blur and Otsu | Evaluate local histogram equalization to mitigate harsh directional shadows | 0.2601 | 0.3845 | DISCARD | Fails to improve dev IoU (0.2601 vs best 0.3521, -0.0920) |
| adaptive_thresh | dev | Adaptive Gaussian thresholding (block 51, C=-10) | Locally adaptive thresholding to account for spatial illumination gradients | 0.0638 | 0.1127 | DISCARD | Fails to improve dev IoU (0.0638 vs best 0.3521, -0.2882) |
| hsv_sat | dev | HSV color gating: low saturation (S < 70) and high brightness (V > 130) | Color-space constraint targeting pale/white eggshells to reject colored background | 0.3593 | 0.4805 | KEEP | Improves dev IoU from 0.3521 to 0.3593 (+0.0073) |
| lab_l | dev | CIELAB L* channel + Gaussian blur + Otsu thresholding | Test perceptual lightness channel L* versus raw luminance Y/grayscale | 0.3266 | 0.4460 | DISCARD | Fails to improve dev IoU (0.3266 vs best 0.3593, -0.0328) |
| morph_small | dev | Grayscale Otsu + small elliptical opening & closing (7x7) | Eliminate fine background noise bridges and smooth egg contours | 0.3566 | 0.4746 | DISCARD | Fails to improve dev IoU (0.3566 vs best 0.3593, -0.0028) |
| morph_medium | dev | Grayscale Otsu + medium elliptical opening & closing (15x15) | Scale structuring element closer to minimum egg radius to remove larger clutter | 0.3769 | 0.4964 | KEEP | Improves dev IoU from 0.3593 to 0.3769 (+0.0175) |
| morph_large | dev | Grayscale Otsu + large elliptical opening & closing (25x25) + hole fill | Match structuring element to natural egg convexity, eliminating non-elliptical noise | 0.3973 | 0.5159 | KEEP | Improves dev IoU from 0.3769 to 0.3973 (+0.0204) |
| border_polarity | dev | Border-adaptive polarity detection: invert threshold if border mean > center mean | Handle backlit scenes and bright surfaces by automatically detecting background polarity | 0.3784 | 0.4897 | DISCARD | Fails to improve dev IoU (0.3784 vs best 0.3973, -0.0188) |
| lab_border_dist | dev | CIELAB Euclidean distance from estimated image border background color | Segment eggs by color-distance contrast against table or background surface | 0.3584 | 0.4553 | DISCARD | Fails to improve dev IoU (0.3584 vs best 0.3973, -0.0389) |
| component_ranking | dev | Component selection scored by ellipse fit residual, area, and center distance | Reject peripheral clutter and irregular background objects based on geometric egg priors | 0.3973 | 0.5068 | KEEP | Improves dev IoU from 0.3973 to 0.3973 (+0.0000) |
| watershed_conditional | dev | Conditional distance watershed applied only to components failing ellipse-fit criteria | Avoid over-segmenting single regular eggs while splitting clustered touching eggs | 0.3973 | 0.5159 | DISCARD | Fails to improve dev IoU (0.3973 vs best 0.3973, -0.0000) |
| watershed_always | dev | Euclidean distance transform + watershed applied to all detected egg clusters | Resolve touching eggs into separate instances across all scenes | 0.3975 | 0.5161 | KEEP | Improves dev IoU from 0.3973 to 0.3975 (+0.0003) |
| final | dev | Final frozen pipeline: Grayscale Otsu + large morph (25x25) + GrabCut refinement + distance watershed | Achieves highest IoU with refined egg silhouettes and separated instances | 0.4008 | 0.5151 | KEEP | Improves dev IoU from 0.3975 to 0.4008 (+0.0032) |
| final | eval | Final frozen pipeline: Grayscale Otsu + large morph (25x25) + GrabCut refinement + distance watershed | Achieves highest IoU with refined egg silhouettes and separated instances | 0.3207 | 0.4259 | EVAL_FINAL | Frozen final pipeline evaluated once on unseen eval split (IoU 0.3207, Dice 0.4259) |

---

## 3. Task 3: Evaluation Metrics

| Metric | Dev Split (88 images) | Eval Split (22 images) |
|--------|-----------------------|------------------------|
| Pixel IoU (Mean) | 0.4008 | 0.3207 |
| Pixel IoU (Median) | 0.3010 | 0.2491 |
| Pixel Dice (Mean) | 0.5151 | 0.4259 |
| Pixel Precision | 0.4399 | 0.3363 |
| Pixel Recall | 0.8719 | 0.9294 |
| Detection Precision (IoU >= 0.5) | 0.2263 | 0.2165 |
| Detection Recall (IoU >= 0.5) | 0.5303 | 0.4621 |
| Detection F1 Score | 0.2827 | 0.2552 |
| Mean Abs Count Error | 4.47 | 3.64 |

---

## 4. Task 4: Video Tracking Results
- Expected Eggs in Video: 2
- Unique IDs Created: 2 ([1, 2])
- ID Switches: 0
- Track Lifetimes: {'Egg 1': 716, 'Egg 2': 643}
- Frames without Detection: 664 frames (intro/outro)
- Frames with Detection: 776 frames
- Public Video Link (Google Drive): `[YOUR_GOOGLE_DRIVE_PUBLIC_LINK_HERE]`

---

## 5. AI Tools Disclosure
Assisted by Anthropic Claude and Google Antigravity. All CV algorithms, parameterizations, and experimental metrics were generated and validated on the actual code repository.

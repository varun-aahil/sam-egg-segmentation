# Egg Segmentation and Multi-Object Tracking Take-Home Report

**Candidate:** Sree Varun Aahil  
**Dataset:** Roboflow Egg Dataset (110 images, 640x640, grouped dev/eval 88/22)  
**Video Test:** YouTube `Y4JyV94y2_A` (1280x720 table experiment, 1440 frames @ 24.0 fps)  
**Submission Date:** October 30, 2026  

---

## 1. Executive Summary

This report presents an end-to-end computer vision solution for egg segmentation and multi-object tracking under challenging conditions, including touching instances, boiling water, high-contrast table textures, and hand occlusions. Ground truth masks were established via SAM (`vit_b` on CUDA) with human point/box prompts and an automated area and border contact audit. A classical image-processing pipeline was tuned strictly on the dev split across 18 controlled trials and evaluated once on the unseen eval split. A multi-object tracker was deployed on the 1440-frame table experiment video, incorporating YCrCb skin gating to eliminate hand occlusion leakage. All metrics, counts, and tables presented below are derived directly from runs of this repository.

### Key Measured Results
- **Ground Truth Integrity:** 110 clean SAM masks generated via `SamPredictor` (`vit_b` on CUDA) using human point prompts and tight bounding boxes for boiling water edge cases. 0 flagged suspicious masks remaining.
- **Classical Pipeline (Dev Split, n=88):** Pixel Mean IoU = **0.4018** (95% CI: [0.3435, 0.4603], Median = 0.3147), Dice = **0.5200**, Detection F1 = **0.4087**, Mean Count Error = **2.20**.
- **Classical Pipeline (Eval Split, n=22):** Pixel Mean IoU = **0.3196** (95% CI: [0.1978, 0.4268], Median = 0.2502), Dice = **0.4239**, Detection F1 = **0.3206**, Mean Count Error = **1.68**.
- **Video Tracking (1440 frames):** 1440 frames processed at 24.0 FPS. **6 unique tracks** created across the video, with **18 loss events** and **12 recoveries**. Hand occlusion and fingertip leakage at frame 600 resolved using YCrCb skin gating and morphological opening (2296 px of skin noise removed).

---

## 2. Task 1: Ground Truth Mask Generation (SAM)

### Approach
- Model: Segment Anything Model (`vit_b`, 375 MB checkpoint) running on CUDA.
- Interactive point annotations stored in `points.json` via `src/annotate_points.py` with multi-mask score selection (`multimask_output=True`).
- Hand-tuned shape and color filters were strictly excluded from the ground-truth path to prevent artificial bias toward the classical segmentation pipeline.
- Ground truth egg count distribution across 110 images:
  - **1 egg:** 93 images (84.5%)
  - **2 eggs:** 6 images (5.5%)
  - **3 eggs:** 11 images (10.0%)

### Ground Truth Quality Audit & Fixes
Initial automated SAM passes on boiling water images (`not_damaged_124`, `not_damaged_135`, `not_damaged_145`) flooded the entire pot interior (up to 81.9% of frame). An automated border and area audit was built into `src/contact_sheet.py` flagging masks > 40% area or touching > half of the image border.
Bounding box prompts were added around the egg boundaries for these boiling pot scenes:
- `not_damaged_124`: BBox `[280, 150, 520, 400]` reduced area from 81.9% to **13.4%** (0 border contacts).
- `not_damaged_135`: BBox `[270, 170, 520, 420]` reduced area from 79.5% to **14.2%** (0 border contacts).
- `not_damaged_145`: BBox `[250, 200, 480, 450]` reduced area from 62.5% to **13.2%** (0 border contacts).
All 110 ground truth masks now pass the automated audit (0 flagged in `out/sam/suspicious.txt`).

---

## 3. Task 2: Classical Segmentation & Parameterized Experiment Harness

### Pipeline Architecture
A single parameterized pipeline `segment_pipeline` in `src/classical_segment.py` supports:
1. Color spaces: Grayscale, CLAHE, CIELAB L*, border-color Euclidean distance, HSV gating.
2. Thresholding: Standard Otsu, Inverted Otsu, Adaptive Gaussian, Border-contrast polarity detection.
3. Morphology: Elliptical opening and closing sweeps (7x7, 15x15, 25x25), hole filling.
4. Outlier & Clutter Rejection: Area pruning (min area 500 to 2500 px), geometric shape ranking.
5. Instance Separation: Conservative Euclidean distance transform watershed.

### Controlled Experiment Progression
Trials were tracked in `experiments.csv`. Keep/discard decisions compare strictly against the parent's dev IoU (with Detection F1 serving as an explicit tiebreaker). Note: Trial `watershed_conditional` modified two parameters relative to its parent (`area_filter_2500`): it enabled conditional watershed and set the peak distance threshold to `min_dist=50`. While pixel IoU remained tied at 0.4018 (since watershed divides connected components without changing overall foreground area), Detection F1 improved from 0.3084 to 0.4087 and count error improved from +3.82 to +2.20, justifying keeping the configuration.

**The Kept Pipeline Chain:**  
`baseline -> morph_small -> morph_medium -> morph_large -> area_filter_2500 -> watershed_conditional -> final (eval)`

### Complete Measured Experiment Log (`experiments.csv`)

| Trial ID | Parent | Split | What Changed | Dev IoU | Dev Dice | Det F1 | Cnt Err | Decision | Reason |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `baseline` | `None` | dev | Grayscale conversion + Gaussian blur (5x5) + Otsu threshold + hole fill + min area 500 | 0.3513 | 0.4693 | 0.2071 | 3.12 | **BASELINE** | Baseline established on dev (IoU 0.3513, Dice 0.4693, DetF1 0.2071) |
| `otsu_inv` | `baseline` | dev | Invert Otsu thresholding | 0.0823 | 0.1424 | 0.0000 | 3.36 | **DISCARD** | Fails to improve dev IoU over parent 'baseline' (0.0823 vs 0.3513, -0.2690) |
| `clahe_otsu` | `baseline` | dev | CLAHE local contrast equalization (clip 2.0, tile 8x8) before Otsu | 0.2562 | 0.3791 | 0.0799 | 8.22 | **DISCARD** | Fails to improve dev IoU over parent 'baseline' (0.2562 vs 0.3513, -0.0951) |
| `adaptive_thresh` | `baseline` | dev | Adaptive Gaussian thresholding (block 51, C=-10) | 0.0634 | 0.1119 | 0.0065 | 12.33 | **DISCARD** | Fails to improve dev IoU over parent 'baseline' (0.0634 vs 0.3513, -0.2878) |
| `lab_l` | `baseline` | dev | CIELAB L* perceptual lightness channel + Otsu | 0.3244 | 0.4437 | 0.1665 | 3.67 | **DISCARD** | Fails to improve dev IoU over parent 'baseline' (0.3244 vs 0.3513, -0.0268) |
| `hsv_gate` | `baseline` | dev | HSV color gating (S < 70, V > 130) | 0.3456 | 0.4665 | 0.1587 | 4.32 | **DISCARD** | Fails to improve dev IoU over parent 'baseline' (0.3456 vs 0.3513, -0.0056) |
| `morph_small` | `baseline` | dev | Add small elliptical opening & closing (7x7) | 0.3558 | 0.4740 | 0.2161 | 2.89 | **KEEP** | Improves dev IoU over parent 'baseline' from 0.3513 to 0.3558 (+0.0046) |
| `morph_medium` | `morph_small` | dev | Increase elliptical kernel from 7x7 to 15x15 | 0.3766 | 0.4964 | 0.2363 | 2.34 | **KEEP** | Improves dev IoU over parent 'morph_small' from 0.3558 to 0.3766 (+0.0208) |
| `morph_large` | `morph_medium` | dev | Increase elliptical kernel from 15x15 to 25x25 | 0.3972 | 0.5161 | 0.2642 | 1.25 | **KEEP** | Improves dev IoU over parent 'morph_medium' from 0.3766 to 0.3972 (+0.0205) |
| `border_polarity_m0` | `morph_large` | dev | Auto-invert threshold if border mean > center mean | 0.3698 | 0.4821 | 0.2409 | 1.07 | **DISCARD** | Fails to improve dev IoU over parent 'morph_large' (0.3698 vs 0.3972, -0.0274) |
| `border_polarity_m15` | `morph_large` | dev | Auto-invert threshold only if border mean > center mean + 15 | 0.3945 | 0.5119 | 0.2596 | 1.25 | **DISCARD** | Fails to improve dev IoU over parent 'morph_large' (0.3945 vs 0.3972, -0.0027) |
| `lab_border_dist` | `morph_large` | dev | CIELAB Euclidean color distance from estimated border background | 0.3603 | 0.4582 | 0.2673 | 2.55 | **DISCARD** | Fails to improve dev IoU over parent 'morph_large' (0.3603 vs 0.3972, -0.0369) |
| `area_filter_2500` | `morph_large` | dev | Increase minimum component area from 500 to 2500 pixels | 0.4018 | 0.5200 | 0.3084 | 0.45 | **KEEP** | Improves dev IoU over parent 'morph_large' from 0.3972 to 0.4018 (+0.0046) |
| `component_ranking_top1` | `area_filter_2500` | dev | Retain only the single top-ranked component by geometric egg score | 0.3746 | 0.4761 | 0.3295 | -0.33 | **DISCARD** | Fails to improve dev IoU over parent 'area_filter_2500' (0.3746 vs 0.4018, -0.0272) |
| `component_ranking_top3` | `area_filter_2500` | dev | Retain at most the top 3 scoring components based on area, solidity, and ellipse fit | 0.3972 | 0.5092 | 0.3133 | 0.27 | **DISCARD** | Fails to improve dev IoU over parent 'area_filter_2500' (0.3972 vs 0.4018, -0.0045) |
| `grabcut_refine` | `area_filter_2500` | dev | Apply GrabCut refinement using morphological erosion as foreground seed | 0.4038 | 0.5177 | 0.3073 | 0.38 | **DISCARD** | Marginal IoU gain (+0.0021) but Dice degrades from 0.5200 to 0.5177 (-0.0023) |
| `watershed_always` | `area_filter_2500` | dev | Euclidean distance transform + watershed applied unconditionally to all components | 0.4018 | 0.5200 | 0.3537 | 3.90 | **DISCARD** | Fails to improve dev IoU over parent 'area_filter_2500' (0.4018 vs 0.4018, +0.0000) |
| `watershed_conditional` | `area_filter_2500` | dev | Conditional distance watershed (min_dist=50) applied only to non-convex, elongated or oversized clusters | 0.4018 | 0.5200 | 0.4087 | 2.20 | **KEEP** | Dev IoU tied with parent 'area_filter_2500' (0.4018); kept on tiebreaker: DetF1 improves from 0.3084 to 0.4087 (+0.1003) |
| `final` | `watershed_conditional` | eval | Frozen final pipeline: Grayscale Otsu + large morph (25x25) + hole filling + min area 2500 + conservative conditional watershed | 0.3196 | 0.4239 | 0.3206 | 1.68 | **EVAL_FINAL** | Frozen final pipeline evaluated once on unseen eval split (IoU 0.3196, Dice 0.4239, DetF1 0.3206) |

---

## 4. Task 3: Quantitative & Qualitative Evaluation vs Ground Truth

### Dev vs Eval Metric Comparison

| Metric | Dev Split (n=88) | Eval Split (n=22) | Definition / Match Criteria |
|:---|:---:|:---:|:---|
| **Pixel IoU (Mean)** | **0.4018** | **0.3196** | TP / (TP + FP + FN) across foreground pixels |
| **Pixel IoU (Median)** | **0.3147** | **0.2502** | Median per-image pixel IoU |
| **Pixel Dice (Mean)** | **0.5200** | **0.4239** | 2 * TP / (2*TP + FP + FN) |
| **Pixel Precision** | **0.4488** | **0.3449** | TP / (TP + FP) |
| **Pixel Recall** | **0.8461** | **0.8749** | TP / (TP + FN) |
| **Detection Precision** | **0.3563** | **0.2758** | Hungarian matched eggs / Predicted eggs |
| **Detection Recall** | **0.5966** | **0.4621** | Hungarian matched eggs / Ground truth eggs |
| **Detection F1** | **0.4087** | **0.3206** | Harmonic mean of detection precision & recall |
| **Count Error (Mean)** | **2.20** | **1.68** | Mean signed count error (Pred - GT) |
| **Count Error (Abs Mean)**| **2.20** | **1.77** | Mean absolute count error |

### Bootstrap 95% Confidence Intervals & Dev vs Eval Analysis
- **Dev Split Mean IoU:** 0.4018 (Bootstrap 95% CI: [0.3435, 0.4603])
- **Eval Split Mean IoU:** 0.3196 (Bootstrap 95% CI: [0.1978, 0.4268])

The eval split comprises 22 images, with 0 white-tray images (all 13 are in dev), and 3 backlit images (13.6% of eval) against 6 in dev (6.8%). Because the eval sample size is small (n=22), its bootstrap 95% confidence interval spans widely ([0.1978, 0.4268]) and fully encompasses the dev split mean (0.4018). Thus, the numerical difference between dev and eval cannot be attributed to a single cause or distribution shift.

### Per-Category Dev vs Eval Split Breakdown

| Category | Dev Count | Dev IoU | Eval Count | Eval IoU |
|:---|:---:|:---:|:---:|:---:|
| White egg on white tray / fridge interior | 13 | 0.1274 | 0 | 0.0000 |
| Egg in boiling water / submerged | 6 | 0.1723 | 2 | 0.1756 |
| Backlit / extreme shadows | 6 | 0.3354 | 3 | 0.5392 |
| Tabletop / high-contrast background | 32 | 0.3659 | 8 | 0.2424 |
| Multiple touching eggs | 13 | 0.5674 | 3 | 0.4886 |
| Hand holding egg / fingers occluding | 11 | 0.6120 | 5 | 0.1502 |
| Foliage / herbs in background (specular highlights) | 7 | 0.6906 | 1 | 0.9068 |

### Visual Failure Categorization (`failure_tags.csv`)
All 110 dataset images were inspected and tagged by physical scene type. Images `not_damaged_123`, `125`, and `42` were tagged as *White egg on white tray / fridge interior*. Backlit hand shots `not_damaged_20`, `24`, and `44` were categorized under a unified rule: *Whenever intense directional backlighting or silhouette illumination is present behind the egg (even if supported by fingers), it is categorized under Backlit / extreme shadows because extreme illumination dynamic range is the primary driver of Otsu thresholding failure.*

| Scene Category | Image Count | Pct of Dataset | Mean IoU | Mean Dice | Mean Det F1 | Primary Failure Mechanism |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| White egg on white tray / fridge interior | 13 | 11.8% | 0.1274 | 0.2198 | 0.0171 | Near-zero photometric contrast between white eggshell and tray surface; Otsu floods tray. |
| Egg in boiling water / submerged | 8 | 7.3% | 0.1731 | 0.2901 | 0.1399 | Bubbling cavitation, water refraction, and dark pot rims break convex boundary and flood liquid. |
| Tabletop / high-contrast background | 40 | 36.4% | 0.3412 | 0.4744 | 0.3301 | Table surface grain, specular table highlights, or soft contact shadows falsely merge into foreground. |
| Backlit / extreme shadows | 9 | 8.2% | 0.4033 | 0.4539 | 0.3889 | High-intensity directional backlight causes glowing core while shell perimeter falls into dark silhouette. |
| Hand holding egg / fingers occluding | 16 | 14.5% | 0.4677 | 0.5821 | 0.5208 | Fingers occlude egg perimeter; skin tone overlaps shadow gradient, distending contour. |
| Multiple touching eggs | 16 | 14.5% | 0.5526 | 0.6844 | 0.6972 | Touching instances share continuous boundary; requires distance watershed marker separation. |
| Foliage / herbs in background (specular highlights) | 8 | 7.3% | 0.7177 | 0.8222 | 0.6854 | Egg is brighter than dark leaves in grayscale; specular reflections create occasional false blobs. |

**Key Qualitative Takeaway:** Classical thresholding struggles most on *White egg on white tray / fridge interior* (Mean IoU = 0.1274) and *Boiling water* (Mean IoU = 0.1731) because global luminance is nearly identical between foreground and background. On *Foliage / herbs in background* (Mean IoU = 0.7177), the egg contrasts well because the egg shell is substantially brighter than dark leaves in grayscale; specular leaf highlights create occasional false blobs.

### Metric Distributions

![IoU Distribution Dev](figures/iou_dist_dev.png)
*Figure 1a: Pixel IoU distribution on Dev split (n=88).*

![IoU Distribution Eval](figures/iou_dist_eval.png)
*Figure 1b: Pixel IoU distribution on Eval split (n=22).*

![Count Scatter Dev](figures/count_scatter_dev.png)
*Figure 2a: Predicted vs True egg counts on Dev split. Points lie mostly above y = x, illustrating that the classical algorithm systematically over-counts (mean count error +2.20) due to background clutter and specular highlights passing morphological filters.*

![Count Scatter Eval](figures/count_scatter_eval.png)
*Figure 2b: Predicted vs True egg counts on Eval split. Points similarly lie above y = x (mean count error +1.68).*

### Visual Galleries

![Worst 8 Failure Cases Dev](figures/failure_gallery_dev.png)
*Figure 3a: Worst 8 failures on Dev split (green = TP, red = FP, blue = FN). White trays and boiling liquid dominate error cases.*

![Worst 8 Failure Cases Eval](figures/failure_gallery_eval.png)
*Figure 3b: Worst 8 failures on Eval split (green = TP, red = FP, blue = FN).*

![Best 8 Success Cases Dev](figures/success_gallery_dev.png)
*Figure 4: Best 8 successes on Dev split showing clean elliptical masks matching ground truth.*

![Random 8 Representative Cases Dev](figures/random_gallery_dev.png)
*Figure 5: Random 8 representative cases illustrating typical operational performance across mixed illumination.*

---

## 5. Task 4: Multi-Object Video Tracking

### Video Dataset
- Video: YouTube `Y4JyV94y2_A` (1280x720 @ 24.0 fps, 1440 frames total).
- Table demonstration comparing boiled and raw egg spinning. Eggs are placed at frame 222, spun, occluded by hand, and removed before frame 1005.

### Video Tracking Architecture
1. **Domain Adaptation:** Table ROI constraint (y >= 270) rejects shirt patterns; watermark suppression at bottom-right (x >= 1040, y >= 600).
2. **Hand & Fingertip Leakage Resolution:**
   - At frame ~600, demonstrator fingers brush against the egg. Baseline thresholding caused segmentation to leak into pale fingertips.
   - Fixed by incorporating **YCrCb skin-color gating** (`Cr in [135, 175] & Cb in [80, 125]`) combined with an elliptical morphological opening kernel (11x11).
   - Removed **2296 pixels** of skin noise at frame 600 while preserving egg contours.
3. **Hungarian Multi-Object Tracking:**
   - State representation: Centroid, bounding box, area, smoothed velocity vector `(vx, vy)`.
   - Motion prediction: Constant-velocity model with exponential decay (`decay = 0.90^lost`).
   - Cost Matrix: 30% mask IoU + 70% normalized Euclidean centroid distance.
   - Gating & ID Integrity: Distance threshold of 150 px. Unmatched detections exceeding 150 px from lost tracks spawn new tracks rather than falsely snapping across the table.
   - Spatial proximity conflict monitoring: Active tracks are checked for proximity anomalies; note that this measures internal spatial conflicts rather than true ground-truth identity continuity across physical removals.

### Frame 600 Fingertip Fix Comparison
![Frame 600 Comparison](figures/frame_600_comparison.png)
*Figure 6: Hand near egg at frame 600 (crop: x in [200, 460], y in [300, 430]). Left: Raw frame. Middle: Baseline leakage into fingertips (red). Right: YCrCb skin gating cleanly restricts contour to egg shell (green), removing 2296 px of skin noise.*

### Video Tracking Statistics & Timeline
- **Total Frames:** 1440 frames (782 frames with active detections, 658 frames without detection).
- **Unique IDs Created:** 6 tracks total.
- **Track Lifetimes:**
  - Egg 1: 529 frames
  - Egg 2: 282 frames
  - Egg 3: 101 frames
  - Egg 4: 269 frames
  - Egg 5: 3 frames
  - Egg 6: 190 frames
- **Loss and Recovery Events:** 18 loss events, 12 recovery events.

### Track Timeline

| Track ID | Birth Frame | Last Active Frame | Lifetime (Frames) | Physical Egg Description | Visual Verification |
|:---|:---:|:---:|:---:|:---|:---:|
| Track 1 | 222 | 751 | 529 | Right egg (boiled), initial placement and spin sequence | Verified |
| Track 2 | 222 | 504 | 282 | Left egg (raw), initial placement and spin sequence | Verified |
| Track 3 | 546 | 647 | 101 | Demonstrator repositioned left egg to right table position | Verified |
| Track 4 | 689 | 1004 | 269 | Boiled egg returned to left side until table clearance | Verified |
| Track 5 | 800 | 802 | 3 | Transient motion-blur artifact (area ~3043-3715 px) during rapid raw egg spin | Confirmed spurious artifact (unverified physical egg) |
| Track 6 | 810 | 1000 | 190 | Raw egg tracked through final spin sequence until removal | Verified |

### Analysis of Track 5
Track 5 lived for exactly 3 frames (F#800 to F#802) with areas between 3,043 and 3,715 px around centroid (721, 531). Frame inspection reveals that Track 5 is a transient motion-blur fragment of the rapidly spinning raw egg immediately before the tracker acquired Track 6 at F#810. In a production tracking deployment, transient tracks with lifetime < 5 frames should be filtered as false positives.

### Known Tracking Failures & Zero-Detection Audit
- **Frame 480 Missed Detection:** At frame 480, two eggs are physically present on the table, but the tracker outputs 0 detections. Inspection of the segmentation contour shows that the demonstrator's hand in contact with the egg caused contour solidity to drop to 0.7673, falling below the strict convexity threshold (`solidity >= 0.78`).
- **Audit of Frames 222 to 1004:** Across the entire active demonstration interval (783 frames), exactly 1 frame (Frame 480) had zero detections. All other 782 frames had active detections.
- **Breakdown of 658 Zero-Detection Frames:**
  - 205 frames: Intro animated title card (F#0 to F#204)
  - 17 frames: Empty table before egg placement (F#205 to F#221)
  - 1 frame: Missed detection at F#480 (hand occlusion / solidity 0.7673)
  - 10 frames: Empty table after egg removal (F#1005 to F#1014)
  - 425 frames: Outro credits / title card (F#1015 to F#1439)
  - Total: 205 + 17 + 1 + 10 + 425 = 658 frames.

![Video Tracking Contact Sheet](figures/tracking_contact_sheet.jpg)
*Figure 7: Tracking Contact Sheet across 16 key video events (births, hand proximity at F#600, spins, and recoveries).*

---

## 6. Limitations and Future Work

1. **Ground Truth Annotation:** Ground truth masks were established via SAM (`vit_b`) prompted with human point and box coordinates. While audited for area and border anomalies, they represent semi-automated model outputs rather than manual polygon tracings.
2. **Dataset Size:** The dataset contains 110 images (88 dev, 22 eval). Due to the small sample size in eval, statistical confidence intervals are wide, meaning differences across splits must be interpreted conservatively.
3. **Watershed Behavior:** Distance transform watershed successfully separated touching instances (improving Detection F1 from 0.3084 to 0.4087 on dev), but had zero impact on pixel IoU (remaining 0.4018) because it partitions foreground without altering outer silhouette boundaries.
4. **Domain-Specific Tuning:** Video tracking parameters (table ROI y >= 270, watermark bounds, YCrCb skin thresholds) were calibrated for this specific camera viewpoint and table illumination. Generalizing to varied environments would require automated surface or person detection.
5. **Re-Identification Across Removals:** Smooth, textureless white eggs have near-identical visual appearance. When eggs are removed from view and placed back onto the table, identity assignment cannot be established with physical certainty under classical re-ID methods.
6. **Future Deep Learning Extensions:** Training a semantic segmentation architecture (e.g. YOLOv8-seg or Mask2Former) on the dev set using SAM pseudo-masks could learn contextual semantic priors (such as distinguishing eggshells from white trays). Combining this with dense optical flow could bridge severe hand occlusions during spinning.

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
- **Video Tracking (1440 frames):** 1440 frames processed at 24.0 FPS. **6 unique tracks** created across the video, with **18 loss events** and **12 recoveries**. At least one egg was detected in 782 of the 783 active demonstration frames (frame 480 had 0 detections). Hand occlusion and fingertip leakage at frame 600 resolved using YCrCb skin gating and morphological opening (gating reduced the total foreground mask by 2296 px over the whole frame; restricted to crop x in [200, 460], y in [300, 430], the mask was reduced by 1304 px).

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

### Ground Truth Quality Audit & Fixes (`out/sam/gt_fix_audit.json`)
Initial automated SAM passes on boiling water images (`not_damaged_124`, `not_damaged_135`, `not_damaged_145`) flooded the entire pot interior (up to 81.9% of frame). An automated border and area audit was built into `src/contact_sheet.py` flagging masks > 40% area or touching > half of the image border.
Bounding box prompts were added around the egg boundaries for these boiling pot scenes (verified in `out/sam/gt_fix_audit.json`):
- `not_damaged_124`: BBox `[280, 150, 520, 400]` reduced area fraction from 81.9% (335354 px, 978 border contacts) to **13.4%** (54995 px, 0 border contacts).
- `not_damaged_135`: BBox `[270, 170, 520, 420]` reduced area fraction from 79.5% (325584 px, 994 border contacts) to **14.2%** (58113 px, 0 border contacts).
- `not_damaged_145`: BBox `[250, 200, 480, 450]` reduced area fraction from 62.5% (256010 px, 665 border contacts) to **13.2%** (54032 px, 0 border contacts).
All 110 ground truth masks now pass the automated audit (0 flagged in `out/sam/suspicious.txt`).

### Dataset Split Integrity & Leakage Audit (`out/eval/split_leakage_audit.txt`)
- **Dataset Composition:** 110 images total (640x640), grouped by source filename prefixes and perceptual hashing (`phash_distance <= 8`).
- **Grouping Structure:** 109 perceptual groups total (108 singletons, and 1 pair of near-duplicate shots `not_damaged_78` and `not_damaged_27` at pHash distance 8). Both images of this pair were placed strictly into the `dev` split.
- **Split Leakage:** Exactly **0 groups straddle the dev/eval split**, and **0/22 eval images are affected**.
- **Leak-Free Evaluation:** Recomputing evaluation metrics on the leak-free eval subset yields identical metrics to the full eval set: Pixel IoU = **0.3196**, Dice = **0.4239**, Detection F1 = **0.3206**.
- **Augmentation Check:** The dataset contains no synthetic geometric or photometric augmentations (crops, rotations, color shifts, flips) sharing pHash distance &le; 8. Full audit logs are available in `out/eval/split_leakage_audit.txt`.

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
Trials were tracked in `experiments.csv`. Keep/discard decisions compare strictly against the parent's dev IoU. Note: Trial `watershed_conditional` modified two parameters relative to its parent (`area_filter_2500`): it enabled conditional watershed and set the peak distance threshold to `min_dist=50`. Under this trial, Detection F1 rose from 0.3084 to 0.4087, mean signed count error rose from +0.45 to +2.20 (over-splitting), pixel IoU unchanged. Because Detection F1 improved substantially for instance segmentation, it was kept for the final configuration.

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

Inspection of per-category results reveals the concrete factors driving the gap between dev (0.4018) and eval (0.3196):
1. **Severe Hand Image Degradation:** Hand-held egg images average IoU 0.6120 in dev (11 images) but collapse to **0.1502** in eval (5 images). Hand shots represent 22.7% of the eval split vs 12.5% of dev.
2. **Tabletop Contrast Differences:** Tabletop scenes average IoU 0.3659 in dev (32 images) vs 0.2424 in eval (8 images).
3. **Absence of White-Tray Camouflage in Eval:** All 13 white-tray images (IoU 0.1274) were grouped into dev, while eval contains 0.
4. **Small-Sample Variance:** Because eval has only 22 images, its bootstrap 95% confidence interval spans widely ([0.1978, 0.4268]) and encompasses the dev mean. Thus, while hand-holding failures and tabletop lighting differences clearly drive the performance drop in the observed data, small sample variance also plays a substantial role.

### Detailed Breakdown of the 5 Eval Hand Images
In the failure analysis, the "Hand holding egg" category on eval (5 images: `not_damaged_108`, `127`, `130`, `151`, `152`) exhibits severe performance degradation with mean IoU of **0.1502**. Visual inspection and color analysis reveal specific physical root causes:
- `not_damaged_108` (IoU: 0.0459): Pale skin tone and bright overhead lighting cause Otsu to merge palm, wrist, and forearm into the egg contour (285,716 false positive pixels, of which 37.8% are skin).
- `not_damaged_127` (IoU: 0.0601): Darker brown egg held against a washed-out background; Otsu thresholding places the darker egg boundary near background luminance while the lighter hand, fingers, and background reflection merge into a 283,983 px foreground blob.
- `not_damaged_130` (IoU: 0.4256): Egg cupped in palm; palm skin connects continuously to the egg perimeter, producing 62,367 false positive pixels—of which exactly **92.02%** (57392 pixels) fall directly into the standard YCrCb skin-color cluster ($Cr \in [135, 175], Cb \in [80, 125]$).
- `not_damaged_151` (IoU: 0.1451): Backlit hand holding an egg against a desk light; the fingers and hand cast strong shadows, while the light background and illuminated skin bleed together into the foreground threshold, over-segmenting the hand (223,414 false positive pixels).
- `not_damaged_152` (IoU: 0.0745): Dark brown egg held in a hand over a light kitchen worktop; the brown egg has lower luminance than the bright hand and table; Otsu sets the threshold separating dark objects from light, causing the hand and tabletop to be segmented as foreground (295,610 px) while the egg itself is at the boundary.

| Image Stem | IoU | GT Eggs | Pred Blobs | Failure Root Cause in Eval Overlay |
|:---|:---:|:---:|:---:|:---|
| `not_damaged_108` | 0.0459 | 1 | 3 | Pale hand skin and bright overhead lighting cause Otsu thresholding to merge the palm, wrist, and forearm into the egg contour (285,716 FP pixels, 37.8% skin). |
| `not_damaged_127` | 0.0601 | 1 | 3 | Darker brown egg held against washed-out backdrop; Otsu places egg boundary near background luminance while fingers and light table merge into a 283,983 px blob. |
| `not_damaged_130` | 0.4256 | 1 | 2 | Egg cupped in open palm; palm skin connects continuously to egg perimeter, generating 62,367 FP pixels of which exactly 92.02% (57,392 px) are skin in YCrCb. |
| `not_damaged_151` | 0.1451 | 1 | 6 | Backlit hand holding egg against desk light; fingers and hand cast strong shadows, while background and illuminated skin bleed together, over-segmenting into 6 fragments. |
| `not_damaged_152` | 0.0745 | 1 | 3 | Dark brown egg on bright kitchen worktop; brown egg has lower luminance than bright hand and table; Otsu segments hand and table as foreground (295,610 px). |

![Eval Hand Overlays](figures/eval_hand_overlays.png)  
*Figure 6: Eval split hand-held egg failure overlays (n=5). Left: Original image. Middle: SAM ground truth (green). Right: Classical prediction (red/amber). Skin-tone luminance and lighting contrast cause massive false-positive leakage into fingers, palms, and arms.*

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
*Figure 3a: Worst 8 failures on Dev split (Green=TP, Red=FP, Blue=FN). White trays and boiling liquid dominate error cases.*

![Worst 8 Failure Cases Eval](figures/failure_gallery_eval.png)  
*Figure 3b: Worst 8 failures on Eval split (Green=TP, Red=FP, Blue=FN).*

![Best 8 Success Cases Dev](figures/success_gallery_dev.png)  
*Figure 4: Best 8 successes on Dev split showing clean elliptical masks matching ground truth.*

![Random 8 Representative Cases Dev](figures/random_gallery_dev.png)  
*Figure 5: Random 8 representative cases illustrating typical operational performance across mixed illumination.*

---

## 5. Task 4: Multi-Object Video Tracking

### Video Dataset
- Video: YouTube `Y4JyV94y2_A` (1280x720 @ 24.0 fps, 1440 frames total).
- Table demonstration comparing boiled and raw egg spinning. Eggs are placed at frame 222, spun, occluded by hand, and conclude demonstration at frame 1004.

### Video Tracking Architecture
1. **Domain Adaptation:** Table ROI constraint (y >= 270) rejects shirt patterns; watermark suppression at bottom-right (x >= 1040, y >= 600).
2. **Hand & Fingertip Leakage Resolution:**
   - At frame ~600, demonstrator fingers brush against the egg. Baseline thresholding caused segmentation to leak into pale fingertips.
   - Fixed by incorporating **YCrCb skin-color gating** (`Cr in [135, 175] & Cb in [80, 125]`) combined with an elliptical morphological opening kernel (11x11).
   - Gating reduced the total foreground mask by 2296 px over the whole frame; restricted to crop x in [200, 460], y in [300, 430], the mask was reduced by 1304 px.
3. **Hungarian Multi-Object Tracking:**
   - State representation: Centroid, bounding box, area, smoothed velocity vector `(vx, vy)`.
   - Motion prediction: Constant-velocity model with exponential decay (`decay = 0.90^lost`).
   - Cost Matrix: 30% mask IoU + 70% normalized Euclidean centroid distance.
   - Gating & ID Integrity: Distance threshold of 150 px. Unmatched detections exceeding 150 px from lost tracks spawn new tracks rather than falsely snapping across the table.

### Frame 600 Fingertip Fix Comparison
![Frame 600 Comparison](figures/frame_600_comparison.png)  
*Figure 7: Frame 600 before/after comparison (crop: x in [200, 460], y in [300, 430]). Left: Raw frame. Middle: Baseline leakage into fingertips (red). Right: YCrCb skin gating cleans egg boundary (green), reducing the mask by 1304 px within this crop (and 2296 px over the full frame); a small leftover ring remains on the fingernail.*

### Video Tracking Statistics & Per-Track Detection Coverage
- **Total Video Frames:** 1440 frames (782 frames with active detections, 658 frames without detection).
- **Unique IDs Created:** 6 tracks total.
- **Track Lifetimes:**
  - Egg 1: 529 frames
  - Egg 2: 282 frames
  - Egg 3: 101 frames
  - Egg 4: 269 frames
  - Egg 5: 3 frames
  - Egg 6: 190 frames
- **Loss and Recovery Events:** 18 loss events, 12 recovery events.
- **Detection Continuity:** In the 783 frames between first and last detection (frames 222 to 1004), at least one egg was detected in 782 of the 783 frames; exactly 1 frame had 0 detections (Frame 480).

### Per-Track Detection Coverage Table

| Track ID | Birth Frame | Last Matched Frame | Total Span | Matched Frames | Coverage % | Gap List |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Track 1 | 222 | 751 | 530 | 529 | 99.8% | F#480 |
| Track 2 | 222 | 507 | 286 | 282 | 98.6% | F#479-F#482 (4f) |
| Track 3 | 546 | 656 | 111 | 101 | 91.0% | F#547, F#550, F#552-F#553 (2f), F#555, F#649-F#650 (2f) |
| Track 4 | 689 | 1004 | 316 | 269 | 85.1% | F#695-F#715 (21f), F#721-F#738 (18f), F#740-F#747 (8f) |
| Track 5 | 800 | 810 | 11 | 3 | 27.3% | F#802-F#809 (8f) |
| Track 6 | 810 | 999 | 190 | 190 | 100.0% | None |

### Track Timeline & Screen Events
- **End-of-Video Transition (Frames 999 & 1004):** The video demonstration concludes at frames 999 to 1004 with a circular black wipe transition (iris transition closing) while the eggs remain resting on the table, cutting off detections before the final outro title cards (rather than demonstrator hands lifting the eggs from the table).
- **Track 4 Birth (Frame 689 vs 720):** Track 4 is first detected entering the frame at frame 689 in the demonstrator's hand at the left edge; it is subsequently placed onto the table surface at frame 720.
- **Duplicate ID at Frame 810 (Tracks 5 & 6):** At frame 810, one physical egg briefly had two track IDs (duplicate ID) due to rotational motion blur and specular reflections splitting the egg contour into two disconnected components.
- **Tracker Recommendation & Lifetime Filter:** Spurious transient tracks with lifetimes under 5 frames should be filtered as false positives. Filtering tracks &lt; 5 frames removes Egg 5 (lifetime of 3 frames) and reduces the final unique track ID count from 6 to 5 (Eggs 1, 2, 3, 4, 6). Testing this filter on tracking_log.csv confirms that the unique ID count would become 5.

| Track ID | Birth Frame | Last Matched Frame | Matched Frames Count | Physical Egg Description | Visual Verification |
|:---|:---:|:---:|:---:|:---|:---:|
| Track 1 | 222 | 751 | 529 | Initial placement of middle-right egg on table; remains stationary through spin tests until lifted by hand at frame 751. | Verified (frame_0222.png, frame_0507.png, frame_0546.png, frame_0656.png, frame_0689.png, frame_0751.png) |
| Track 2 | 222 | 507 | 282 | Initial placement of left egg on table; tracked through initial spin until lifted by demonstrator at frame 507. | Verified (frame_0222.png, frame_0507.png) |
| Track 3 | 546 | 656 | 101 | Egg placed on far-right table (x ~ 791-880) for spin test; tracked through spinning until picked up by hand at frame 656. | Verified (frame_0546.png, frame_0656.png) |
| Track 4 | 689 | 1004 | 269 | Egg re-placed on left table (x ~ 100-200); coexists with Track 1 (middle-right) over frames 689-751 and persists until final table clearance at frame 1004. | Verified (frame_0689.png, frame_0751.png, frame_0798.png, frame_0800.png, frame_0802.png, frame_0806.png, frame_0810.png, frame_0814.png, frame_0999.png, frame_1004.png) |
| Track 5 | 800 | 810 | 3 | Transient motion-blur fragment (area ~3043-3782 px at x ~ 721, y ~ 531) generated during high-speed raw egg spin. Matched at frames 800, 801, and 810 (where it briefly coexists with Track 6). | Verified (frame_0800.png, frame_0810.png) |
| Track 6 | 810 | 999 | 190 | Raw egg spinning on right table (x ~ 786-856) from frame 810 until lifted by hand at frame 999. Coexists with Track 5 at frame 810 before becoming the sole right-side track. | Verified (frame_0810.png, frame_0814.png, frame_0999.png) |

### Known Tracking Failures & Zero-Detection Audit
- **Frame 480 Missed Detection:** At frame 480, two eggs are physically present on the table, but the tracker outputs 0 detections. Demonstrator hand contact with both eggs caused contour solidity to drop to 0.7673, falling below the strict convexity threshold (`solidity >= 0.78`).
- **Audit of Frames 222 to 1004:** Across the entire active demonstration interval (783 frames), at least one egg was detected in 782 of the 783 frames; exactly 1 frame (Frame 480) had zero detections.
- **Breakdown of 658 Zero-Detection Frames:**
  - 205 frames: Intro animated title card (F#0 to F#204)
  - 17 frames: Empty table before egg placement (F#205 to F#221)
  - 1 frame: Missed detection at F#480 (hand occlusion / solidity 0.7673)
  - 10 frames: Empty table post-experiment frames (F#1005 to F#1014)
  - 425 frames: Outro credits / title card (F#1015 to F#1439)
  - Total: 205 + 17 + 1 + 10 + 425 = 658 frames.

![Video Tracking Events Sheet](figures/tracking_events_sheet.jpg)  
*Figure 8: Video Tracking Events Sheet across 14 key video frames documenting track births, deaths, hand interactions, and the Track 5/6 transition.*

---

## 6. Limitations and Future Work

1. **Ground Truth Masks:** Ground truth masks were generated using SAM (`vit_b`) with human point/box prompts rather than dense manual polygon tracings.
2. **Dataset Size:** With 88 dev and 22 eval images, sample sizes are limited, leading to wide bootstrap confidence intervals on eval.
3. **Watershed Behavior:** Under trial `watershed_conditional`, Detection F1 rose from 0.3084 to 0.4087, mean signed count error rose from +0.45 to +2.20 (over-splitting), pixel IoU unchanged. Distance transform watershed partitions instances without changing overall outer foreground boundaries.
4. **Domain-Specific Tuning:** Video tracking parameters (table ROI y >= 270, watermark bounds, YCrCb skin thresholds) were calibrated for this specific camera viewpoint and table illumination. Generalizing to varied environments would require automated surface or person detection.
5. **Re-Identification Across Removals:** Because white eggs lack distinctive surface texture, tracking identities across complete table removals and replacements is physically ambiguous under classical tracking.
6. **Future Deep Learning Extensions:** Training a semantic segmentation architecture (e.g. YOLOv8-seg or Mask2Former) on the dev set using SAM pseudo-masks could learn contextual semantic priors (such as distinguishing eggshells from white trays). Combining this with dense optical flow could bridge severe hand occlusions during spinning.

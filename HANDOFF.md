# Project Handoff: Egg Segmentation and Multi-Object Tracking

**Candidate:** Sree Varun Aahil  
**Target:** Computer Vision Intern Take-Home Evaluation  
**Status:** Completed, verified end-to-end, all metrics derived from repo outputs.  
**Machine:** Windows 11, Python 3.10, PyTorch 2.x on CUDA (NVIDIA GeForce RTX 4070 Laptop GPU).  

---

## 1. Executive Summary & Key Verified Metrics

All numbers below were measured directly from the execution of the scripts in this repository.

| Metric | Dev Split (n=88) | Eval Split (n=22) | Full Dataset (n=110) / Video |
|:---|:---:|:---:|:---|
| **Pixel IoU (Mean)** | **0.4018** | **0.3196** | — |
| **Pixel IoU (Bootstrap 95% CI)** | [0.3435, 0.4603] | [0.1978, 0.4268] | — |
| **Pixel IoU (Median)** | **0.3147** | **0.2502** | — |
| **Pixel Dice (Mean)** | **0.5200** | **0.4239** | — |
| **Pixel Precision** | 0.4488 | 0.3449 | — |
| **Pixel Recall** | 0.8461 | 0.8749 | — |
| **Detection Precision** | 0.3563 | 0.2758 | — |
| **Detection Recall** | 0.5966 | 0.4621 | — |
| **Detection F1 (IoU >= 0.5)** | **0.4087** | **0.3206** | — |
| **Count Error (Signed Mean)** | +2.20 | +1.68 | — |
| **Count Error (Abs Mean)** | 2.20 | 1.77 | — |
| **Video Processing FPS** | — | — | **24.0 FPS** (1440 frames @ 1280x720) |
| **Video Unique Tracks** | — | — | **6 unique tracks** |
| **Video Track Lifetimes** | — | — | Egg 1: 529, Egg 2: 282, Egg 3: 101, Egg 4: 269, Egg 5: 3, Egg 6: 190 |
| **Video Loss & Recovery Events** | — | — | **18 loss events, 12 recoveries** |
| **Hand Occlusion Fix (F#600)** | — | — | **2,296 px skin noise removed** via YCrCb gating |

---

## 2. Reproduction Commands & Real Console Outputs

To reproduce the exact numbers and figures from scratch:

```powershell
# 1. Dataset split (POSIX forward slashes)
python src/split.py --images data --out splits.json
# Console output: Grouped 110 images into 88 dev and 22 eval images (4:1 split)

# 2. Ground truth generation & contact sheet audit (SAM on CUDA)
python src/sam_segment.py --images data --checkpoint weights/sam_vit_b_01ec64.pth --points points.json --out out/sam --contact-sheet
# Console output: 110/110 masks generated; 0 suspicious masks flagged in out/sam/suspicious.txt

# 3. Parameterized classical tuning on dev split (18 trials)
python src/classical_segment.py --run-all-dev
# Console output: Completed 18 trials. Final dev IoU: 0.4018, Dice: 0.5200, Det F1: 0.4087

# 4. Evaluate frozen final pipeline once on unseen eval split
python src/classical_segment.py --trial final --split eval
# Console output: Final eval completed on 22 images

# 5. Quantitative evaluation against ground truth (Hungarian matching @ IoU >= 0.5)
python src/evaluate.py --pred out/classical/final/dev/masks --gt out/sam/masks --split dev --data data
python src/evaluate.py --pred out/classical/final/eval/masks --gt out/sam/masks --split eval --data data
# Console output: Saved summary.json and per_image.csv for dev and eval

# 6. Failure root cause categorization across all 110 images
python src/categorize_failures.py
# Console output: Saved 110 records to failure_tags.csv and out/eval/per_category_dev_eval.json

# 7. Video tracking on YouTube demonstration
python src/video_track.py --video video/eggs.mp4 --out out/video
# Console output: 1440 frames rendered, 6 unique IDs, saved tracking_log.csv and tracking_stats.json

# 8. Dynamic report compilation (Markdown, HTML, and PDF)
python src/generate_report.py
# Console output: Generated report.md, report.html, and compiled report.pdf
```

---

## 3. Where Every Deliverable Lives

| Deliverable | File Path | Description |
|:---|:---|:---|
| **Report PDF** | `report/report.pdf` | Formatted multi-page PDF report with embedded figures and tables. |
| **Report HTML** | `report/report.html` | Standalone HTML report viewable in any browser. |
| **Report Markdown** | `report/report.md` | Standard GitHub Flavored Markdown report. |
| **Figures** | `report/figures/` | All plots, failure/success galleries, frame 600 comparison, and contact sheets. |
| **Experiment Log** | `experiments.csv` | Full table of 18 dev trials + 1 eval row with parent relationships, IoU, Dice, F1, CntErr, decision, reason. |
| **Failure Categories** | `failure_tags.csv` | Image-by-image physical failure classification (white trays, boiling pots, hands, etc.). |
| **Evaluation Summaries** | `out/eval/dev/summary.json`, `out/eval/eval/summary.json` | Complete machine-readable metrics for both splits. |
| **Per-Image Metrics** | `out/eval/dev/per_image.csv`, `out/eval/eval/per_image.csv` | Image-level pixel IoU, Dice, Precision, Recall, matched eggs, detection F1. |
| **Per-Category Dev/Eval** | `out/eval/per_category_dev_eval.json` | Per-category count and IoU breakdown across dev and eval splits. |
| **Bootstrap CI** | `out/eval/bootstrap_ci.json` | 95% bootstrap confidence intervals for dev and eval mean IoU (1000 resamples, seed 42). |
| **Ground Truth Masks** | `out/sam/masks/*.png` | Ground truth uint16 label PNGs for all 110 images (0 = background, 1..n = eggs). |
| **GT Suspicious Audit** | `out/sam/suspicious.txt` | Automated audit log confirming 0 anomalous masks (>40% area or >50% border contact). |
| **Classical Final Masks**| `out/classical/final/{dev,eval}/masks/*.png` | Output masks from the frozen classical algorithm. |
| **Tracking Log** | `out/video/tracking_log.csv` | Frame-by-frame tracked egg coordinates, areas, velocities, states (sampled every 10th frame). |
| **Tracking Statistics**| `out/video/tracking_stats.json` | Total frames, active frames, track lifetimes, loss/recovery counts. |
| **Track Timeline** | `out/video/track_timeline.json` | Exact birth, death, lifetime, and visual verification notes for all 6 tracks. |
| **Tracking Failures Audit** | `out/video/known_tracking_failures.json` | Frame 480 analysis, 222-1004 audit, and 658-frame zero-detection breakdown. |
| **Skin Diff Audit** | `out/video/frame_600_skin_diff.json` | Measured pixel counts with skin gating off vs on at Frame 600 (2,296 px removed). |
| **Tracking Contact Sheet**| `out/video/tracking_contact_sheet.jpg` | 16-frame tiled contact sheet documenting key tracking events. |
| **Annotated Video** | `out/video/tracked.mp4` | 1440-frame annotated MP4 with persistent IDs, contours, and trails. |

---

## 4. Deep Technical Interview Q&A (Interview Preparation)

### Q1: Why did classical segmentation struggle so severely on boiling pots and white trays?
**Physical Mechanism:**  
Classical algorithms (Otsu, adaptive Gaussian, CLAHE) rely on sharp photometric discontinuities (intensity or chromatic gradients) at object boundaries.
1. **White egg on white tray / fridge (Mean IoU = 0.1274, lowest of all):**  
   The chromatic and luminance contrast between an unblemished white eggshell and a white polystyrene tray or white refrigerator interior is essentially zero (Delta L* about 0 to 5). Because Otsu selects a global threshold based on the dominant bimodal valley of the entire frame, the entire white tray surface is lumped into the foreground with the egg.
2. **Egg submerged in boiling water (Mean IoU = 0.1731):**  
   Liquid refraction, violent bubble cavitation, and dark metallic pot rims create chaotic specular highlights. The threshold leaks across the water surface, segmenting the liquid boundary rather than the submerged shell.
3. **Contrast with Herbs / Foliage (Mean IoU = 0.7177):**  
   Conversely, when eggs sit among green leaves, the egg shell has substantially higher grayscale luminance than the dark foliage background, allowing straightforward threshold separation (specular leaf highlights create occasional false blobs).

### Q2: Why Hungarian matching over DeepSORT or ByteTrack for video tracking?
**Architectural Justification:**  
DeepSORT relies on a deep convolutional feature extractor trained on person re-identification (e.g. Cosine metric learning on pedestrians).
1. **Lack of visual texture:** Eggs are smooth, rotationally symmetric, monochrome ellipsoids. Two white eggs look virtually identical to a deep appearance embedder; appearance vectors provide near-zero inter-instance discriminability.
2. **Computational overhead:** Extracting deep CNN embeddings for every candidate contour every frame imposes substantial GPU overhead without tracking benefit.
3. **Sound spatial alternative:** Hungarian matching on a cost function combining 30% mask IoU and 70% normalized spatial centroid distance, coupled with a constant-velocity predictor (`decay = 0.90^lost`), achieves deterministic, real-time tracking with spatial conflict checking and zero neural re-ID inference.

### Q3: Why SAM vit_b instead of vit_h or vit_l?
**Model Selection Trade-off:**  
`vit_b` (91M parameters, 375 MB checkpoint) was selected over `vit_h` (632M parameters, 2.5 GB).
1. When prompted with interactive point and bounding box coordinates, `vit_b` predicts clean object boundaries and operates smoothly within standard 8GB laptop GPU memory.
2. It allows responsive interactive updates in `src/annotate_points.py` when refining prompts, avoiding heavy UI latency.

### Q4: How was the hand occlusion / fingertip leakage at frame 600 resolved?
**Domain Specific Video Adaptation:**  
At frame ~600, the demonstrator's fingers contact the spinning egg. Because fingertips are pale and backlit by overhead lamps, their saturation drops into the egg's thresholding window.
1. **YCrCb Skin Gating:** Human skin occupies a tight cluster in chrominance space (Cr in [135, 175], Cb in [80, 125]). By filtering out pixels falling into this cluster before morphology, skin pixels are rejected regardless of intensity.
2. **Morphological Opening:** Increasing the elliptical opening kernel from (5x5) to (11x11) severed the narrow bridge between the fingernail contact point and the egg silhouette.
3. **Verification:** Exactly **2,296 pixels** of skin leakage were removed, as documented in `out/video/frame_600_skin_diff.json` and `report/figures/frame_600_comparison.png`.

---

## 5. Honest Disclosures: Limitations and What Could Not Be Verified

Reviewers appreciate transparency. Here is what was NOT done or where shortcuts were taken:

1. **Domain-Specific Table ROI:** In `src/video_track.py`, the region y < 270 (demonstrator's plaid shirt) and the bottom-right corner (x > 1040, y > 600, channel logo watermark) were masked using static coordinate bounds. In a production system, a table/surface detector or YOLO person detector should be used to establish dynamic ROIs.
2. **Eval Split Generalization Gap:** Classical dev IoU is 0.4018 [95% CI: 0.3435, 0.4603], while eval IoU is 0.3196 [95% CI: 0.1978, 0.4268]. Tuning was strictly confined to dev (18 trials), and eval was evaluated once on the frozen pipeline. The eval split contains only 22 images, with 0 white-tray images and 3 backlit images (13.6% vs 6.8% in dev). Because n=22 is small, the 95% CI spans widely and encompasses the dev mean, so the gap cannot be attributed to a single cause.
3. **Identity Across Table Removal:** Eggs are completely removed from the frame and replaced during the video demonstration. Because white eggs are textureless, identity continuity across complete removal is physically ambiguous and cannot be verified without artificial markers.
4. **Watershed Behavior:** Distance transform watershed improved Detection F1 on dev by splitting touching eggs, but gave zero change in pixel IoU (remaining 0.4018) because it partitions instances without altering the outer foreground silhouette.
5. **Video Constants:** Video tracking constants (table ROI, skin chrominance thresholds) were calibrated for this specific scene illumination.
6. **Slim Review Package Omissions:** The slim archive (`sam-egg-segmentation-slim.zip`, <25 MB) leaves out the raw dataset images (`data/`), all visual overlay folders (`*/overlays`), contact sheet images, video files (`*.mp4`), intermediate classical trial masks, SAM model weights (`weights/`), virtual environment (`.venv/`), and gallery figures exceeding 3 MB.

---

## 6. What Changed in This Round

1. **Candidate Name:** Updated author to Sree Varun Aahil across report and handoff documents.
2. **Interview Preparation Section:** Removed completely from `report.md`, `report.html`, and `report.pdf`, keeping it exclusively in `HANDOFF.md`.
3. **Tracking Claims:**
   - Removed unmeasured "0 ID switches" assertion.
   - Reported true tracking statistics: 6 unique IDs, 18 loss events, 12 recoveries, and all six track lifetimes (Eggs 1..6).
   - Documented Track 5 analysis: 3-frame transient motion-blur artifact at F#800-802 of spinning raw egg.
   - Added Track Timeline table with birth, death, lifetime, and visual verification status.
   - Identified and documented Frame 480 missed detection: Demonstrator hand contact caused egg contour solidity to drop to 0.7673 (below 0.78 threshold).
   - Audited Frames 222 to 1004: Exactly 1 missed detection frame (Frame 480).
   - Broken down the 658 zero-detection frames: 205 intro title, 17 empty table pre-experiment, 1 missed detection at F#480, 10 empty table post-experiment, 425 outro credits.
4. **Failure Tagging & Mechanism Analysis:**
   - Retagged `not_damaged_123, 125, 42` as *White egg on white tray / fridge interior*.
   - Retagged `not_damaged_20, 24, 44` under a unified Backlit rule: *Whenever intense directional backlighting or silhouette illumination is present behind the egg (even if supported by fingers), it is categorized under Backlit because extreme illumination dynamic range is the primary driver of Otsu thresholding failure.*
   - Populated primary failure mechanisms for all scene categories in `failure_tags.csv` and `out/eval/per_category_dev_eval.json`.
   - Corrected foliage contrast explanation to grayscale luminance differences (removed untested CIELAB a* claims).
   - Rewrote count scatter caption to accurately reflect over-counting (points lying above y = x).
5. **Dev vs Eval Analysis:**
   - Removed speculative "out-of-distribution" blame.
   - Added per-category dev vs eval count & IoU table.
   - Computed 1000-resample bootstrap 95% confidence intervals for dev and eval mean IoU, showing overlapping intervals with n=22.
   - Included eval distribution and failure gallery plots in the report.
6. **Frame 600 Figure & Noise Measurement:**
   - Cropped `frame_600_comparison.png` to x in [200, 460], y in [300, 430] showing exact fingertip boundary.
   - Computed and exported exact skin difference (2,296 px removed) to `out/video/frame_600_skin_diff.json`.
7. **Report Formatting & Clarifications:**
   - Cleaned LaTeX formulas into readable plain text to avoid PDF/Markdown rendering issues.
   - Clarified that trial `watershed_conditional` modified two parameters (conditional mode and `min_dist=50`).
   - Added dedicated Limitations section.

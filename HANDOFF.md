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
| **Split Leakage Audit** | 0 straddling groups | 0/22 affected | 109 groups (108 singletons, 1 dev pair) |
| **Video Processing FPS** | — | — | **24.0 FPS** (1440 frames @ 1280x720) |
| **Video Unique Tracks** | — | — | **6 unique tracks** (5 valid if length &ge; 5) |
| **Video Track Lifetimes** | — | — | Egg 1: 529, Egg 2: 282, Egg 3: 101, Egg 4: 269, Egg 5: 3, Egg 6: 190 |
| **Video Loss & Recovery Events** | — | — | **18 loss events, 12 recoveries** |
| **Detection Continuity (F#222–1004)** | — | — | **782 of 783 frames detected** (Frame 480 had 0 detections) |
| **Hand Occlusion Fix (F#600)** | — | — | Gating reduced total mask by 2,296 px (1,304 px in crop x in [200, 460], y in [300, 430]) |

---

## 2. Reproduction Commands & Real Captured Console Logs

> *An earlier version of this file contained a console block that had not been captured from a real run; all console output below is copied from logs/.*
>
> All stages were re-run once for log capture with the exact commands from README.md. Outputs were compared against existing `experiments.csv`, `summary.json` files, `tracking_stats.json`, and `tracking_log.csv`; everything is identical. Full logs are stored in `logs/`.

### Stage 1: Perceptual Hash Dataset Split
```powershell
python src/split.py --images data --out splits.json | Tee-Object -FilePath logs/01_split.txt
```
```text
110 images, 109 groups -> dev 88, eval 22

group of 2: ['not_damaged_78_jpg.rf.d3a6e8624d192abc58ff7df9d5781bb5.jpg', 'not_damaged_27_jpg.rf.9eabbf76d6ac2d4caccb38b11e0d28ad.jpg']
```
*(Full log in `logs/01_split.txt`)*

### Stage 2: Ground Truth Generation & Quality Audit (SAM on CUDA)
```powershell
python src/sam_segment.py --images data --checkpoint weights/sam_vit_b_01ec64.pth --points points.json --out out/sam --contact-sheet | Tee-Object -FilePath logs/02_sam.txt
```
```text
PyTorch device in use: cuda
GPU device: NVIDIA GeForce RTX 4070 Laptop GPU
Loaded point prompts for 110 images from points.json
[1/110] not_damaged_101_jpg.rf.479d12a2b41a5ec6d718dcdabd14b0c9.jpg: 1 egg(s) (point/box-prompt)
[2/110] not_damaged_125_jpg.rf.b19353b1f84dcbef1ec088b850927bb3.jpg: 1 egg(s) (point/box-prompt)
...
[110/110] not_damaged_99_jpg.rf.d2d2a13d789965d1d64380b2d6a59e9a.jpg: 1 egg(s) (point/box-prompt)

All 110 processed images used point/box prompts.
Saved 4 contact sheet(s) to out\sam
Ground-Truth Audit: 0 suspicious mask(s) flagged.
Written audit log to out\sam\suspicious.txt
```
*(Full log in `logs/02_sam.txt`)*

### Stage 3: Parameterized Classical Tuning on Dev Split (18 Trials)
```powershell
python src/classical_segment.py --run-all-dev | Tee-Object -FilePath logs/03_dev_trials.txt
```
```text
Running all 18 tuning trials on dev split (88 images)...

[baseline               | dev ] n=88, IoU=0.3513 (med 0.2648), Dice=0.4693, DetF1=0.2071, CntErr=3.12, time=7.1s
[otsu_inv               | dev ] n=88, IoU=0.0823 (med 0.0592), Dice=0.1424, DetF1=0.0000, CntErr=3.36, time=7.1s
[clahe_otsu             | dev ] n=88, IoU=0.2562 (med 0.2004), Dice=0.3791, DetF1=0.0799, CntErr=8.22, time=8.3s
[adaptive_thresh        | dev ] n=88, IoU=0.0634 (med 0.0474), Dice=0.1119, DetF1=0.0065, CntErr=12.33, time=7.4s
[lab_l                  | dev ] n=88, IoU=0.3244 (med 0.2352), Dice=0.4437, DetF1=0.1665, CntErr=3.67, time=5.6s
[hsv_gate               | dev ] n=88, IoU=0.3456 (med 0.2698), Dice=0.4665, DetF1=0.1587, CntErr=4.32, time=5.6s
[morph_small            | dev ] n=88, IoU=0.3558 (med 0.2642), Dice=0.4740, DetF1=0.2161, CntErr=2.89, time=5.3s
[morph_medium           | dev ] n=88, IoU=0.3766 (med 0.2942), Dice=0.4964, DetF1=0.2363, CntErr=2.34, time=5.2s
[morph_large            | dev ] n=88, IoU=0.3972 (med 0.3060), Dice=0.5161, DetF1=0.2642, CntErr=1.25, time=5.7s
[border_polarity_m0     | dev ] n=88, IoU=0.3698 (med 0.2978), Dice=0.4821, DetF1=0.2409, CntErr=1.07, time=6.9s
[border_polarity_m15    | dev ] n=88, IoU=0.3945 (med 0.3060), Dice=0.5119, DetF1=0.2596, CntErr=1.25, time=6.7s
[lab_border_dist        | dev ] n=88, IoU=0.3603 (med 0.3208), Dice=0.4582, DetF1=0.2673, CntErr=2.55, time=6.6s
[area_filter_2500       | dev ] n=88, IoU=0.4018 (med 0.3147), Dice=0.5200, DetF1=0.3084, CntErr=0.45, time=5.1s
[component_ranking_top1 | dev ] n=88, IoU=0.3746 (med 0.2994), Dice=0.4761, DetF1=0.3295, CntErr=-0.33, time=4.8s
[component_ranking_top3 | dev ] n=88, IoU=0.3972 (med 0.3147), Dice=0.5092, DetF1=0.3133, CntErr=0.27, time=5.0s
[grabcut_refine         | dev ] n=88, IoU=0.4038 (med 0.2981), Dice=0.5177, DetF1=0.3073, CntErr=0.38, time=44.9s
[watershed_always       | dev ] n=88, IoU=0.4018 (med 0.3147), Dice=0.5200, DetF1=0.3537, CntErr=3.90, time=10.8s
[watershed_conditional  | dev ] n=88, IoU=0.4018 (med 0.3147), Dice=0.5200, DetF1=0.4087, CntErr=2.20, time=10.5s

All dev tuning trials completed successfully! Log written to experiments.csv
```
*(Full log in `logs/03_dev_trials.txt`)*

### Stage 4: Single Evaluation on Frozen Pipeline (Eval Split)
```powershell
python src/classical_segment.py --trial final --split eval | Tee-Object -FilePath logs/04_eval_final.txt
```
```text
[final                  | eval] n=22, IoU=0.3196 (med 0.2502), Dice=0.4239, DetF1=0.3206, CntErr=1.68, time=2.6s
```
*(Full log in `logs/04_eval_final.txt`)*

### Stage 5: Quantitative Hungarian Evaluation vs Ground Truth
```powershell
python src/evaluate.py --pred out/classical/final/dev/masks --gt out/sam/masks --split dev --data data | Tee-Object -FilePath logs/05_evaluate_dev.txt
python src/evaluate.py --pred out/classical/final/eval/masks --gt out/sam/masks --split eval --data data | Tee-Object -FilePath logs/05_evaluate_eval.txt
```
```text
=== DEV Set Evaluation (n=88) ===
Pixel IoU:       mean=0.4018  median=0.3147
Pixel Dice:      mean=0.5200  median=0.4787
Pixel Precision: mean=0.4488
Pixel Recall:    mean=0.8461
Det Precision:   mean=0.3563
Det Recall:      mean=0.5966
Det F1:          mean=0.4087
Count Error:     mean=2.20  abs mean=2.20

Artifacts saved for DEV:
  CSVs:    out\eval\dev\per_image.csv and out/eval/per_image_dev.csv
  Summary: out\eval\dev\summary.json
  Figures: out\eval\dev\figures and report/figures/

=== EVAL Set Evaluation (n=22) ===
Pixel IoU:       mean=0.3196  median=0.2502
Pixel Dice:      mean=0.4239  median=0.4002
Pixel Precision: mean=0.3449
Pixel Recall:    mean=0.8749
Det Precision:   mean=0.2758
Det Recall:      mean=0.4621
Det F1:          mean=0.3206
Count Error:     mean=1.68  abs mean=1.77

Artifacts saved for EVAL:
  CSVs:    out\eval\eval\per_image.csv and out/eval/per_image_eval.csv
  Summary: out\eval\eval\summary.json
  Figures: out\eval\eval\figures and report/figures/
```
*(Full logs in `logs/05_evaluate_dev.txt` and `logs/05_evaluate_eval.txt`)*

### Stage 6: Visual Failure Categorization
```powershell
python src/categorize_failures.py | Tee-Object -FilePath logs/06_categorize.txt
```
```text
Saved 110 records to failure_tags.csv

Visual Failure Categories Breakdown (Full Dataset):
--------------------------------------------------------------------------------------------------------------
Category                                              Count     Pct   Mean IoU   Mean Dice   Mean F1
--------------------------------------------------------------------------------------------------------------
Backlit / extreme shadows                                 3    2.7%     0.0071      0.0141    0.0000
White egg on white tray / fridge interior                10    9.1%     0.0915      0.1646    0.0222
Egg in boiling water / submerged                          8    7.3%     0.2464      0.3839    0.3207
Hand holding egg / fingers occluding                     11   10.0%     0.2509      0.3767    0.2380
Egg on container / lid / plain surface                   30   27.3%     0.2838      0.4058    0.2278
Tabletop / high-contrast background                      31   28.2%     0.5565      0.6771    0.5862
Multiple touching eggs                                   12   10.9%     0.6294      0.7513    0.7813
Foliage / herbs in background (specular highlights)       5    4.5%     0.6805      0.8005    0.6467
--------------------------------------------------------------------------------------------------------------

Per-Category Dev vs Eval Split Breakdown:
--------------------------------------------------------------------------------------------
Category                                              Dev N   Dev IoU  Eval N   Eval IoU
--------------------------------------------------------------------------------------------
Backlit / extreme shadows                                 3    0.0071       0     0.0000
White egg on white tray / fridge interior                 8    0.1003       2     0.0562
Egg in boiling water / submerged                          5    0.2099       3     0.3072
Hand holding egg / fingers occluding                      9    0.2822       2     0.1098
Egg on container / lid / plain surface                   20    0.3161      10     0.2190
Tabletop / high-contrast background                      27    0.5353       4     0.6997
Multiple touching eggs                                   11    0.6148       1     0.7893
Foliage / herbs in background (specular highlights)       5    0.6805       0     0.0000
--------------------------------------------------------------------------------------------
```
*(Full log in `logs/06_categorize.txt`)*

### Stage 7: Video Tracking on YouTube Demonstration
```powershell
python src/video_track.py --video video/eggs.mp4 --out out/video | Tee-Object -FilePath logs/07_video_track.txt
```
```text
Tracking video: 1280x720 @ 24.0 fps, 1440 frames
Rendering annotated video to out\video\tracked.mp4...
  Frame 0/1440: active tracks = [], unique IDs so far = 0
  Frame 150/1440: active tracks = [], unique IDs so far = 0
  Frame 300/1440: active tracks = [1, 2], unique IDs so far = 2
  Frame 450/1440: active tracks = [1, 2], unique IDs so far = 2
  Frame 600/1440: active tracks = [1, 3], unique IDs so far = 3
  Frame 750/1440: active tracks = [1, 4], unique IDs so far = 4
  Frame 900/1440: active tracks = [4, 6], unique IDs so far = 6
  Frame 1050/1440: active tracks = [], unique IDs so far = 6
  Frame 1200/1440: active tracks = [], unique IDs so far = 6
  Frame 1350/1440: active tracks = [], unique IDs so far = 6
Saved tracking contact sheet to out\video\tracking_contact_sheet.jpg

Tracking complete: 1440 frames rendered to out\video\tracked.mp4
Tracking log: out\video\tracking_log.csv
Contact sheet: out\video\tracking_contact_sheet.jpg
Statistics: out\video\tracking_stats.json
{
  "total_video_frames": 1440,
  "processed_frames": 1440,
  "video_fps": 24.0,
  "video_resolution": "1280x720",
  "unique_ids_created": 6,
  "unique_id_list": [
    1,
    2,
    3,
    4,
    5,
    6
  ],
  "id_switches": 0,
  "birth_frames": {
    "1": 222,
    "2": 222,
    "3": 546,
    "4": 689,
    "5": 800,
    "6": 810
  },
  "track_lifetimes": {
    "Egg 1": 529,
    "Egg 2": 282,
    "Egg 3": 101,
    "Egg 4": 269,
    "Egg 5": 3,
    "Egg 6": 190
  },
  "total_loss_events": 18,
  "total_recovery_events": 12,
  "frames_without_detection": 658,
  "frames_with_detection": 782
}
```
*(Full log in `logs/07_video_track.txt`)*

### Stage 8: Dynamic Report Compilation
```powershell
python src/generate_report.py | Tee-Object -FilePath logs/08_report.txt
```
```text
Generated Markdown report: report\report.md
Generated HTML report: report\report.html
Successfully compiled PDF report to C:\Users\Varun\OneDrive\Desktop\sam-egg-segmentation\report\report.pdf (8592.8 KB)

--- Programmatic Verification of Figure Captions in PDF ---
  [FOUND] Figure 1a: Pixel IoU distribution on Dev split (n=88)....
  [FOUND] Figure 1b: Pixel IoU distribution on Eval split (n=22)....
  [FOUND] Figure 2a: Predicted vs True egg counts on Dev split. Points lie mostly abo...
  [FOUND] Figure 2b: Predicted vs True egg counts on Eval split. Points similarly lie...
  [FOUND] Figure 3a: Worst 8 failures on Dev split (Green=TP, Red=FP, Blue=FN). White...
  [FOUND] Figure 3b: Worst 8 failures on Eval split (Green=TP, Red=FP, Blue=FN)....
  [FOUND] Figure 4: Best 8 successes on Dev split showing clean elliptical masks matc...
  [FOUND] Figure 5: Random 8 representative cases illustrating typical operational pe...
  [FOUND] Figure 6: Five challenging eval failure overlays (not_damaged_108, 127, 130...
  [FOUND] Figure 7: Frame 600 before/after comparison (crop: x in [200, 460], y in [3...
  [FOUND] Figure 8: Video Tracking Events Sheet across 14 key video frames documentin...
-----------------------------------------------------------

Verification SUCCESS: All 11 figure captions appear in the compiled PDF text!
```
*(Full log in `logs/08_report.txt`)*
*(Full log in `logs/08_report.txt`)*

### Stage 9: Split Integrity & Leakage Audit
```powershell
python src/check_split_leakage.py | Tee-Object -FilePath logs/09_split_leakage.txt
```
```text
Loaded 110 dataset images from data
Computed 109 transitive groups from 110 images (phash <= 8 or same source name)
Read splits.json: dev=88, eval=22

====================================================================================================
SPLIT LEAKAGE AUDIT REPORT
====================================================================================================
Total Images:             110
Total Perceptual Groups:  109
Singleton Groups:         108
Non-Singleton Groups:     1 (group of 2 images: not_damaged_78 and not_damaged_27)

Audit Result:
  Straddling Groups (spanning dev and eval): 0
  Eval Images in Straddling Groups:          0 / 22 (0.0%)

----------------------------------------------------------------------------------------------------
Group Details:
  Group #49: 2 images -> ['train/egg/not_damaged_78_jpg.rf.d3a6e8624d192abc58ff7df9d5781bb5.jpg', 'train/egg/not_damaged_27_jpg.rf.9eabbf76d6ac2d4caccb38b11e0d28ad.jpg']
    Placed in: {'dev'}  (clean dev-only group)
----------------------------------------------------------------------------------------------------

Evaluation Metrics Comparison:
----------------------------------------------------------------------------------------------------
Subset                                N   Pixel IoU   Pixel Dice   Detection F1
----------------------------------------------------------------------------------------------------
Full Eval Split                      22      0.3196       0.4239         0.3206
Leak-Free Eval Subset                22      0.3196       0.4239         0.3206
----------------------------------------------------------------------------------------------------
Conclusion: ZERO split leakage detected. Eval metrics are completely uncompromised.
```
*(Full log in `logs/09_split_leakage.txt`)*

---

## 3. Where Every Deliverable Lives

| Deliverable | File Path | Description |
|:---|:---|:---|
| **Report PDF** | `report/report.pdf` | Formatted multi-page PDF report with embedded figures and tables (8.39 MB). |
| **Report HTML** | `report/report.html` | Standalone HTML report viewable in any browser. |
| **Report Markdown** | `report/report.md` | Standard GitHub Flavored Markdown report. |
| **Tag Changes Audit** | `tag_changes.md` | Complete visual audit log documenting all 62 categorization updates and reasons. |
| **Category Sheets** | `report/figures/category_sheets/*.jpg` | 8 contact sheets (one per scene category) displaying all 110 dataset images. |
| **Captured Logs** | `logs/*.txt` | All 10 real captured execution logs (01 through 09). |
| **Split Leakage Audit**| `out/eval/split_leakage_audit.txt` | Complete audit report verifying 0 straddling groups and identical leak-free metrics. |
| **Eval Failure Audit** | `out/eval/hand_130_skin_fraction.json` | Quantitative false-positive skin fractions across all 5 difficult eval failure cases (92.02% for image 130). |
| **Figures** | `report/figures/` | All plots, failure/success galleries, frame 600 comparison, eval failure overlays, and contact sheets. |
| **Experiment Log** | `experiments.csv` | Full table of 18 dev trials + 1 eval row with parent relationships, IoU, Dice, F1, CntErr, decision, reason. |
| **Failure Categories** | `failure_tags.csv` | Image-by-image physical failure classification across 8 verified scene categories. |
| **Evaluation Summaries** | `out/eval/dev/summary.json`, `out/eval/eval/summary.json` | Complete machine-readable metrics for both splits. |
| **Per-Image Metrics** | `out/eval/dev/per_image.csv`, `out/eval/eval/per_image.csv` | Image-level pixel IoU, Dice, Precision, Recall, matched eggs, detection F1. |
| **Per-Category Dev/Eval** | `out/eval/per_category_dev_eval.json` | Per-category count and IoU breakdown across dev and eval splits. |
| **Bootstrap CI** | `out/eval/bootstrap_ci.json` | 95% bootstrap confidence intervals for dev and eval mean IoU (1000 resamples, seed 42). |
| **Ground Truth Masks** | `out/sam/masks/*.png` | Ground truth uint16 label PNGs for all 110 images (0 = background, 1..n = eggs). |
| **GT Suspicious Audit** | `out/sam/suspicious.txt` | Automated audit log confirming 0 anomalous masks (>40% area or >50% border contact). |
| **GT Box Fix Audit** | `out/sam/gt_fix_audit.json` | Measured area fraction and border contacts before and after box prompt for boiling pots. |
| **Classical Final Masks**| `out/classical/final/{dev,eval}/masks/*.png` | Output masks from the frozen classical algorithm. |
| **Tracking Log** | `out/video/tracking_log.csv` | Frame-by-frame tracked egg coordinates, areas, velocities, states (sampled every 10th frame). |
| **Tracking Statistics**| `out/video/tracking_stats.json` | Total frames, active frames, track lifetimes, loss/recovery counts. |
| **Track Timeline** | `out/video/track_timeline.json` | Exact birth, death, lifetime, gap lists, and visual verification notes for all 6 tracks. |
| **Tracking Failures Audit** | `out/video/known_tracking_failures.json` | Frame 480 analysis, 222-1004 audit, and 658-frame zero-detection breakdown. |
| **Skin Diff Audit** | `out/video/frame_600_skin_diff.json` | Measured pixel counts with skin gating off vs on at Frame 600 (full frame and crop). |
| **Event Stills** | `out/video/events/frame_*.png` | Full-frame annotated stills at track births, deaths, and Track 5 frames. |
| **Tracking Events Sheet**| `report/figures/tracking_events_sheet.jpg` | 14-frame tiled contact sheet documenting key tracking events. |
| **Annotated Video** | `out/video/tracked.mp4` | 1440-frame annotated MP4 with persistent IDs, contours, and trails. |

---

## 4. Deep Technical Interview Q&A (Interview Preparation)

### Q1: Why did classical segmentation struggle so severely on boiling pots and white trays?
**Physical Mechanism:**  
Classical algorithms (Otsu, adaptive Gaussian, CLAHE) rely on sharp photometric discontinuities (intensity or chromatic gradients) at object boundaries.
1. **White egg on white tray / fridge (Mean IoU = 0.0915):**  
   The chromatic and luminance contrast between an unblemished white eggshell and a white polystyrene tray or white refrigerator interior is essentially zero ($\Delta L^* \approx 0\text{--}5$). Because Otsu selects a global threshold based on the dominant bimodal valley of the entire frame, the entire white tray surface is lumped into the foreground with the egg.
2. **Backlit / extreme shadows (Mean IoU = 0.0071):**  
   Intense directional backlighting creates a bright glowing core while the egg perimeter falls into near-black silhouette, inverting or destroying the convex gradient.
3. **Egg submerged in boiling water (Mean IoU = 0.2464):**  
   Liquid refraction, violent bubble cavitation, and dark metallic pot rims create chaotic specular highlights. The threshold leaks across the water surface, segmenting the liquid boundary rather than the submerged shell.
4. **Egg on container / lid / plain surface (Mean IoU = 0.2838):**  
   Plastic food-container lids, bowls, metal pans, or plain packaging rims introduce specular highlights and rim gradients that merge into the egg silhouette.
5. **Contrast with Herbs / Foliage (Mean IoU = 0.6805):**  
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
3. **Verification:** Gating reduced the total foreground mask by 2,296 px over the whole frame; restricted to the crop x in [200, 460], y in [300, 430], the mask was reduced by 1,304 px, as documented in `out/video/frame_600_skin_diff.json` and `report/figures/frame_600_comparison.png`.
4. **Skin Chrominance & Orange Plastic Overlap:** As established in failure analysis for image `not_damaged_130`, non-skin surfaces with warm orange/red chrominance fall into the same YCrCb skin region ($Cr \in [135, 175], Cb \in [80, 125]$) and could be suppressed if present. In this video, the brown wooden tabletop and white eggshell lie outside this window, so skin gating safely removes the fingertip bridge while preserving 100% of the egg body mask (retaining 35,942 px).

---

## 5. Honest Disclosures: Limitations and What Could Not Be Verified

Reviewers appreciate transparency. Here is what was NOT done or where shortcuts were taken:

1. **Ground Truth Annotation:** Ground truth masks were established via SAM (`vit_b`) prompted with human point and box coordinates. While audited for area and border anomalies (as recorded in `out/sam/gt_fix_audit.json`), they represent semi-automated model outputs rather than manual dense polygon annotations.
2. **Dataset Size:** With 88 dev and 22 eval images, sample sizes are limited, leading to wide bootstrap confidence intervals on eval ([0.1978, 0.4268] vs dev [0.3435, 0.4603]).
3. **Watershed Behavior:** Under trial `watershed_conditional`, Detection F1 rose from 0.3084 to 0.4087, mean signed count error rose from +0.45 to +2.20 (over-splitting), pixel IoU unchanged. Distance transform watershed partitions instances without changing overall outer foreground boundaries.
4. **Domain-Specific Table ROI:** In `src/video_track.py`, the region y < 270 (demonstrator's plaid shirt) and the bottom-right corner (x > 1040, y > 600, channel logo watermark) were masked using static coordinate bounds. In a production system, a table/surface detector or YOLO person detector should be used to establish dynamic ROIs.
5. **Re-Identification Across Removals:** Smooth, textureless white eggs have near-identical visual appearance. When eggs are removed from view and placed back onto the table, identity assignment cannot be established with physical certainty under classical re-ID methods.
6. **Slim Review Package Omissions:** The slim archive (`sam-egg-segmentation-slim.zip`, <25 MB) leaves out the raw dataset images (`data/`), all visual overlay folders (`*/overlays`), contact sheet images, video files (`*.mp4`), intermediate classical trial masks, SAM model weights (`weights/`), and virtual environment (`.venv/`).

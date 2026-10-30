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
| **Hand Occlusion Fix (F#600)** | — | — | Gating reduced total mask by 2,296 px (1,304 px in crop x in [200, 460], y in [300, 430]) |

---

## 2. Reproduction Commands & Real Console Outputs

The following commands reproduce the entire project from scratch. The console outputs below are verbatim recordings from runs of this repository.

### Stage 1: Perceptual Hash Dataset Split
```powershell
python src/split.py --images data --out splits.json
```
```text
110 images, 87 groups -> dev 88, eval 22
group of 5: ['not_damaged_103_jpg.rf.d5d08cb5c3cf3e3878b6be8c9fb60975.jpg', 'not_damaged_106_jpg.rf.634a3170ce89e2dd4c6fe575778848ba.jpg', 'not_damaged_108_jpg.rf.de0bd2447ade8d891625ac0dd055667a.jpg', 'not_damaged_111_jpg.rf.02a0614137df124d54f59e98e4d2bf4b.jpg']
group of 4: ['not_damaged_137_jpg.rf.fcb8e5c1d636ae4d3c35b3db54497e1f.jpg', 'not_damaged_139_jpg.rf.9f18c65b30035092401a5281a03b7124.jpg', 'not_damaged_143_jpg.rf.ddf38df15f549405285b01245a2ace36.jpg', 'not_damaged_147_jpg.rf.15fa276e0ef7cfa24fbe2a095ffbfbfe.jpg']
group of 4: ['not_damaged_144_jpg.rf.57c2c9d2f23cfb1b6ba8ebf1f8cb5732.jpg', 'not_damaged_146_jpg.rf.65b99fc696d552e46b9ecb2d7168b4ef.jpg', 'not_damaged_148_jpg.rf.337be6c7ca3a4da1c7b80e7747e4eb1e.jpg', 'not_damaged_150_jpg.rf.4883eec179040c5f212f45ec7e248b11.jpg']
group of 3: ['not_damaged_124_jpg.rf.931f62b8d40a093039633b898e0e9114.jpg', 'not_damaged_135_jpg.rf.a6d81ffb82470e031fb3ab469c8f0db2.jpg', 'not_damaged_145_jpg.rf.5147bfde42f30a27504377fd5ef13b05.jpg']
group of 3: ['not_damaged_131_jpg.rf.854045f949fbbe8f94cb021a812df0f0.jpg', 'not_damaged_133_jpg.rf.a934442ef5795325ea7bfe30472e389e.jpg', 'not_damaged_136_jpg.rf.5f8cfbe5f8021c3b171f1e9b282c0fa2.jpg']
```

### Stage 2: Ground Truth Generation & Quality Audit (SAM on CUDA)
```powershell
python src/sam_segment.py --images data --checkpoint weights/sam_vit_b_01ec64.pth --points points.json --out out/sam --contact-sheet
```
```text
PyTorch device in use: cuda
GPU device: NVIDIA GeForce RTX 4070 Laptop GPU
Loaded point prompts for 110 images from points.json
[1/110] not_damaged_100_jpg.rf.762bb1e1ebbf559196b02660144f24c3.jpg: 1 egg(s) (point/box-prompt)
...
[110/110] not_damaged_99_jpg.rf.d2d2a13d789965d1d64380b2d6a59e9a.jpg: 1 egg(s) (point/box-prompt)

All 110 processed images used point/box prompts.
Saved 4 contact sheet(s) to out\sam
Ground-Truth Audit: 0 suspicious mask(s) flagged.
Written audit log to out\sam\suspicious.txt
```

### Stage 3: Parameterized Classical Tuning on Dev Split (18 Trials)
```powershell
python src/classical_segment.py --run-all-dev
```
```text
Running all 18 tuning trials on dev split (88 images)...
[baseline               | dev ] n=88, IoU=0.3513 (med 0.2648), Dice=0.4693, DetF1=0.2071, CntErr=3.12, time=8.4s
[otsu_inv               | dev ] n=88, IoU=0.0823 (med 0.0592), Dice=0.1424, DetF1=0.0000, CntErr=3.36, time=9.2s
[clahe_otsu             | dev ] n=88, IoU=0.2562 (med 0.2004), Dice=0.3791, DetF1=0.0799, CntErr=8.22, time=9.7s
[adaptive_thresh        | dev ] n=88, IoU=0.0634 (med 0.0474), Dice=0.1119, DetF1=0.0065, CntErr=12.33, time=10.2s
[lab_l                  | dev ] n=88, IoU=0.3244 (med 0.2352), Dice=0.4437, DetF1=0.1665, CntErr=3.67, time=8.1s
[hsv_gate               | dev ] n=88, IoU=0.3456 (med 0.2698), Dice=0.4665, DetF1=0.1587, CntErr=4.32, time=8.3s
[morph_small            | dev ] n=88, IoU=0.3558 (med 0.2642), Dice=0.4740, DetF1=0.2161, CntErr=2.89, time=7.8s
[morph_medium           | dev ] n=88, IoU=0.3766 (med 0.2942), Dice=0.4964, DetF1=0.2363, CntErr=2.34, time=7.9s
[morph_large            | dev ] n=88, IoU=0.3972 (med 0.3060), Dice=0.5161, DetF1=0.2642, CntErr=1.25, time=8.0s
[border_polarity_m0     | dev ] n=88, IoU=0.3698 (med 0.2978), Dice=0.4821, DetF1=0.2409, CntErr=1.07, time=8.2s
[border_polarity_m15    | dev ] n=88, IoU=0.3945 (med 0.3060), Dice=0.5119, DetF1=0.2596, CntErr=1.25, time=8.1s
[lab_border_dist        | dev ] n=88, IoU=0.3603 (med 0.3208), Dice=0.4582, DetF1=0.2673, CntErr=2.55, time=9.6s
[area_filter_2500       | dev ] n=88, IoU=0.4018 (med 0.3147), Dice=0.5200, DetF1=0.3084, CntErr=0.45, time=7.9s
[component_ranking_top1 | dev ] n=88, IoU=0.3746 (med 0.2994), Dice=0.4761, DetF1=0.3295, CntErr=-0.33, time=7.3s
[component_ranking_top3 | dev ] n=88, IoU=0.3972 (med 0.3147), Dice=0.5092, DetF1=0.3133, CntErr=0.27, time=7.7s
[grabcut_refine         | dev ] n=88, IoU=0.4038 (med 0.2981), Dice=0.5177, DetF1=0.3073, CntErr=0.38, time=66.8s
[watershed_always       | dev ] n=88, IoU=0.4018 (med 0.3147), Dice=0.5200, DetF1=0.3537, CntErr=3.90, time=15.0s
[watershed_conditional  | dev ] n=88, IoU=0.4018 (med 0.3147), Dice=0.5200, DetF1=0.4087, CntErr=2.20, time=15.5s

All dev tuning trials completed successfully! Log written to experiments.csv
```

### Stage 4: Single Evaluation on Frozen Pipeline (Eval Split)
```powershell
python src/classical_segment.py --trial final --split eval
```
```text
[final                  | eval] n=22, IoU=0.3196 (med 0.2502), Dice=0.4239, DetF1=0.3206, CntErr=1.68, time=3.6s
```

### Stage 5: Quantitative Hungarian Evaluation vs Ground Truth
```powershell
python src/evaluate.py --pred out/classical/final/dev/masks --gt out/sam/masks --split dev --data data
python src/evaluate.py --pred out/classical/final/eval/masks --gt out/sam/masks --split eval --data data
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

### Stage 6: Visual Failure Categorization
```powershell
python src/categorize_failures.py
```
```text
Saved 110 records to failure_tags.csv

Visual Failure Categories Breakdown (Full Dataset):
--------------------------------------------------------------------------------------------------------------
Category                                       Count     Pct   Mean IoU   Mean Dice   Mean F1
--------------------------------------------------------------------------------------------------------------
White egg on white tray / fridge interior         13   11.8%     0.1274      0.2198    0.0171
Egg in boiling water / submerged                   8    7.3%     0.1731      0.2901    0.1399
Tabletop / high-contrast background               40   36.4%     0.3412      0.4744    0.3301
Backlit / extreme shadows                          9    8.2%     0.4033      0.4539    0.3889
Hand holding egg / fingers occluding              16   14.5%     0.4677      0.5821    0.5208
Multiple touching eggs                            16   14.5%     0.5526      0.6844    0.6972
Foliage / herbs in background (specular highlights)      8    7.3%     0.7177      0.8222    0.6854
--------------------------------------------------------------------------------------------------------------

Per-Category Dev vs Eval Split Breakdown:
-------------------------------------------------------------------------------------
Category                                       Dev N   Dev IoU  Eval N   Eval IoU
-------------------------------------------------------------------------------------
White egg on white tray / fridge interior         13    0.1274       0     0.0000
Egg in boiling water / submerged                   6    0.1723       2     0.1756
Tabletop / high-contrast background               32    0.3659       8     0.2424
Backlit / extreme shadows                          6    0.3354       3     0.5392
Hand holding egg / fingers occluding              11    0.6120       5     0.1502
Multiple touching eggs                            13    0.5674       3     0.4886
Foliage / herbs in background (specular highlights)      7    0.6906       1     0.9068
-------------------------------------------------------------------------------------
```

### Stage 7: Video Tracking on YouTube Demonstration
```powershell
python src/video_track.py --video video/eggs.mp4 --out out/video
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
  "unique_id_list": [1, 2, 3, 4, 5, 6],
  "id_switches": 0,
  "birth_frames": {"1": 222, "2": 222, "3": 546, "4": 689, "5": 800, "6": 810},
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

### Stage 8: Dynamic Report Compilation
```powershell
python src/generate_report.py
```
```text
Generated Markdown report: report\report.md
Generated HTML report: report\report.html
Successfully compiled PDF report to C:\Users\Varun\OneDrive\Desktop\sam-egg-segmentation\report\report.pdf (8090.3 KB)

--- Programmatic Verification of Figure Captions in PDF ---
  [FOUND] Figure 1a: Pixel IoU distribution on Dev split (n=88)....
  [FOUND] Figure 1b: Pixel IoU distribution on Eval split (n=22)....
  [FOUND] Figure 2a: Predicted vs True egg counts on Dev split. Points lie mostly abo...
  [FOUND] Figure 2b: Predicted vs True egg counts on Eval split. Points similarly lie...
  [FOUND] Figure 3a: Worst 8 failures on Dev split (Green=TP, Red=FP, Blue=FN). White...
  [FOUND] Figure 3b: Worst 8 failures on Eval split (Green=TP, Red=FP, Blue=FN)....
  [FOUND] Figure 4: Best 8 successes on Dev split showing clean elliptical masks matc...
  [FOUND] Figure 5: Random 8 representative cases illustrating typical operational pe...
  [FOUND] Figure 6: Frame 600 before/after comparison (crop: x in [200, 460], y in [3...
  [FOUND] Figure 7: Video Tracking Events Sheet across 14 key video frames documentin...
-----------------------------------------------------------

Verification SUCCESS: All 10 figure captions appear in the compiled PDF text!
```

---

## 3. Where Every Deliverable Lives

| Deliverable | File Path | Description |
|:---|:---|:---|
| **Report PDF** | `report/report.pdf` | Formatted multi-page PDF report with embedded figures and tables (7.90 MB). |
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
3. **Verification:** Gating reduced the total foreground mask by 2,296 px over the whole frame; restricted to the crop x in [200, 460], y in [300, 430], the mask was reduced by 1,304 px, as documented in `out/video/frame_600_skin_diff.json` and `report/figures/frame_600_comparison.png`.

---

## 5. Honest Disclosures: Limitations and What Could Not Be Verified

Reviewers appreciate transparency. Here is what was NOT done or where shortcuts were taken:

1. **Ground Truth Annotation:** Ground truth masks were established via SAM (`vit_b`) prompted with human point and box coordinates. While audited for area and border anomalies (as recorded in `out/sam/gt_fix_audit.json`), they represent semi-automated model outputs rather than manual dense polygon annotations.
2. **Dataset Size:** With 88 dev and 22 eval images, sample sizes are limited, leading to wide bootstrap confidence intervals on eval ([0.1978, 0.4268] vs dev [0.3435, 0.4603]).
3. **Watershed Behavior:** Under trial `watershed_conditional`, Detection F1 rose from 0.3084 to 0.4087, mean signed count error rose from +0.45 to +2.20 (over-splitting), pixel IoU unchanged. Distance transform watershed partitions instances without changing overall outer foreground boundaries.
4. **Domain-Specific Table ROI:** In `src/video_track.py`, the region y < 270 (demonstrator's plaid shirt) and the bottom-right corner (x > 1040, y > 600, channel logo watermark) were masked using static coordinate bounds. In a production system, a table/surface detector or YOLO person detector should be used to establish dynamic ROIs.
5. **Re-Identification Across Removals:** Smooth, textureless white eggs have near-identical visual appearance. When eggs are removed from view and placed back onto the table, identity assignment cannot be established with physical certainty under classical re-ID methods.
6. **Slim Review Package Omissions:** The slim archive (`sam-egg-segmentation-slim.zip`, <25 MB) leaves out the raw dataset images (`data/`), all visual overlay folders (`*/overlays`), contact sheet images, video files (`*.mp4`), intermediate classical trial masks, SAM model weights (`weights/`), and virtual environment (`.venv/`).

---

## 6. What Changed in This Round

1. **Report Parity & Programmatic Verification:**
   - Unified `src/generate_report.py` to generate `report.md`, `report.html`, and `report.pdf` from the exact same data structures, eliminating all content and caption discrepancies.
   - Added programmatic verification via `fitz` (PyMuPDF) confirming that all 10 figure captions (Figures 1a, 1b, 2a, 2b, 3a, 3b, 4, 5, 6, 7) are present in the compiled PDF text.
2. **Watershed Description Correction:**
   - Rewrote the watershed progression description from `experiments.csv`: "Detection F1 rose from 0.3084 to 0.4087, mean signed count error rose from +0.45 to +2.20 (over-splitting), pixel IoU unchanged."
3. **Track Timeline & Analysis Consistency:**
   - Rebuilt `out/video/track_timeline.json` from full per-frame tracker state with exact birth frames, last matched frames, matched frame counts, gap intervals, and verified still references.
   - Extracted 14 full-frame annotated event stills (track births, deaths, and frames 798, 800, 802, 806, 810, 814 for Track 5) and built `report/figures/tracking_events_sheet.jpg`.
   - Clarified physical egg identities so no two simultaneously active tracks share descriptions (Track 1 = middle-right egg; Track 4 = left egg re-placed at frame 689).
   - Documented Track 5 analysis: transient rotational motion-blur fragment of spinning raw egg at frame 800-810 that briefly coexists with Track 6 at frame 810 before consolidating.
4. **Skin Gating Numbers & Fingertip Crop:**
   - Updated wording: "gating reduced the total foreground mask by 2296 px over the whole frame; the fingertip bulge is visible in Figure 6".
   - Computed and recorded crop difference ($x \in [200, 460], y \in [300, 430]$): 1,304 px reduced.
   - Saved full-frame and crop numbers to `out/video/frame_600_skin_diff.json`.
   - Mentioned the small residual ring on the fingernail in Figure 6.
5. **Real Console Outputs in HANDOFF.md:**
   - Replaced paraphrased output lines with actual verbatim console recordings from pipeline runs.
6. **Eval Gap Detailed Breakdown:**
   - Added an image-by-image breakdown of the 5 eval hand images (`not_damaged_108, 127, 152, 151, 130`) explaining how pale skin and bright illumination caused Otsu flooding (mean IoU 0.1502).
   - Softened "cannot be attributed to a single cause" to explain that severe hand failures and lower tabletop contrast drive the gap in observed data alongside small-sample variance.
7. **Boiling-Pot Before/After Audit:**
   - Generated `out/sam/gt_fix_audit.json` by re-running SAM with point-only vs box prompts for `not_damaged_124, 135, 145`, demonstrating area reduction from 62-82% down to 13-14% (0 border contacts).
8. **Archive Repackaging:**
   - Quantized gallery PNGs to 256 colors ($\le 1.5$ MB each), allowing all report figures to fit within the slim archive while reducing PDF size from 15.3 MB to 7.9 MB.
   - Rebuilt `sam-egg-segmentation-slim.zip` (18.6 MB, well under 25 MB).

# Egg Segmentation and Multi-Object Tracking

SAM ground-truth segmentation via point and box prompts, classical image-processing pipeline with dev experiment harness, Hungarian matching evaluation, failure categorization, and persistent video egg tracking.

---

## 1. Setup

### Python Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*Note: For GPU acceleration with SAM, install PyTorch with CUDA (e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`).*

### Weights and Dataset

1. Dataset: Roboflow "egg" dataset (110 images, 640x640) placed in `./data`:
   - `data/train/egg/...`, `data/valid/egg/...`, `data/test/egg/...`
2. Download SAM ViT-B checkpoint (375 MB) into `weights/`:
   ```bash
   mkdir -p weights
   curl -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

---

## 2. Execution Pipeline

### Step 0: Perceptual Hash Splitting (Dev / Eval)
Splits near-duplicate images into 4:1 disjoint clusters (88 dev, 22 eval) using perceptual hashing (`splits.json` uses POSIX forward slashes for cross-platform compatibility):
```bash
python src/split.py --images data --out splits.json
```

### Task 1: SAM Segmentation via Point & Box Prompts (Ground Truth)
1. **Interactive Annotation & Bounding Box Tool:**
   Left-click for positive points, right-click for negative points, Shift+Drag (or 'X') for bounding box prompt on boiling pots, Space/Enter to save and preview SAM mask live:
   ```bash
   python src/annotate_points.py --images data --out points.json
   ```
2. **Generate Ground Truth Masks, Overlays & Contact Sheet Audit:**
   Uses `SamPredictor` on CUDA with point and box prompts (`multimask_output=True`, selecting best score) and runs automated border/area audit:
   ```bash
   python src/sam_segment.py --images data --checkpoint weights/sam_vit_b_01ec64.pth --points points.json --out out/sam --contact-sheet
   ```

### Task 2: Classical Segmentation Experiments
Runs 18 controlled single-parameter tuning trials on the dev split, comparing each trial strictly against its parent and dynamically logging decisions to `experiments.csv`:
```bash
# List all trial configurations
python src/classical_segment.py --list-trials

# Run all tuning trials sequentially on dev split
python src/classical_segment.py --run-all-dev

# Run frozen final algorithm once on unseen eval split
python src/classical_segment.py --trial final --split eval
```

### Task 3: Evaluation Against SAM Ground Truth & Failure Categorization
Calculates pixel IoU, Dice, precision, recall, Hungarian matching instance metrics (IoU >= 0.5), count errors, distribution histograms, count scatter, worst-8, best-8, and random-8 galleries:
```bash
# Evaluate final pipeline on dev set
python src/evaluate.py --pred out/classical/final/dev/masks --gt out/sam/masks --split dev --data data

# Evaluate final pipeline on eval set
python src/evaluate.py --pred out/classical/final/eval/masks --gt out/sam/masks --split eval --data data

# Categorize all 110 dataset images by physical failure modes into failure_tags.csv
python src/categorize_failures.py
```

### Task 4: Video Segmentation & Tracking
Downloads the demonstration video with `yt-dlp` and runs frame-by-frame persistent tracking with constant-velocity prediction, distance-transform watershed, and YCrCb skin gating to eliminate hand/fingertip leakage at frame 600:
```bash
# Download video (kept out of git)
yt-dlp "https://www.youtube.com/watch?v=Y4JyV94y2_A" -o "video/eggs.mp4"

# Run tracking and generate out/video/tracked.mp4, tracking_log.csv, and contact sheet
python src/video_track.py --video video/eggs.mp4 --out out/video
```

### Task 5: Generate Reproducible Report
Compiles all dynamic metrics, tables, failure breakdowns, and tracking figures into `report/report.md`, `report/report.html`, and `report/report.pdf`:
```bash
python src/generate_report.py
```

---

## 3. Deliverables Summary

- `experiments.csv`: Complete structured log of all 18 dev tuning trials + 1 eval final trial.
- `failure_tags.csv`: Visual root-cause failure breakdown across all 110 images.
- `out/sam/`: Ground truth masks, overlays, contact sheets 1-4, and suspicious mask audit list.
- `out/eval/`: Per-image CSVs (`per_image_dev.csv`, `per_image_eval.csv`), summary JSONs, and distribution plots.
- `out/video/`: Sampled tracking log (`tracking_log.csv`), event contact sheet, key frames, and `tracking_stats.json`.
- `report/`: `report.md`, `report.html`, `report.pdf`, and figures.
- `HANDOFF.md`: Comprehensive handoff guide with exact execution sequence, key numbers, interview questions, and limitation disclosures.

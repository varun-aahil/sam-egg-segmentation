# Egg Segmentation and Tracking

SAM ground-truth segmentation via point prompts, classical image-processing pipeline with dev experiment harness, Hungarian matching evaluation, and persistent video egg tracking.

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

### Task 1: SAM Segmentation via Point Prompts (Ground Truth)
1. **Interactive Annotation (optional if points.json exists):**
   Left-click points on each egg (right-click for background negatives, `n`/space for next, `u` to undo):
   ```bash
   python src/annotate_points.py --images data --out points.json
   ```
2. **Generate Ground Truth Masks & Overlays:**
   Uses `SamPredictor` on CUDA with point prompts (`multimask_output=True`, selecting best score) and generates contact sheets:
   ```bash
   python src/sam_segment.py --images data --checkpoint weights/sam_vit_b_01ec64.pth --points points.json --out out/sam --contact-sheet
   ```

### Task 2: Classical Segmentation Experiments
Runs the 15-trial experiment progression exclusively on the dev split and appends measured metrics, parameters, and dynamic decisions to `experiments.csv`:
```bash
# List all trial configurations
python src/classical_segment.py --list-trials

# Run all 15 trials sequentially on dev split (derives decisions KEEP/DISCARD dynamically)
python src/classical_segment.py --run-all-dev

# Run frozen final algorithm once on eval split
python src/classical_segment.py --trial final --split eval
```

### Task 3: Evaluation Against SAM Ground Truth
Calculates pixel IoU, Dice, precision, recall, Hungarian matching instance metrics (IoU >= 0.5), count errors, distribution histograms, count scatter, and worst-8 failure galleries:
```bash
# Evaluate final pipeline on dev set
python src/evaluate.py --pred out/classical/final/dev/masks --gt out/sam/masks --split dev --data data

# Evaluate final pipeline on eval set
python src/evaluate.py --pred out/classical/final/eval/masks --gt out/sam/masks --split eval --data data
```

### Task 4: Video Segmentation & Tracking
Downloads the demonstration video with `yt-dlp` and runs frame-by-frame persistent tracking with constant-velocity prediction and distance-transform watershed:
```bash
# Download video (kept out of git)
yt-dlp "https://www.youtube.com/watch?v=Y4JyV94y2_A" -o "video/eggs.mp4"

# Run tracking and generate out/video/tracked.mp4 + tracking statistics
python src/video_track.py --video video/eggs.mp4 --out out/video
```

### Generate Reproducible Report
Compiles all experiment logs, evaluation figures, failure analysis, and tracking previews into `report/report.md`, `report/report.html`, and `report/report.pdf`:
```bash
python src/generate_report.py
```

---

## 3. Repository Structure

```
├── data/                     # Ignored: 110 dataset images
├── out/                      # Ignored: generated masks, overlays, video
├── video/                    # Ignored: downloaded eggs.mp4
├── weights/                  # Ignored: SAM vit_b weights
├── report/
│   ├── figures/              # Committed: evaluation plots, failure gallery, tracking stills
│   ├── report.html           # Compiled HTML report
│   ├── report.md             # Markdown report
│   └── report.pdf            # Compiled PDF report
├── src/
│   ├── annotate_points.py    # Task 1: Interactive OpenCV point prompt tool
│   ├── classical_segment.py  # Task 2: Parameterized pipeline & experiment harness
│   ├── contact_sheet.py      # Task 1: Contact sheet generator for review
│   ├── evaluate.py           # Task 3: Hungarian matching & evaluation plots
│   ├── generate_report.py    # Report compiler (HTML/MD/PDF)
│   ├── sam_segment.py        # Task 1: SAM predictor with point prompts
│   ├── split.py              # Step 0: Perceptual hash dev/eval split
│   └── video_track.py        # Task 4: Persistent Hungarian video tracking
├── experiments.csv           # Complete measured 15-trial experiment log
├── points.json               # Annotated point prompts for all 110 images
├── splits.json               # 88 dev / 22 eval image path split (POSIX format)
├── requirements.txt          # Python dependencies
└── README.md
```

---

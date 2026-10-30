"""Generate eval hand overlays figure and compute skin fraction for not_damaged_130."""
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image

data_root = Path('data')
eval_masks_dir = Path('out/classical/final/eval/masks')
sam_masks_dir = Path('out/sam/masks')
figures_dir = Path('report/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

with open('splits.json') as f:
    eval_imgs = json.load(f)['eval']

img_keys = ['108', '127', '130', '151', '152']
mapped = {}
for rel_path in eval_imgs:
    for k in img_keys:
        if f'not_damaged_{k}_' in rel_path:
            mapped[k] = rel_path

rows = []
cell_w, cell_h = 320, 320

# Header
header = np.full((40, cell_w * 3 + 120, 3), 245, dtype=np.uint8)
cv2.putText(header, 'Original Image', (120 + cell_w // 4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
cv2.putText(header, 'SAM Ground Truth', (120 + cell_w + cell_w // 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
cv2.putText(header, 'Classical Prediction', (120 + cell_w * 2 + cell_w // 6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

hand_stats = {}

for k in img_keys:
    rel_path = mapped[k]
    img_path = data_root / rel_path
    stem = Path(rel_path).stem
    sam_path = sam_masks_dir / f'{stem}.png'
    pred_path = eval_masks_dir / f'{stem}.png'
    
    img = cv2.imread(str(img_path))
    sam = cv2.imread(str(sam_path), cv2.IMREAD_UNCHANGED)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
    
    sam_bin = (sam > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    
    inter = np.logical_and(sam_bin, pred_bin).sum()
    union = np.logical_or(sam_bin, pred_bin).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (sam_bin.sum() + pred_bin.sum()) if (sam_bin.sum() + pred_bin.sum()) > 0 else 0.0
    
    # YCrCb skin
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_mask = (cr >= 135) & (cr <= 175) & (cb >= 80) & (cb <= 125)
    fp = np.logical_and(pred_bin == 1, sam_bin == 0)
    fp_count = int(fp.sum())
    skin_in_fp = int(np.logical_and(fp, skin_mask).sum())
    skin_frac = skin_in_fp / fp_count if fp_count > 0 else 0.0
    
    hand_stats[k] = {
        'image': rel_path,
        'iou': round(float(iou), 4),
        'dice': round(float(dice), 4),
        'sam_area': int(sam_bin.sum()),
        'pred_area': int(pred_bin.sum()),
        'fp_count': fp_count,
        'skin_in_fp': skin_in_fp,
        'skin_fraction': round(float(skin_frac), 4),
    }
    
    # SAM overlay: green
    sam_overlay = img.copy()
    sam_overlay[sam_bin == 1] = cv2.addWeighted(img[sam_bin == 1], 0.4, np.full_like(img[sam_bin == 1], (0, 220, 0)), 0.6, 0)
    cnts_sam, _ = cv2.findContours(sam_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sam_overlay, cnts_sam, -1, (0, 255, 0), 2)
    
    # Pred overlay: red/amber
    pred_overlay = img.copy()
    pred_overlay[pred_bin == 1] = cv2.addWeighted(img[pred_bin == 1], 0.4, np.full_like(img[pred_bin == 1], (0, 70, 255)), 0.6, 0)
    cnts_pred, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pred_overlay, cnts_pred, -1, (0, 0, 255), 2)
    
    c1 = cv2.resize(img, (cell_w, cell_h))
    c2 = cv2.resize(sam_overlay, (cell_w, cell_h))
    c3 = cv2.resize(pred_overlay, (cell_w, cell_h))
    
    label_strip = np.full((cell_h, 120, 3), 240, dtype=np.uint8)
    cv2.putText(label_strip, f"#{k}", (15, cell_h // 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(label_strip, f"IoU {iou:.3f}", (10, cell_h // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2)
    
    row = np.hstack([label_strip, c1, c2, c3])
    rows.append(row)

grid = np.vstack([header] + rows)

out_fig = figures_dir / 'eval_hand_overlays.png'
pil_img = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
# Convert to adaptive 256-color palette to minimize size while preserving crisp quality
pil_quantized = pil_img.quantize(colors=256)
pil_quantized.save(out_fig, optimize=True)
print(f"Saved {out_fig}, size: {out_fig.stat().st_size / 1024:.1f} KB")

out_json = Path('out/eval/hand_130_skin_fraction.json')
out_json.parent.mkdir(parents=True, exist_ok=True)
skin_pct = f"{hand_stats['130']['skin_fraction'] * 100:.2f}%"
out_json.write_text(json.dumps({
    'image_id': 'not_damaged_130',
    'filename': mapped['130'],
    'sam_area_px': hand_stats['130']['sam_area'],
    'pred_area_px': hand_stats['130']['pred_area'],
    'false_positive_px': hand_stats['130']['fp_count'],
    'skin_fp_px': hand_stats['130']['skin_in_fp'],
    'skin_fraction_in_fp': hand_stats['130']['skin_fraction'],
    'skin_fraction_pct': skin_pct,
    'all_eval_hand_stats': hand_stats
}, indent=2), encoding="utf-8")
print(f"Saved {out_json}")

"""Generate contact sheets for each failure category to visually audit and retag images."""
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np

def safe_filename(name):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

def main():
    with open("splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    
    # Map stem to relative path
    stem_to_rel = {}
    for s in splits["dev"] + splits["eval"]:
        stem_to_rel[Path(s).stem] = s

    with open("failure_tags.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cats = {}
    for r in rows:
        c = r["category"]
        if c not in cats:
            cats[c] = []
        cats[c].append(r)

    out_dir = Path("report/figures/category_sheets")
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb_w, thumb_h = 240, 240
    pad_bottom = 50

    for cat_name, cat_rows in cats.items():
        n = len(cat_rows)
        cols = min(4, n)
        rows_count = math.ceil(n / cols)

        sheet_w = cols * thumb_w
        sheet_h = rows_count * (thumb_h + pad_bottom)
        sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)

        for idx, r in enumerate(cat_rows):
            col_idx = idx % cols
            row_idx = idx // cols

            stem = r["image"]
            rel_p = stem_to_rel[stem]
            img_p = Path("data") / rel_p
            img = cv2.imread(str(img_p))
            if img is None:
                print(f"Error reading {img_p}")
                continue

            thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            x = col_idx * thumb_w
            y = row_idx * (thumb_h + pad_bottom)

            sheet[y : y + thumb_h, x : x + thumb_w] = thumb

            # Label area
            short_stem = stem.split("_jpg")[0]
            label_text = f"{short_stem} [{r['split']}]"
            metric_text = f"IoU: {float(r['iou']):.3f} | Dice: {float(r['dice']):.3f}"

            cv2.rectangle(sheet, (x, y + thumb_h), (x + thumb_w, y + thumb_h + pad_bottom), (230, 230, 230), -1)
            cv2.putText(sheet, label_text, (x + 8, y + thumb_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            cv2.putText(sheet, metric_text, (x + 8, y + thumb_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)
            cv2.rectangle(sheet, (x, y), (x + thumb_w, y + thumb_h + pad_bottom), (180, 180, 180), 1)

        fname = safe_filename(cat_name) + ".jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"Saved {out_path} ({n} images, {out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()


import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def build_contact_sheets(overlay_dir, out_dir, cols=5, rows=6, thumb_size=200):
    """Tile overlay images into contact sheets for rapid visual review."""
    overlay_paths = sorted(Path(overlay_dir).glob("*.jpg"))
    if not overlay_paths:
        print(f"No JPG overlays found in {overlay_dir}")
        return

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    per_sheet = cols * rows
    num_sheets = math.ceil(len(overlay_paths) / per_sheet)

    print(f"Creating {num_sheets} contact sheet(s) for {len(overlay_paths)} overlays...")
    for s_idx in range(num_sheets):
        batch = overlay_paths[s_idx * per_sheet:(s_idx + 1) * per_sheet]
        grid = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)

        for b_i, p in enumerate(batch):
            r = b_i // cols
            c = b_i % cols
            img = cv2.imread(str(p))
            if img is not None:
                thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                short_name = p.stem.split(".rf.")[0]
                cv2.rectangle(thumb, (0, 0), (thumb_size, 20), (0, 0, 0), -1)
                cv2.putText(thumb, short_name[:24], (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                grid[r * thumb_size:(r + 1) * thumb_size, c * thumb_size:(c + 1) * thumb_size] = thumb

        sheet_path = out_p / f"contact_sheet_{s_idx + 1}.jpg"
        cv2.imwrite(str(sheet_path), grid)
        print(f"  Saved contact sheet: {sheet_path}")


def main():
    ap = argparse.ArgumentParser(description="Tile overlay images into contact sheets")
    ap.add_argument("--overlays", default="out/sam/overlays", help="directory of overlay images")
    ap.add_argument("--out", default="out/sam", help="output directory for contact sheets")
    ap.add_argument("--cols", type=int, default=5, help="columns per sheet")
    ap.add_argument("--rows", type=int, default=6, help="rows per sheet")
    args = ap.parse_args()

    build_contact_sheets(args.overlays, args.out, cols=args.cols, rows=args.rows)


if __name__ == "__main__":
    main()


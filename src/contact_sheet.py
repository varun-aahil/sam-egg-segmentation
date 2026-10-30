"""Tile SAM overlays into contact sheets and audit ground-truth masks for suspicious cases.

Usage:
  python src/contact_sheet.py --overlays out/sam/overlays --masks out/sam/masks --out out/sam
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def audit_mask(mask_path):
    """Audit a single SAM mask for suspicious area or border contact."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return 0, False, ["Failed to read mask"]

    h, w = m.shape[:2]
    num_eggs = len(set(m.flat) - {0})
    fg = m > 0
    fg_area = int(np.sum(fg))
    area_frac = fg_area / (h * w)

    top = int(np.sum(fg[0, :]))
    bottom = int(np.sum(fg[h - 1, :]))
    left = int(np.sum(fg[:, 0]))
    right = int(np.sum(fg[:, w - 1]))
    total_border = top + bottom + left + right
    total_perimeter = 2 * h + 2 * w

    is_suspicious = False
    reasons = []

    # Criterion 1: Area above 40% of the image
    if area_frac > 0.40:
        is_suspicious = True
        reasons.append(f"Area {area_frac:.1%} > 40% ({fg_area}/{h*w} px)")

    # Criterion 2: Touching more than half of the image border
    border_half = total_perimeter // 2
    edge_half_w = w // 2
    edge_half_h = h // 2

    if total_border > border_half:
        is_suspicious = True
        reasons.append(f"Total border contact {total_border} > half perimeter ({border_half})")
    elif top > edge_half_w or bottom > edge_half_w or left > edge_half_h or right > edge_half_h:
        is_suspicious = True
        reasons.append(f"Edge contact > half edge [top={top}/{w}, bot={bottom}/{w}, l={left}/{h}, r={right}/{h}]")

    return num_eggs, is_suspicious, reasons, area_frac, total_border


def build_contact_sheets_and_audit(overlay_dir, mask_dir, out_dir, cols=5, rows=6, thumb_size=200):
    """Tile overlay images into contact sheets and audit for suspicious masks."""
    overlay_paths = sorted(Path(overlay_dir).glob("*.jpg"))
    mask_dir_p = Path(mask_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not overlay_paths:
        print(f"No JPG overlays found in {overlay_dir}")
        return

    histogram = {}
    suspicious_records = []

    # Audit each mask
    audit_results = {}
    for p in overlay_paths:
        mask_path = mask_dir_p / f"{p.stem}.png"
        if mask_path.exists():
            n_eggs, is_susp, reasons, a_frac, b_contact = audit_mask(mask_path)
            histogram[n_eggs] = histogram.get(n_eggs, 0) + 1
            audit_results[p.stem] = (n_eggs, is_susp, reasons, a_frac, b_contact)
            if is_susp:
                suspicious_records.append({
                    "stem": p.stem,
                    "n_eggs": n_eggs,
                    "area_frac": a_frac,
                    "border_contact": b_contact,
                    "reasons": reasons
                })
        else:
            audit_results[p.stem] = (0, True, ["Mask file missing"], 0.0, 0)
            suspicious_records.append({
                "stem": p.stem,
                "n_eggs": 0,
                "area_frac": 0.0,
                "border_contact": 0,
                "reasons": ["Mask file missing"]
            })

    # Write out/sam/suspicious.txt
    susp_txt_path = out_p / "suspicious.txt"
    with open(susp_txt_path, "w", encoding="utf-8") as f:
        f.write("# Suspicious SAM Masks Audit\n")
        f.write("# Flagged if area > 40% of image or touching > half of image border\n")
        f.write(f"# Total flagged: {len(suspicious_records)} / {len(overlay_paths)}\n\n")
        for rec in suspicious_records:
            reasons_str = "; ".join(rec["reasons"])
            f.write(f"{rec['stem']} | eggs={rec['n_eggs']} | area={rec['area_frac']:.1%} | border={rec['border_contact']} | {reasons_str}\n")

    # Render contact sheets
    per_sheet = cols * rows
    num_sheets = math.ceil(len(overlay_paths) / per_sheet)

    print(f"\nBuilding {num_sheets} contact sheet(s) for {len(overlay_paths)} overlays...")
    for s_idx in range(num_sheets):
        batch = overlay_paths[s_idx * per_sheet:(s_idx + 1) * per_sheet]
        grid = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)

        for b_i, p in enumerate(batch):
            r = b_i // cols
            c = b_i % cols
            img = cv2.imread(str(p))
            if img is not None:
                thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                n_eggs, is_susp, _, a_frac, _ = audit_results.get(p.stem, (1, False, [], 0.0, 0))

                short_name = p.stem.split(".rf.")[0][:18]
                header_text = f"{short_name} ({n_eggs}e)"

                if is_susp:
                    # Highlight suspicious in bright red
                    cv2.rectangle(thumb, (0, 0), (thumb_size - 1, thumb_size - 1), (0, 0, 255), 3)
                    cv2.rectangle(thumb, (0, 0), (thumb_size, 22), (0, 0, 200), -1)
                    cv2.putText(thumb, f"!SUSP! {header_text}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                else:
                    cv2.rectangle(thumb, (0, 0), (thumb_size, 20), (0, 0, 0), -1)
                    cv2.putText(thumb, header_text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

                grid[r * thumb_size:(r + 1) * thumb_size, c * thumb_size:(c + 1) * thumb_size] = thumb

        sheet_path = out_p / f"contact_sheet_{s_idx + 1}.jpg"
        cv2.imwrite(str(sheet_path), grid)
        print(f"  Saved contact sheet: {sheet_path}")

    # Print summary and histogram
    total_images = len(overlay_paths)
    print("\n" + "=" * 60)
    print("GROUND TRUTH EGG-COUNT HISTOGRAM:")
    print("=" * 60)
    for k in sorted(histogram.keys()):
        pct = (histogram[k] / total_images) * 100
        print(f"  {k} egg(s): {histogram[k]:3d} images ({pct:5.1f}%)")
    print(f"  Total:    {total_images:3d} images (100.0%)")

    print("\n" + "=" * 60)
    print(f"SUSPICIOUS MASKS AUDIT ({len(suspicious_records)} flagged):")
    print("=" * 60)
    if suspicious_records:
        for rec in suspicious_records:
            print(f"  * {rec['stem']}")
            for r in rec["reasons"]:
                print(f"      - {r}")
        print(f"\nAudit list saved to: {susp_txt_path}")
    else:
        print("  None. All masks pass the area (<40%) and border contact checks.")
    print("=" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Tile overlays into contact sheets and audit ground truth masks")
    ap.add_argument("--overlays", default="out/sam/overlays", help="directory of overlay images")
    ap.add_argument("--masks", default="out/sam/masks", help="directory of label mask PNGs")
    ap.add_argument("--out", default="out/sam", help="output directory for contact sheets & suspicious.txt")
    ap.add_argument("--cols", type=int, default=5, help="columns per sheet")
    ap.add_argument("--rows", type=int, default=6, help="rows per sheet")
    args = ap.parse_args()

    build_contact_sheets_and_audit(args.overlays, args.masks, args.out, cols=args.cols, rows=args.rows)


if __name__ == "__main__":
    main()

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

EXTS = {".jpg", ".jpeg", ".png"}


def load_point_prompts(points_file):
    """Load point prompts keyed by filename."""
    p = Path(points_file)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        print(f"Warning: could not parse {points_file}: {e}")
        return {}


def masks_from_points(predictor, egg_prompts):
    """Predict one mask per egg prompt using SamPredictor (points + optional box)."""
    masks = []
    for egg in egg_prompts:
        pts = egg.get("points", [])
        lbls = egg.get("labels", [])
        box = egg.get("box", None)

        box_arr = np.array(box, dtype=np.float32) if box is not None and len(box) == 4 else None
        coords = np.array(pts, dtype=np.float32) if pts else None
        point_labels = np.array(lbls, dtype=np.int32) if lbls else None

        if coords is not None and len(coords) > 0:
            pred_masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=point_labels,
                box=box_arr,
                multimask_output=True
            )
        elif box_arr is not None:
            pred_masks, scores, _ = predictor.predict(
                box=box_arr,
                multimask_output=True
            )
        else:
            continue

        best_idx = int(np.argmax(scores))
        masks.append(pred_masks[best_idx])
    return masks


def masks_auto(gen, img_rgb, min_frac=0.005, max_frac=0.70):
    """Standard SAM automatic mask fallback without hand-tuned shape/color filters."""
    h, w = img_rgb.shape[:2]
    total_area = h * w
    out = []
    candidates = sorted(gen.generate(img_rgb), key=lambda m: -m["predicted_iou"])
    for m in candidates:
        area = m["area"]
        frac = area / total_area
        if frac < min_frac or frac > max_frac:
            continue
        seg = m["segmentation"]
        # Drop masks that overlap heavily with an already-selected mask
        if any((seg & k).sum() > 0.5 * seg.sum() for k in out):
            continue
        out.append(seg)
    return out


def draw_overlay(img_bgr, labels):
    """Draw colored segmentation masks and boundaries on the image."""
    rng = np.random.RandomState(42)
    over = img_bgr.copy()
    num_eggs = int(labels.max())
    for i in range(1, num_eggs + 1):
        m = labels == i
        if not np.any(m):
            continue
        color = rng.randint(60, 255, 3).tolist()
        over[m] = (0.5 * over[m] + 0.5 * np.array(color)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, color, 2)
    return over


def main():
    ap = argparse.ArgumentParser(description="Generate SAM ground-truth masks using point and box prompts")
    ap.add_argument("--images", default="data", help="root folder of dataset")
    ap.add_argument("--checkpoint", default="weights/sam_vit_b_01ec64.pth", help="SAM checkpoint path")
    ap.add_argument("--points", default="points.json", help="path to point prompts JSON")
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--out", default="out/sam")
    ap.add_argument("--image", help="process only a single image matching this name/stem")
    ap.add_argument("--contact-sheet", action="store_true", help="build contact sheet of all overlays")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device in use: {device}")
    if device == "cuda":
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    predictor = SamPredictor(sam)
    auto_generator = None

    points_data = load_point_prompts(args.points)
    print(f"Loaded point prompts for {len(points_data)} images from {args.points}")

    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    all_paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTS)
    if args.image:
        paths = [p for p in all_paths if args.image.lower() in p.name.lower()]
        if not paths:
            print(f"No image matching '{args.image}' found.")
            return
        print(f"Processing single image: {paths[0].name}")
    else:
        paths = all_paths

    fallback_images = []

    for idx, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if p.name in points_data and points_data[p.name]:
            predictor.set_image(rgb)
            masks = masks_from_points(predictor, points_data[p.name])
            mode = "point/box-prompt"
        else:
            fallback_images.append(p.name)
            if auto_generator is None:
                auto_generator = SamAutomaticMaskGenerator(sam)
            masks = masks_auto(auto_generator, rgb)
            mode = "auto-fallback"

        labels = np.zeros(rgb.shape[:2], np.uint16)
        for i, m in enumerate(masks, 1):
            labels[m] = i

        cv2.imwrite(str(out / "masks" / f"{p.stem}.png"), labels)
        cv2.imwrite(str(out / "overlays" / f"{p.stem}.jpg"), draw_overlay(bgr, labels))
        print(f"[{idx+1}/{len(paths)}] {p.name}: {len(masks)} egg(s) ({mode})")

    if fallback_images:
        print(f"\nImages that used fallback auto mode ({len(fallback_images)}):")
        for fn in fallback_images:
            print(f"  {fn}")
    else:
        print(f"\nAll {len(paths)} processed images used point/box prompts.")

    # Contact sheet and suspicious audit
    try:
        from contact_sheet import build_contact_sheets_and_audit
    except ImportError:
        from src.contact_sheet import build_contact_sheets_and_audit
    build_contact_sheets_and_audit(out / "overlays", out / "masks", out)


if __name__ == "__main__":
    main()

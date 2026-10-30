"""Interactive point and box prompt annotator for SAM ground truth.

Usage:
  python src/annotate_points.py --images data --out points.json
  python src/annotate_points.py --image not_damaged_135
  python src/annotate_points.py --review-suspicious
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Optional PyTorch + SAM imports for immediate re-segmentation
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sam_segment import draw_overlay

EXTS = {".jpg", ".jpeg", ".png"}

PALETTE = [
    (0, 255, 0),      # green
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (0, 165, 255),    # orange
    (203, 192, 255),  # pink
    (255, 144, 30),   # light blue
]


def load_annotations(json_path):
    p = Path(json_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: could not parse {json_path}: {e}")
            return {}
    return {}


def save_annotations(json_path, data):
    Path(json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_sam_for_image(predictor, bgr, egg_prompts):
    """Run SAM predictor on a single image using points and optional boxes."""
    if predictor is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    labels = np.zeros(rgb.shape[:2], dtype=np.uint16)

    for i, egg in enumerate(egg_prompts, 1):
        pts = egg.get("points", [])
        lbls = egg.get("labels", [])
        box = egg.get("box", None)

        box_arr = np.array(box, dtype=np.float32) if box is not None else None
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
        labels[pred_masks[best_idx]] = i

    return labels


def render_display(img_bgr, eggs, current_egg_idx, img_name, idx, total,
                   box_mode=False, drag_box=None, sam_overlay=None):
    if sam_overlay is not None:
        vis = sam_overlay.copy()
    else:
        vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Draw existing boxes and points for all eggs
    for e_idx, egg in enumerate(eggs):
        color = PALETTE[e_idx % len(PALETTE)]
        box = egg.get("box", None)
        if box is not None and len(box) == 4:
            bx0, by0, bx1, by1 = [int(v) for v in box]
            cv2.rectangle(vis, (bx0, by0), (bx1, by1), color, 2)
            cv2.putText(vis, f"Egg {e_idx+1} Box", (bx0 + 4, max(15, by0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        pts = egg.get("points", [])
        lbls = egg.get("labels", [])
        for (x, y), lbl in zip(pts, lbls):
            ix, iy = int(x), int(y)
            if lbl == 1:
                cv2.circle(vis, (ix, iy), 6, color, -1)
                cv2.circle(vis, (ix, iy), 8, (0, 0, 0), 1)
                cv2.putText(vis, f"E{e_idx+1}", (ix + 9, iy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                cv2.drawMarker(vis, (ix, iy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(vis, "bg", (ix + 9, iy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Draw current drag box
    if drag_box is not None:
        dx0, dy0, dx1, dy1 = drag_box
        cv2.rectangle(vis, (dx0, dy0), (dx1, dy1), (0, 255, 255), 2)
        cv2.putText(vis, "Drawing Box...", (dx0 + 4, max(15, dy0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Top info banner (75px)
    banner = np.zeros((75, w, 3), dtype=np.uint8)
    title = f"[{idx+1}/{total}] {img_name}"
    mode_text = "[BOX MODE ACTIVE: drag mouse to set box]" if box_mode else "[POINT MODE: L-click=+pt, R-click=-pt, Shift/Mid-drag=box]"
    info = f"Target: Egg {current_egg_idx+1} | Total eggs: {len(eggs)} | {mode_text}"
    cv2.putText(banner, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    banner_color = (0, 255, 255) if box_mode else (0, 255, 0)
    cv2.putText(banner, info, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner_color, 1)

    # Status / SAM note
    sam_note = "SAM preview active (Press R to refresh)" if sam_overlay is not None else "Press R to run SAM & preview mask on current image"
    cv2.putText(banner, sam_note, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # Bottom instructions bar (32px)
    footer = np.zeros((32, w, 3), dtype=np.uint8)
    help_text = "L-click:+pt | R-click:-pt | Drag:box | X:box mode | D:del box | E:new egg | U:undo | R:run SAM | Space:save+next"
    cv2.putText(footer, help_text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)

    combined = np.vstack([banner, vis, footer])
    return combined


def main():
    ap = argparse.ArgumentParser(description="Interactive point and box prompt annotator for SAM")
    ap.add_argument("--images", default="data", help="dataset root directory")
    ap.add_argument("--out", default="points.json", help="path to save annotations JSON")
    ap.add_argument("--sam-out", default="out/sam", help="SAM output root directory")
    ap.add_argument("--checkpoint", default="weights/sam_vit_b_01ec64.pth", help="SAM weights checkpoint")
    ap.add_argument("--image", help="specific image filename or substring to edit")
    ap.add_argument("--review-suspicious", action="store_true", help="review only suspicious images from suspicious.txt")
    ap.add_argument("--start", type=int, default=-1, help="image index to start at (0-based)")
    ap.add_argument("--no-sam", action="store_true", help="disable live SAM predictor loading")
    args = ap.parse_args()

    root = Path(args.images)
    all_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if not all_paths:
        print(f"No images found in {root}")
        return

    data = load_annotations(args.out)
    sam_out_dir = Path(args.sam_out)
    (sam_out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (sam_out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    # Filter paths if --image or --review-suspicious specified
    if args.image:
        query = args.image.strip().lower()
        paths = [p for p in all_paths if query in p.name.lower()]
        if not paths:
            print(f"No image matching '{args.image}' found.")
            return
        print(f"Found {len(paths)} image(s) matching '{args.image}'.")
    elif args.review_suspicious:
        susp_file = sam_out_dir / "suspicious.txt"
        if not susp_file.exists():
            print(f"No suspicious.txt found at {susp_file}. Please run src/contact_sheet.py first.")
            return
        susp_names = [line.split()[0].strip() for line in susp_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
        paths = [p for p in all_paths if any(sn in p.name for sn in susp_names)]
        if not paths:
            print("No matching suspicious images found in dataset.")
            return
        print(f"Loaded {len(paths)} suspicious image(s) for review from {susp_file}.")
    else:
        paths = all_paths

    total = len(paths)

    # Initialize SAM predictor if requested and available
    predictor = None
    if HAS_SAM and not args.no_sam and Path(args.checkpoint).exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM ({device}) from {args.checkpoint}...")
        try:
            sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
            predictor = SamPredictor(sam)
            print("SAM predictor ready for live re-segmentation.")
        except Exception as e:
            print(f"Warning: could not load SAM: {e}")

    # Determine start index
    if args.start >= 0:
        start_idx = min(args.start, total - 1)
    else:
        start_idx = 0

    print(f"\nAnnotating {total} images. Starting at [{start_idx + 1}/{total}]: {paths[start_idx].name}")
    print("\nControls:")
    print("  Left Click       : Add positive point (foreground)")
    print("  Right Click      : Add negative point (background)")
    print("  Shift / Mid-drag : Draw bounding box [x0, y0, x1, y1]")
    print("  X                : Toggle Box Mode (Left-click drag draws box)")
    print("  D                : Delete bounding box for current egg")
    print("  E or Tab         : Start next egg instance on this image")
    print("  U                : Undo last point or box")
    print("  C                : Clear current egg annotations")
    print("  R                : Re-run SAM on current image and preview mask")
    print("  Space / Enter    : Save changes, re-run SAM if edited, advance to next image")
    print("  S                : Skip image without saving")
    print("  B                : Go back to previous image")
    print("  Q or Esc         : Save annotations and exit\n")

    win_name = "Egg Prompt Annotator (SAM Ground Truth)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    OFFSET_Y = 75  # Top banner height

    state = {
        "current_eggs": [],
        "current_egg_idx": 0,
        "box_mode": False,
        "is_dragging": False,
        "drag_start": None,
        "drag_box": None,
        "sam_overlay": None,
        "is_dirty": False,
        "history": [],  # ("pt", egg_idx, pt, lbl) or ("box", egg_idx, old_box)
        "needs_redraw": True,
    }

    def on_mouse(event, x, y, flags, param):
        img_y = y - OFFSET_Y
        img_x = x
        in_bounds = (0 <= img_y < 640 and 0 <= img_x < 640)

        # Dragging box prompt (via Box mode, Shift key, or Middle button)
        use_box_drag = state["box_mode"] or (flags & cv2.EVENT_FLAG_SHIFTKEY) or (flags & cv2.EVENT_FLAG_MBUTTON) or (event == cv2.EVENT_MBUTTONDOWN)

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN) and in_bounds and use_box_drag:
            state["is_dragging"] = True
            state["drag_start"] = (int(img_x), int(img_y))
            state["drag_box"] = (int(img_x), int(img_y), int(img_x), int(img_y))
            state["needs_redraw"] = True
            return

        elif event == cv2.EVENT_MOUSEMOVE and state["is_dragging"]:
            sx, sy = state["drag_start"]
            cx = max(0, min(639, int(img_x)))
            cy = max(0, min(639, int(img_y)))
            x0, x1 = min(sx, cx), max(sx, cx)
            y0, y1 = min(sy, cy), max(sy, cy)
            state["drag_box"] = (x0, y0, x1, y1)
            state["needs_redraw"] = True
            return

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP) and state["is_dragging"]:
            state["is_dragging"] = False
            sx, sy = state["drag_start"]
            cx = max(0, min(639, int(img_x)))
            cy = max(0, min(639, int(img_y)))
            x0, x1 = min(sx, cx), max(sx, cx)
            y0, y1 = min(sy, cy), max(sy, cy)
            state["drag_box"] = None

            if (x1 - x0) > 8 and (y1 - y0) > 8:
                idx = state["current_egg_idx"]
                while len(state["current_eggs"]) <= idx:
                    state["current_eggs"].append({"points": [], "labels": []})
                old_box = state["current_eggs"][idx].get("box", None)
                state["current_eggs"][idx]["box"] = [x0, y0, x1, y1]
                state["history"].append(("box", idx, old_box))
                state["is_dirty"] = True
                state["sam_overlay"] = None
                print(f"  Egg {idx+1} box set: [{x0}, {y0}, {x1}, y1={y1}]")
            state["needs_redraw"] = True
            return

        # Single point clicks (only when not box dragging)
        if in_bounds and not use_box_drag:
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = state["current_egg_idx"]
                while len(state["current_eggs"]) <= idx:
                    state["current_eggs"].append({"points": [], "labels": []})
                pt = [int(img_x), int(img_y)]
                state["current_eggs"][idx]["points"].append(pt)
                state["current_eggs"][idx]["labels"].append(1)
                state["history"].append(("pt", idx, pt, 1))
                state["is_dirty"] = True
                state["sam_overlay"] = None
                state["needs_redraw"] = True

            elif event == cv2.EVENT_RBUTTONDOWN:
                idx = state["current_egg_idx"]
                while len(state["current_eggs"]) <= idx:
                    state["current_eggs"].append({"points": [], "labels": []})
                pt = [int(img_x), int(img_y)]
                state["current_eggs"][idx]["points"].append(pt)
                state["current_eggs"][idx]["labels"].append(0)
                state["history"].append(("pt", idx, pt, 0))
                state["is_dirty"] = True
                state["sam_overlay"] = None
                state["needs_redraw"] = True

    cv2.setMouseCallback(win_name, on_mouse)

    curr_i = start_idx
    while 0 <= curr_i < total:
        p = paths[curr_i]
        bgr = cv2.imread(str(p))
        if bgr is None:
            curr_i += 1
            continue

        # Load existing egg prompts
        if p.name in data:
            state["current_eggs"] = []
            for e in data[p.name]:
                egg_dict = {
                    "points": [list(pt) for pt in e.get("points", [])],
                    "labels": list(e.get("labels", []))
                }
                if "box" in e and len(e["box"]) == 4:
                    egg_dict["box"] = list(e["box"])
                state["current_eggs"].append(egg_dict)
        else:
            state["current_eggs"] = []

        state["current_egg_idx"] = max(0, len(state["current_eggs"]) - 1)
        state["history"] = []
        state["is_dirty"] = False
        state["sam_overlay"] = None
        state["needs_redraw"] = True

        # If SAM mask already exists, load existing overlay for reference
        existing_ov = sam_out_dir / "overlays" / f"{p.stem}.jpg"
        if existing_ov.exists():
            state["sam_overlay"] = cv2.imread(str(existing_ov))

        while True:
            if state["needs_redraw"]:
                display = render_display(bgr, state["current_eggs"],
                                         state["current_egg_idx"],
                                         p.name, curr_i, total,
                                         box_mode=state["box_mode"],
                                         drag_box=state["drag_box"],
                                         sam_overlay=state["sam_overlay"])
                cv2.imshow(win_name, display)
                state["needs_redraw"] = False

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            # Quit (Q or Esc)
            if key in (ord('q'), ord('Q'), 27):
                cleaned = [e for e in state["current_eggs"] if e.get("points") or e.get("box")]
                if cleaned:
                    data[p.name] = cleaned
                save_annotations(args.out, data)
                print(f"\nSaved annotations to {args.out}. Exiting.")
                cv2.destroyAllWindows()
                return

            # Toggle Box Mode (X)
            elif key in (ord('x'), ord('X')):
                state["box_mode"] = not state["box_mode"]
                mode_str = "ENABLED" if state["box_mode"] else "DISABLED"
                print(f"  Box drawing mode {mode_str}")
                state["needs_redraw"] = True

            # Delete Box on current egg (D)
            elif key in (ord('d'), ord('D')):
                idx = state["current_egg_idx"]
                if idx < len(state["current_eggs"]) and "box" in state["current_eggs"][idx]:
                    old_box = state["current_eggs"][idx].pop("box", None)
                    state["history"].append(("box", idx, old_box))
                    state["is_dirty"] = True
                    state["sam_overlay"] = None
                    print(f"  Deleted box on Egg {idx+1}")
                    state["needs_redraw"] = True

            # Next egg instance (E or Tab)
            elif key in (ord('e'), ord('E'), 9):
                curr_egg = state["current_eggs"][state["current_egg_idx"]] if state["current_egg_idx"] < len(state["current_eggs"]) else None
                if curr_egg and (curr_egg.get("points") or curr_egg.get("box")):
                    state["current_egg_idx"] = len(state["current_eggs"])
                    state["current_eggs"].append({"points": [], "labels": []})
                    state["needs_redraw"] = True
                    print(f"  Started Egg {state['current_egg_idx']+1}")

            # Undo (U)
            elif key in (ord('u'), ord('U')):
                if state["history"]:
                    item = state["history"].pop()
                    if item[0] == "pt":
                        _, h_egg_idx, pt, lbl = item
                        if h_egg_idx < len(state["current_eggs"]):
                            egg = state["current_eggs"][h_egg_idx]
                            if egg["points"]:
                                egg["points"].pop()
                                egg["labels"].pop()
                    elif item[0] == "box":
                        _, h_egg_idx, old_box = item
                        if h_egg_idx < len(state["current_eggs"]):
                            if old_box is not None:
                                state["current_eggs"][h_egg_idx]["box"] = old_box
                            else:
                                state["current_eggs"][h_egg_idx].pop("box", None)

                    while len(state["current_eggs"]) > 1 and not (state["current_eggs"][-1].get("points") or state["current_eggs"][-1].get("box")):
                        state["current_eggs"].pop()
                    state["current_egg_idx"] = max(0, len(state["current_eggs"]) - 1)
                    state["is_dirty"] = True
                    state["sam_overlay"] = None
                    state["needs_redraw"] = True
                    print("  Undo performed")

            # Clear (C)
            elif key in (ord('c'), ord('C')):
                state["current_eggs"] = []
                state["current_egg_idx"] = 0
                state["history"] = []
                state["is_dirty"] = True
                state["sam_overlay"] = None
                state["needs_redraw"] = True
                print("  Cleared all annotations on current image")

            # Run SAM live on current image (R)
            elif key in (ord('r'), ord('R')):
                cleaned = [e for e in state["current_eggs"] if e.get("points") or e.get("box")]
                if cleaned and predictor is not None:
                    print(f"  Running SAM live on {p.name}...")
                    labels = run_sam_for_image(predictor, bgr, cleaned)
                    if labels is not None:
                        ov = draw_overlay(bgr, labels)
                        state["sam_overlay"] = ov
                        cv2.imwrite(str(sam_out_dir / "masks" / f"{p.stem}.png"), labels)
                        cv2.imwrite(str(sam_out_dir / "overlays" / f"{p.stem}.jpg"), ov)
                        print(f"  SAM mask updated: {int(labels.max())} egg(s), fg={np.sum(labels>0)} px")
                elif predictor is None:
                    print("  SAM predictor not loaded. Run without --no-sam or check checkpoint path.")
                state["needs_redraw"] = True

            # Save and advance to Next image (Space, Enter, or N)
            elif key in (32, 13, ord('n'), ord('N')):
                cleaned = [e for e in state["current_eggs"] if e.get("points") or e.get("box")]
                if cleaned:
                    data[p.name] = cleaned
                elif p.name in data:
                    del data[p.name]
                save_annotations(args.out, data)

                # If modified and predictor is available, re-run SAM only for this image
                if state["is_dirty"] and predictor is not None and cleaned:
                    labels = run_sam_for_image(predictor, bgr, cleaned)
                    if labels is not None:
                        ov = draw_overlay(bgr, labels)
                        cv2.imwrite(str(sam_out_dir / "masks" / f"{p.stem}.png"), labels)
                        cv2.imwrite(str(sam_out_dir / "overlays" / f"{p.stem}.jpg"), ov)
                        print(f"  [SAVED & RE-SEGMENTED] {p.name}: {int(labels.max())} egg(s)")

                curr_i += 1
                break

            # Skip without saving (S)
            elif key in (ord('s'), ord('S')):
                curr_i += 1
                break

            # Back to previous image (B or P)
            elif key in (ord('b'), ord('B'), ord('p'), ord('P')):
                cleaned = [e for e in state["current_eggs"] if e.get("points") or e.get("box")]
                if cleaned:
                    data[p.name] = cleaned
                save_annotations(args.out, data)
                curr_i = max(0, curr_i - 1)
                break

    save_annotations(args.out, data)
    print(f"\nAll images processed! Saved annotations to {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

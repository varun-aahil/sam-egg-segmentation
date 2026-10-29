import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}

# Distinct colors for drawing multiple egg instances
PALETTE = [
    (0, 255, 0),    # green
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 165, 255),  # orange
    (203, 192, 255),# pink
    (255, 144, 30), # light blue
]


def load_annotations(json_path):
    p = Path(json_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_annotations(json_path, data):
    Path(json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_display(img_bgr, eggs, current_egg_idx, img_name, idx, total):
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Draw all annotated points
    for e_idx, egg in enumerate(eggs):
        color = PALETTE[e_idx % len(PALETTE)]
        pts = egg.get("points", [])
        lbls = egg.get("labels", [])
        for (x, y), lbl in zip(pts, lbls):
            if lbl == 1:
                # Positive point: colored filled circle + crosshair
                cv2.circle(vis, (int(x), int(y)), 6, color, -1)
                cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 0), 1)
                cv2.putText(vis, f"E{e_idx+1}", (int(x) + 9, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                # Negative point: red cross
                cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(vis, "bg", (int(x) + 9, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Top info banner
    banner = np.zeros((65, w, 3), dtype=np.uint8)
    title = f"[{idx+1}/{total}] {img_name}"
    info = f"Current Target: Egg {current_egg_idx+1} | Total eggs on image: {len(eggs)}"
    cv2.putText(banner, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(banner, info, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

    # Bottom instructions bar
    footer = np.zeros((32, w, 3), dtype=np.uint8)
    help_text = "L-click:+pt | R-click:-pt | E:next egg | U:undo | C:clear | Space:next | B:back | Q:quit"
    cv2.putText(footer, help_text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    combined = np.vstack([banner, vis, footer])
    return combined


def main():
    ap = argparse.ArgumentParser(description="Interactive point prompt annotator for SAM")
    ap.add_argument("--images", default="data", help="dataset root directory")
    ap.add_argument("--out", default="points.json", help="path to save annotations JSON")
    ap.add_argument("--start", type=int, default=-1, help="image index to start at (0-based)")
    args = ap.parse_args()

    root = Path(args.images)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        print(f"No images found in {root}")
        return

    data = load_annotations(args.out)
    total = len(paths)

    # Resume from first unannotated image unless specified
    if args.start >= 0:
        start_idx = min(args.start, total - 1)
    else:
        start_idx = 0
        for i, p in enumerate(paths):
            if p.name not in data:
                start_idx = i
                break
        else:
            print(f"All {total} images are already in {args.out}. Starting at index 0 for review.")
            start_idx = 0

    print(f"Loaded {len(data)} annotated images from {args.out}.")
    print(f"Starting at [{start_idx + 1}/{total}]: {paths[start_idx].name}")
    print("\nControls:")
    print("  Left Click  : Add positive point (foreground)")
    print("  Right Click : Add negative point (background)")
    print("  E or Tab    : Start a new egg instance on this image")
    print("  U           : Undo last point")
    print("  C           : Clear points on current image")
    print("  Space/Enter : Save and advance to next image")
    print("  S           : Skip image without saving changes")
    print("  B           : Go back to previous image")
    print("  Q or Esc    : Save and quit\n")

    win_name = "Egg Point Annotator (SAM Ground Truth)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # State variables for mouse callback
    state = {
        "current_eggs": [],
        "current_egg_idx": 0,
        "needs_redraw": True,
        "history": [],  # (egg_idx, pt, label) for undo
    }

    def on_mouse(event, x, y, flags, param):
        # Account for top banner offset of 65px
        img_y = y - 65
        img_x = x
        if img_y < 0 or img_y >= 640 or img_x < 0 or img_x >= 640:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Positive point
            idx = state["current_egg_idx"]
            while len(state["current_eggs"]) <= idx:
                state["current_eggs"].append({"points": [], "labels": []})
            state["current_eggs"][idx]["points"].append([int(img_x), int(img_y)])
            state["current_eggs"][idx]["labels"].append(1)
            state["history"].append((idx, [int(img_x), int(img_y)], 1))
            state["needs_redraw"] = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Negative point
            idx = state["current_egg_idx"]
            while len(state["current_eggs"]) <= idx:
                state["current_eggs"].append({"points": [], "labels": []})
            state["current_eggs"][idx]["points"].append([int(img_x), int(img_y)])
            state["current_eggs"][idx]["labels"].append(0)
            state["history"].append((idx, [int(img_x), int(img_y)], 0))
            state["needs_redraw"] = True

    cv2.setMouseCallback(win_name, on_mouse)

    curr_i = start_idx
    while 0 <= curr_i < total:
        p = paths[curr_i]
        bgr = cv2.imread(str(p))
        if bgr is None:
            curr_i += 1
            continue

        # Load existing points for this image or start empty
        if p.name in data:
            # Deep copy existing egg list
            state["current_eggs"] = [
                {"points": [list(pt) for pt in e["points"]], "labels": list(e["labels"])}
                for e in data[p.name]
            ]
        else:
            state["current_eggs"] = []

        state["current_egg_idx"] = max(0, len(state["current_eggs"]) - 1)
        state["history"] = []
        state["needs_redraw"] = True

        while True:
            if state["needs_redraw"]:
                display = render_display(bgr, state["current_eggs"],
                                         state["current_egg_idx"],
                                         p.name, curr_i, total)
                cv2.imshow(win_name, display)
                state["needs_redraw"] = False

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            # Quit (Q or Esc)
            if key in (ord('q'), ord('Q'), 27):
                # Clean empty eggs and save
                cleaned = [e for e in state["current_eggs"] if e.get("points")]
                if cleaned:
                    data[p.name] = cleaned
                save_annotations(args.out, data)
                print(f"\nSaved annotations to {args.out}. Exiting.")
                cv2.destroyAllWindows()
                return

            # Next egg (E or Tab)
            elif key in (ord('e'), ord('E'), 9):
                # Ensure current egg has at least one positive point before starting next
                curr_egg = state["current_eggs"][state["current_egg_idx"]] if state["current_egg_idx"] < len(state["current_eggs"]) else None
                if curr_egg and curr_egg["points"]:
                    state["current_egg_idx"] = len(state["current_eggs"])
                    state["current_eggs"].append({"points": [], "labels": []})
                    state["needs_redraw"] = True

            # Undo (U)
            elif key in (ord('u'), ord('U')):
                if state["history"]:
                    h_egg_idx, pt, lbl = state["history"].pop()
                    if h_egg_idx < len(state["current_eggs"]):
                        egg = state["current_eggs"][h_egg_idx]
                        if egg["points"]:
                            egg["points"].pop()
                            egg["labels"].pop()
                    # Clean trailing empty eggs
                    while len(state["current_eggs"]) > 1 and not state["current_eggs"][-1]["points"]:
                        state["current_eggs"].pop()
                    state["current_egg_idx"] = max(0, len(state["current_eggs"]) - 1)
                    state["needs_redraw"] = True

            # Clear (C)
            elif key in (ord('c'), ord('C')):
                state["current_eggs"] = []
                state["current_egg_idx"] = 0
                state["history"] = []
                state["needs_redraw"] = True

            # Save and advance to Next image (Space, Enter, or N)
            elif key in (32, 13, ord('n'), ord('N')):
                cleaned = [e for e in state["current_eggs"] if e.get("points")]
                if cleaned:
                    data[p.name] = cleaned
                elif p.name in data:
                    del data[p.name]
                save_annotations(args.out, data)
                curr_i += 1
                break

            # Skip without saving changes (S)
            elif key in (ord('s'), ord('S')):
                curr_i += 1
                break

            # Back to previous image (B or P)
            elif key in (ord('b'), ord('B'), ord('p'), ord('P')):
                cleaned = [e for e in state["current_eggs"] if e.get("points")]
                if cleaned:
                    data[p.name] = cleaned
                save_annotations(args.out, data)
                curr_i = max(0, curr_i - 1)
                break

    save_annotations(args.out, data)
    print(f"\nAll images reviewed! Saved to {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


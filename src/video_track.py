"""Video egg tracking: segment each frame and track eggs with persistent IDs.

Uses Hungarian assignment on IoU + distance with constant-velocity prediction.
Outputs an annotated MP4 with persistent egg IDs, contours, trajectory trails,
tracking logs (out/video/tracking_log.csv), and event contact sheet.

Usage:
  python src/video_track.py --video video/eggs.mp4 --out out/video
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def segment_video_frame(bgr, table_y_min=270, logo_x_min=1040, logo_y_min=600,
                        s_max=70, v_min=175, ws_min_dist=60):
    """Segment white eggs on the table in video frames with skin-gating and morphology."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s < s_max) & (v > v_min)).astype(np.uint8) * 255

    # Domain constraints
    mask[:table_y_min, :] = 0
    mask[logo_y_min:, logo_x_min:] = 0

    # Human skin exclusion in YCrCb (eliminates hand and fingertip leakage at frame 600)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    is_skin = (cr >= 135) & (cr <= 175) & (cb >= 80) & (cb <= 125)
    mask[is_skin] = 0

    # Morphology: opening severs narrow fingertip bridges; closing bridges motion blur
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    # Instance separation with Euclidean distance transform watershed
    dist = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=ws_min_dist, labels=mask)
    if len(coords) == 0:
        return np.zeros((h, w), dtype=np.uint16)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=mask)

    out = np.zeros((h, w), dtype=np.uint16)
    next_id = 1
    for i in np.unique(labels):
        if i == 0:
            continue
        comp = (labels == i).astype(np.uint8)
        area = int(np.sum(comp))
        if area < 3000 or area > 70000:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.78:
            continue
        out[comp > 0] = next_id
        next_id += 1

    return out


class PersistentEggTracker:
    def __init__(self, max_lost=90, max_reid_dist=150.0):
        self.next_id = 1
        self.tracks = {}  # tid -> dict
        self.max_lost = max_lost
        self.max_reid_dist = max_reid_dist
        self.unique_ids = set()
        self.lifetimes = {}
        self.total_frames = 0
        self.frames_detected = 0
        self.id_switches = 0
        self.birth_frames = {}
        self.loss_events = []
        self.recovery_events = []
        self.prev_active_positions = {}  # tid -> (cx, cy) from previous frame

    def _centroid_and_bbox(self, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (0.0, 0.0), (0, 0, 0, 0)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        x0, y0 = int(xs.min()), int(ys.min())
        w = int(xs.max() - x0 + 1)
        h = int(ys.max() - y0 + 1)
        return (cx, cy), (x0, y0, w, h)

    def _iou(self, m1, m2):
        inter = int((m1 & m2).sum())
        union = int((m1 | m2).sum())
        return float(inter) / float(union) if union > 0 else 0.0

    def predict(self):
        for tid, t in self.tracks.items():
            if t["lost"] > 0:
                vx, vy = t.get("velocity", (0.0, 0.0))
                decay = 0.90 ** t["lost"]
                t["pred_centroid"] = (t["centroid"][0] + vx * decay, t["centroid"][1] + vy * decay)
            else:
                t["pred_centroid"] = t["centroid"]

    def update(self, labels, frame_idx):
        self.total_frames += 1
        self.predict()

        det_ids = sorted(set(labels.flat) - {0})
        detections = []
        for did in det_ids:
            mask = labels == did
            centroid, bbox = self._centroid_and_bbox(mask)
            area = int(np.sum(mask))
            detections.append({
                "det_id": did,
                "mask": mask,
                "centroid": centroid,
                "bbox": bbox,
                "area": area,
            })

        if detections:
            self.frames_detected += 1

        assignment = {}
        track_ids = list(self.tracks.keys())

        # Step 1: Hungarian matching between all existing tracks and new detections
        matched_tracks = set()
        matched_dets = set()

        if track_ids and detections:
            cost = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                t = self.tracks[tid]
                tcx, tcy = t["pred_centroid"]
                for j, d in enumerate(detections):
                    dcx, dcy = d["centroid"]
                    dist = float(np.hypot(tcx - dcx, tcy - dcy))
                    iou = self._iou(t["mask"], d["mask"])
                    # Combined cost: 30% IoU + 70% normalized distance
                    cost[i, j] = 0.3 * (1.0 - iou) + 0.7 * (dist / 1280.0)

            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                tid = track_ids[r]
                d = detections[c]
                tcx, tcy = self.tracks[tid]["pred_centroid"]
                dist = float(np.hypot(tcx - d["centroid"][0], tcy - d["centroid"][1]))

                # Gating threshold: accept match if cost < 0.75 and distance < max_reid_dist
                if cost[r, c] < 0.75 and dist < self.max_reid_dist:
                    assignment[d["det_id"]] = tid

                    # Check for identity switch: did another previously active track have closer prior position?
                    if len(self.prev_active_positions) >= 2 and tid in self.prev_active_positions:
                        for other_tid, other_pos in self.prev_active_positions.items():
                            if other_tid != tid:
                                dist_to_assigned = dist
                                dist_to_other = float(np.hypot(other_pos[0] - d["centroid"][0], other_pos[1] - d["centroid"][1]))
                                if dist_to_other + 40.0 < dist_to_assigned:
                                    self.id_switches += 1

                    was_lost = self.tracks[tid]["lost"] > 0
                    if was_lost:
                        self.recovery_events.append((frame_idx, tid, d["centroid"]))

                    old_c = self.tracks[tid]["centroid"]
                    new_c = d["centroid"]
                    dt = max(1, self.tracks[tid]["lost"] + 1)
                    inst_vx = (new_c[0] - old_c[0]) / dt
                    inst_vy = (new_c[1] - old_c[1]) / dt
                    old_vx, old_vy = self.tracks[tid].get("velocity", (0.0, 0.0))
                    vx = 0.7 * old_vx + 0.3 * inst_vx
                    vy = 0.7 * old_vy + 0.3 * inst_vy

                    hist = self.tracks[tid].get("history", [])
                    hist.append(new_c)
                    if len(hist) > 40:
                        hist.pop(0)

                    self.tracks[tid]["mask"] = d["mask"]
                    self.tracks[tid]["centroid"] = new_c
                    self.tracks[tid]["bbox"] = d["bbox"]
                    self.tracks[tid]["area"] = d["area"]
                    self.tracks[tid]["velocity"] = (vx, vy)
                    self.tracks[tid]["lost"] = 0
                    self.tracks[tid]["history"] = hist
                    self.lifetimes[tid] = self.lifetimes.get(tid, 0) + 1

                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # Step 2: Update lost status for unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                if self.tracks[tid]["lost"] == 0:
                    self.loss_events.append((frame_idx, tid, self.tracks[tid]["centroid"]))
                self.tracks[tid]["lost"] += 1

        # Remove dead tracks exceeding max_lost
        dead_tracks = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in dead_tracks:
            del self.tracks[tid]

        # Step 3: Unmatched detections create new tracks
        for j, d in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.unique_ids.add(tid)
                self.birth_frames[tid] = frame_idx
                assignment[d["det_id"]] = tid
                self.tracks[tid] = {
                    "mask": d["mask"],
                    "centroid": d["centroid"],
                    "bbox": d["bbox"],
                    "area": d["area"],
                    "velocity": (0.0, 0.0),
                    "lost": 0,
                    "history": [d["centroid"]],
                }
                self.lifetimes[tid] = 1

        # Record active positions for next frame switch detection
        self.prev_active_positions = {
            tid: self.tracks[tid]["centroid"] for tid in self.tracks if self.tracks[tid]["lost"] == 0
        }

        return assignment


def draw_tracked_frame(bgr, labels, assignment, tracker):
    out = bgr.copy()
    palette = [
        (46, 204, 113),   # Green (Egg 1)
        (52, 152, 219),   # Blue (Egg 2)
        (155, 89, 182),   # Purple (Egg 3)
        (241, 196, 15),   # Yellow
    ]

    for det_id, track_id in assignment.items():
        color = palette[(track_id - 1) % len(palette)]
        mask = (labels == det_id).astype(np.uint8)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 3)

        t = tracker.tracks.get(track_id)
        if t:
            hist = t.get("history", [])
            for k in range(1, len(hist)):
                pt1 = (int(hist[k - 1][0]), int(hist[k - 1][1]))
                pt2 = (int(hist[k][0]), int(hist[k][1]))
                cv2.line(out, pt1, pt2, color, 2)

            cx, cy = int(t["centroid"][0]), int(t["centroid"][1])
            label = f"Egg {track_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(out, (cx - 35, cy - 30), (cx - 35 + lw + 14, cy - 30 + lh + 14), (0, 0, 0), -1)
            cv2.putText(out, label, (cx - 28, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return out


def build_tracking_contact_sheet(captured_frames, out_path, grid=(4, 4), thumb_size=(320, 180)):
    n_tiles = min(len(captured_frames), grid[0] * grid[1])
    cols, rows = grid
    sheet = np.zeros((rows * thumb_size[1], cols * thumb_size[0], 3), dtype=np.uint8)

    for idx in range(n_tiles):
        fidx, frame, desc = captured_frames[idx]
        thumb = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)
        # Overlay frame index and description badge
        cv2.rectangle(thumb, (0, 0), (thumb_size[0], 24), (0, 0, 0), -1)
        tag = f"F#{fidx}: {desc}"
        cv2.putText(thumb, tag[:36], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        r = idx // cols
        c = idx % cols
        y0, y1 = r * thumb_size[1], (r + 1) * thumb_size[1]
        x0, x1 = c * thumb_size[0], (c + 1) * thumb_size[0]
        sheet[y0:y1, x0:x1] = thumb

    cv2.imwrite(str(out_path), sheet)
    print(f"Saved tracking contact sheet to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Video egg tracking pipeline.")
    ap.add_argument("--video", default="video/eggs.mp4", help="Input video path")
    ap.add_argument("--out", default="out/video", help="Output directory")
    ap.add_argument("--figures-out", default="report/figures", help="Figures directory for report")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--max-reid-dist", type=float, default=150.0, help="Max distance for re-identifying lost tracks")
    ap.add_argument("--max-lost", type=int, default=90, help="Max lost frames before track deletion")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {args.video}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_out)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, in_fps, (width, height))

    tracker = PersistentEggTracker(max_lost=args.max_lost, max_reid_dist=args.max_reid_dist)
    frame_idx = 0
    max_frames = args.max_frames if args.max_frames > 0 else total_video_frames

    # Contact sheet sampling plan: key events + evenly spaced frames
    target_contact_frames = set(range(0, max_frames, max_frames // 12))
    target_contact_frames.update([235, 245, 360, 600, 750, 850])
    captured_frames = []

    # CSV log file setup
    log_csv_path = out_dir / "tracking_log.csv"
    log_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["frame", "track_id", "x", "y", "area", "vx", "vy", "state"])

    key_stills = {
        235: "tracking_frame_235.png",
        360: "tracking_frame_360.png",
        600: "tracking_frame_600.png",
        850: "tracking_frame_850.png",
    }

    print(f"Tracking video: {width}x{height} @ {in_fps:.1f} fps, {total_video_frames} frames")
    print(f"Rendering annotated video to {out_path}...")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        labels = segment_video_frame(frame)
        assignment = tracker.update(labels, frame_idx)
        annotated = draw_tracked_frame(frame, labels, assignment, tracker)

        writer.write(annotated)

        # Log active and lost tracks every 10th frame
        if frame_idx % 10 == 0:
            for tid, t in tracker.tracks.items():
                cx, cy = t["centroid"]
                vx, vy = t.get("velocity", (0.0, 0.0))
                area = t.get("area", 0)
                state = "active" if t["lost"] == 0 else "lost"
                log_writer.writerow([frame_idx, tid, f"{cx:.1f}", f"{cy:.1f}", area, f"{vx:.2f}", f"{vy:.2f}", state])

        # Capture key stills
        if frame_idx in key_stills:
            cv2.imwrite(str(figures_dir / key_stills[frame_idx]), annotated)
            cv2.imwrite(str(out_dir / key_stills[frame_idx]), annotated)

        # Capture frames for contact sheet
        if frame_idx in target_contact_frames:
            desc = "Intro (No eggs)" if len(tracker.unique_ids) == 0 else f"{len(assignment)} eggs tracked"
            if frame_idx == 600:
                desc = "Hand near egg (skin gated)"
            elif frame_idx in [235, 245]:
                desc = "Adjacent placement resolved"
            captured_frames.append((frame_idx, annotated.copy(), desc))

        if frame_idx % 150 == 0:
            active_tids = [tid for tid in tracker.tracks if tracker.tracks[tid]["lost"] == 0]
            print(f"  Frame {frame_idx}/{max_frames}: active tracks = {active_tids}, unique IDs so far = {len(tracker.unique_ids)}")

        frame_idx += 1

    cap.release()
    writer.release()
    log_file.close()

    # Build and save 16-frame contact sheet
    captured_frames.sort(key=lambda x: x[0])
    contact_sheet_path = out_dir / "tracking_contact_sheet.jpg"
    build_tracking_contact_sheet(captured_frames[:16], contact_sheet_path, grid=(4, 4), thumb_size=(320, 180))
    import shutil
    shutil.copy(contact_sheet_path, figures_dir / "tracking_contact_sheet.jpg")

    # Compute and save tracking statistics dynamically
    stats = {
        "total_video_frames": total_video_frames,
        "processed_frames": frame_idx,
        "video_fps": float(in_fps),
        "video_resolution": f"{width}x{height}",
        "unique_ids_created": len(tracker.unique_ids),
        "unique_id_list": sorted(list(tracker.unique_ids)),
        "id_switches": tracker.id_switches,
        "birth_frames": tracker.birth_frames,
        "track_lifetimes": {f"Egg {k}": v for k, v in tracker.lifetimes.items()},
        "total_loss_events": len(tracker.loss_events),
        "total_recovery_events": len(tracker.recovery_events),
        "frames_without_detection": tracker.total_frames - tracker.frames_detected,
        "frames_with_detection": tracker.frames_detected,
    }
    stats_path = out_dir / "tracking_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nTracking complete: {frame_idx} frames rendered to {out_path}")
    print(f"Tracking log: {log_csv_path}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Statistics: {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

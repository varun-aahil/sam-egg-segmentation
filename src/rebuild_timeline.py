import json
from pathlib import Path
import cv2
import numpy as np
try:
    from video_track import segment_video_frame, PersistentEggTracker, draw_tracked_frame
except ImportError:
    from src.video_track import segment_video_frame, PersistentEggTracker, draw_tracked_frame


def main():
    cap = cv2.VideoCapture("video/eggs.mp4")
    tracker = PersistentEggTracker(max_lost=90, max_reid_dist=150.0)

    target_frames = {222, 507, 546, 656, 689, 751, 798, 800, 802, 806, 810, 814, 999, 1004}

    out_events = Path("out/video/events")
    out_events.mkdir(parents=True, exist_ok=True)
    rep_events = Path("report/figures/events")
    rep_events.mkdir(parents=True, exist_ok=True)

    saved_stills = {}
    per_track_records = {}
    frame_idx = 0

    stills_for_sheet = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        labels = segment_video_frame(frame)
        assignment = tracker.update(labels, frame_idx)
        assigned_tids = set(assignment.values())

        for tid in tracker.unique_ids:
            if tid not in per_track_records:
                per_track_records[tid] = []
            is_matched = tid in assigned_tids
            if tid in tracker.tracks:
                cx, cy = tracker.tracks[tid]["centroid"]
                area = tracker.tracks[tid]["area"]
                lost = tracker.tracks[tid]["lost"]
            else:
                cx, cy, area = None, None, None
                lost = -1
            per_track_records[tid].append({
                "frame": frame_idx,
                "matched": is_matched,
                "lost": lost,
                "centroid": (cx, cy) if cx is not None else None,
                "area": area
            })

        if frame_idx in target_frames:
            annotated = draw_tracked_frame(frame, labels, assignment, tracker)
            fn = f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(out_events / fn), annotated)
            cv2.imwrite(str(rep_events / fn), annotated)
            saved_stills[frame_idx] = fn
            stills_for_sheet.append((frame_idx, annotated))
            print(f"Saved still {fn} (matched: {sorted(list(assigned_tids))})")

        frame_idx += 1
    cap.release()

    # Build contact sheet of the 14 event stills (3 cols x 5 rows or 4 cols x 4 rows)
    stills_for_sheet.sort(key=lambda x: x[0])
    cols = 4
    rows = (len(stills_for_sheet) + cols - 1) // cols
    thumb_w, thumb_h = 320, 180
    sheet = np.zeros((rows * (thumb_h + 30), cols * thumb_w, 3), dtype=np.uint8)

    for idx, (f_num, img) in enumerate(stills_for_sheet):
        r = idx // cols
        c = idx % cols
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        y0 = r * (thumb_h + 30)
        x0 = c * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        cv2.putText(sheet, f"Frame {f_num}", (x0 + 10, y0 + thumb_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite("out/video/tracking_events_sheet.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite("report/figures/tracking_events_sheet.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Created tracking_events_sheet.jpg ({sheet.shape[1]}x{sheet.shape[0]})")

    # Let's inspect the positions and physical descriptions for all tracks
    print("\n--- Summary per track ---")
    for tid in sorted(per_track_records.keys()):
        recs = per_track_records[tid]
        matched = [r for r in recs if r["matched"]]
        m_frames = [r["frame"] for r in matched]
        print(f"Track {tid}: first={m_frames[0]}, last_matched={m_frames[-1]}, count={len(m_frames)}")
        c_start = [r["centroid"] for r in matched if r["frame"] == m_frames[0]][0]
        c_end = [r["centroid"] for r in matched if r["frame"] == m_frames[-1]][0]
        print(f"   Start centroid: {c_start}, End centroid: {c_end}")


if __name__ == "__main__":
    main()

"""Build slim submission archive under 25 MB containing code, logs, reports, and audit artifacts."""
import os
import zipfile
from pathlib import Path


def main():
    zip_path = Path("sam-egg-segmentation-slim.zip")
    if zip_path.exists():
        zip_path.unlink()

    report_figures_used = [
        "iou_dist_dev.png",
        "iou_dist_eval.png",
        "count_scatter_dev.png",
        "count_scatter_eval.png",
        "failure_gallery_dev.png",
        "failure_gallery_eval.png",
        "success_gallery_dev.png",
        "random_gallery_dev.png",
        "eval_hand_overlays.png",
        "frame_600_comparison.png",
        "tracking_events_sheet.jpg",
    ]

    def should_include(rel):
        if rel in [
            "README.md",
            "requirements.txt",
            ".gitignore",
            "splits.json",
            "points.json",
            "experiments.csv",
            "failure_tags.csv",
            "tag_changes.md",
            "HANDOFF.md",
        ]:
            return True
        if rel.startswith("src/"):
            return True
        if rel.startswith("logs/"):
            return True
        if rel in ["report/report.pdf", "report/report.md", "report/report.html"]:
            return True
        if rel.startswith("report/figures/category_sheets/"):
            return True
        if rel.startswith("report/figures/"):
            return Path(rel).name in report_figures_used
        if rel in [
            "out/eval/dev/summary.json",
            "out/eval/dev/per_image.csv",
            "out/eval/eval/summary.json",
            "out/eval/eval/per_image.csv",
            "out/eval/split_leakage_audit.txt",
            "out/eval/hand_130_skin_fraction.json",
            "out/eval/bootstrap_ci.json",
            "out/eval/per_category_dev_eval.json",
            "out/eval/composition_check.json",
            "out/eval/per_image_dev.csv",
            "out/eval/per_image_eval.csv",
            "out/eval/summary_dev.json",
            "out/eval/summary_eval.json",
        ]:
            return True
        if rel in [
            "out/video/tracking_stats.json",
            "out/video/tracking_log.csv",
            "out/video/track_timeline.json",
            "out/video/known_tracking_failures.json",
            "out/video/frame_600_skin_diff.json",
            "out/video/skin_gating_egg_loss.json",
        ]:
            return True
        if rel in ["out/sam/suspicious.txt", "out/sam/gt_fix_audit.json"]:
            return True
        if rel.startswith("out/sam/masks/") and rel.endswith(".png"):
            return True
        if rel.startswith("out/classical/final/dev/masks/") and rel.endswith(".png"):
            return True
        if rel.startswith("out/classical/final/eval/masks/") and rel.endswith(".png"):
            return True
        return False

    files_to_zip = []
    for root, dirs, files in os.walk("."):
        for f in files:
            full_p = Path(root) / f
            rel = full_p.relative_to(".").as_posix()
            if should_include(rel):
                files_to_zip.append((full_p, rel))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full_p, rel in files_to_zip:
            zf.write(full_p, arcname=rel)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created {zip_path}: {zip_size_mb:.2f} MB ({len(files_to_zip)} files)")

    # Also build report-only.zip
    rep_zip = Path("report-only.zip")
    with zipfile.ZipFile(rep_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write("report/report.pdf", arcname="report/report.pdf")
    print(f"Created {rep_zip}: {rep_zip.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()


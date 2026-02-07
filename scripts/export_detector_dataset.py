from __future__ import annotations

"""Export reviewed snapshot labels into YOLO training dataset format."""

import argparse
import hashlib
import shutil
import sqlite3
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse CLI options for dataset export."""
    parser = argparse.ArgumentParser(description="Export reviewed detector feedback to YOLO dataset format")
    parser.add_argument("--db-path", default="data/faces.db")
    parser.add_argument("--output-dir", default="data/detector_dataset")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true", help="Delete output-dir before export")
    return parser.parse_args()


def choose_split(sample_key: str, val_ratio: float) -> str:
    """Deterministically assign sample to train/val split."""
    digest = hashlib.sha1(sample_key.encode("utf-8")).hexdigest()
    v = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if v < val_ratio else "train"


def ensure_dirs(base: Path) -> None:
    """Create YOLO expected directory structure."""
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(base: Path) -> None:
    """Write YOLO `dataset.yaml` for one-class person training."""
    content = (
        f"path: {base.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
    )
    (base / "dataset.yaml").write_text(content, encoding="utf-8")


def yolo_line(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> str:
    """Convert pixel box to normalized YOLO annotation line."""
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    """Run export from `snapshot_reviews` into YOLO images/labels layout."""
    args = parse_args()
    db_path = Path(args.db_path)
    out = Path(args.output_dir)

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    if args.clean and out.exists():
        shutil.rmtree(out)

    ensure_dirs(out)
    write_dataset_yaml(out)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS snapshot_reviews (
            source_image TEXT PRIMARY KEY,
            detector_label TEXT CHECK(detector_label IN ('person', 'no_person') OR detector_label IS NULL),
            person_id TEXT,
            notes TEXT,
            reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    rows = conn.execute(
        """
        SELECT source_image, detector_label AS label
        FROM snapshot_reviews
        WHERE detector_label IN ('person', 'no_person')
        ORDER BY source_image ASC
        """
    ).fetchall()
    conn.close()

    model = YOLO(args.model)

    exported = 0
    positives = 0
    negatives = 0
    skipped_missing = 0
    skipped_no_boxes = 0

    for row in rows:
        image_path = Path(str(row["source_image"]))
        label = str(row["label"])

        if not image_path.exists():
            skipped_missing += 1
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            skipped_missing += 1
            continue

        sample_key = str(image_path.resolve())
        stable_id = hashlib.sha1(sample_key.encode("utf-8")).hexdigest()[:12]
        split = choose_split(sample_key, args.val_ratio)
        image_out = out / "images" / split / f"s_{stable_id}_{image_path.name}"
        label_out = out / "labels" / split / f"s_{stable_id}_{image_path.stem}.txt"

        shutil.copy2(image_path, image_out)

        label_lines: list[str] = []
        if label == "person":
            results = model.predict(frame, verbose=False, classes=[0], conf=args.confidence)
            if results and results[0].boxes is not None:
                h, w = frame.shape[:2]
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    label_lines.append(yolo_line(x1, y1, x2, y2, w, h))

            if not label_lines:
                # Positive review without model box currently cannot be converted
                # to a supervised box label yet.
                if image_out.exists():
                    image_out.unlink()
                skipped_no_boxes += 1
                continue
            positives += 1
        else:
            negatives += 1

        label_out.write_text("\n".join(label_lines), encoding="utf-8")
        exported += 1

    print(f"Reviewed rows: {len(rows)}")
    print(f"Exported samples: {exported}")
    print(f"  Positives: {positives}")
    print(f"  Negatives: {negatives}")
    print(f"Skipped missing/corrupt images: {skipped_missing}")
    print(f"Skipped positives without boxes: {skipped_no_boxes}")
    print(f"Dataset dir: {out}")
    print(f"YAML: {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

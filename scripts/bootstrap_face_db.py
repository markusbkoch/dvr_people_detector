from __future__ import annotations

"""Bootstrap face identity samples from snapshots.

This script can rebuild/extend `face_samples` by:
- honoring snapshot review feedback (`snapshot_reviews`)
- skipping `no_person` snapshots
- clustering unassigned faces by embedding similarity
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.face_db import FaceDatabase
from app.face_rules import FaceEmbeddingEngine


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_camera_id(filename: str) -> int | None:
    """Extract channel/camera id prefix from snapshot file name."""
    head = filename.split("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for bootstrap run."""
    parser = argparse.ArgumentParser(description="Bootstrap face database from snapshots using review feedback.")
    parser.add_argument("--snapshot-dir", default="data/snapshots", help="Directory containing snapshot image files")
    parser.add_argument("--db-path", default="data/faces.db", help="SQLite output path")
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.84,
        help="Cosine threshold used to assign a face to an existing auto person",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear persons/face_samples before bootstrapping",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for destructive actions like --rebuild",
    )
    return parser.parse_args()


def _load_snapshot_reviews(db_path: Path) -> dict[str, dict[str, str]]:
    """Load snapshot-level detector/person review metadata keyed by source path."""
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
        "SELECT source_image, COALESCE(detector_label, '') AS detector_label, COALESCE(person_id, '') AS person_id FROM snapshot_reviews"
    ).fetchall()
    conn.close()

    review_by_image: dict[str, dict[str, str]] = {}
    for row in rows:
        source = str(Path(str(row["source_image"])).resolve())
        review_by_image[source] = {
            "detector_label": str(row["detector_label"] or "").strip(),
            "person_id": str(row["person_id"] or "").strip(),
        }
    return review_by_image


def _norm(vec):
    """Return L2-normalized embedding vector."""
    n = float((vec**2).sum() ** 0.5)
    if n <= 0:
        return vec
    return vec / n


def main() -> None:
    """Execute bootstrap flow and print summary metrics."""
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    if not snapshot_dir.exists():
        raise SystemExit(f"Snapshot dir not found: {snapshot_dir}")

    db = FaceDatabase(Path(args.db_path))
    engine = FaceEmbeddingEngine()

    if args.rebuild:
        if not args.yes:
            print("WARNING: --rebuild will permanently delete existing face classifications.")
            print("The following tables will be cleared before bootstrap:")
            print("  - persons")
            print("  - face_samples")
            print("  - detector_reviews")
            print(f"Database: {Path(args.db_path).resolve()}")
            answer = input("Type 'REBUILD' to continue (or anything else to cancel): ").strip()
            if answer != "REBUILD":
                db.close()
                print("Cancelled. No data was deleted.")
                return
        with db._lock:
            db._conn.execute("DELETE FROM detector_reviews")
            db._conn.execute("DELETE FROM face_samples")
            db._conn.execute("DELETE FROM persons")
            db._conn.commit()

    review_by_image = _load_snapshot_reviews(Path(args.db_path))

    # person_id -> (centroid, count)
    centroids: dict[str, tuple[any, int]] = {}

    imported_samples = 0
    skipped_existing = 0
    skipped_no_person = 0
    skipped_no_face = 0
    created_people = 0

    existing_sources: set[str] = set()
    with db._lock:
        rows = db._conn.execute("SELECT source_image FROM face_samples").fetchall()
    for row in rows:
        source = row[0]
        if source:
            existing_sources.add(str(Path(source).resolve()))

    files = sorted([p for p in snapshot_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    for image_path in files:
        resolved_image = str(image_path.resolve())
        if resolved_image in existing_sources:
            skipped_existing += 1
            continue

        review = review_by_image.get(resolved_image, {"detector_label": "", "person_id": ""})
        detector_label = review["detector_label"].lower()
        assigned_person = review["person_id"].strip()

        if detector_label == "no_person":
            skipped_no_person += 1
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        embedding, quality = engine.best_face_embedding(frame)
        if embedding is None:
            skipped_no_face += 1
            continue
        embedding = _norm(embedding)

        person_id = ""

        if assigned_person:
            person_id = assigned_person
            db.upsert_person(person_id=person_id, display_name=person_id)
        else:
            best_person = ""
            best_sim = -1.0
            for existing_person, (centroid, _count) in centroids.items():
                sim = float((embedding * centroid).sum())
                if sim > best_sim:
                    best_sim = sim
                    best_person = existing_person

            if best_person and best_sim >= args.cluster_threshold:
                person_id = best_person
            else:
                created_people += 1
                person_id = f"person_{created_people:04d}"
                while person_id in centroids:
                    created_people += 1
                    person_id = f"person_{created_people:04d}"
                db.upsert_person(person_id=person_id, display_name=person_id)

        if person_id in centroids:
            centroid, count = centroids[person_id]
            new_count = count + 1
            new_centroid = _norm(((centroid * count) + embedding) / float(new_count))
            centroids[person_id] = (new_centroid, new_count)
        else:
            centroids[person_id] = (embedding, 1)

        db.add_face_sample(
            person_id=person_id,
            embedding=embedding,
            source_image=resolved_image,
            camera_id=parse_camera_id(image_path.name),
            captured_at=None,
            quality_score=quality,
            face_box="",
        )

        # If marked as person but without assignment, keep review row as person.
        if detector_label == "person" and not assigned_person:
            with db._lock:
                db._conn.execute(
                    """
                    INSERT INTO snapshot_reviews (source_image, detector_label, person_id, reviewed_at)
                    VALUES (?, 'person', ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(source_image) DO UPDATE SET person_id = excluded.person_id, reviewed_at = CURRENT_TIMESTAMP
                    """,
                    (resolved_image, person_id),
                )
                db._conn.commit()

        existing_sources.add(resolved_image)
        imported_samples += 1

    counts = db.person_sample_counts()
    db.close()

    print(f"Scanned files: {len(files)}")
    print(f"Imported face samples: {imported_samples}")
    print(f"Skipped existing samples: {skipped_existing}")
    print(f"Skipped no-person labels: {skipped_no_person}")
    print(f"Skipped no-face detections: {skipped_no_face}")
    print(f"People created: {len(counts)}")
    for person_id, count in sorted(counts.items()):
        print(f"  - {person_id}: {count} samples")


if __name__ == "__main__":
    main()

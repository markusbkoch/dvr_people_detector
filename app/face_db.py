from __future__ import annotations

"""SQLite storage and matching utilities for face embeddings."""

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class FaceMatch:
    """Best-match result returned by embedding lookup."""

    person_id: Optional[str]
    similarity: float


class FaceDatabase:
    """Thread-safe SQLite access layer for face samples and identities."""

    def __init__(self, db_path: Path) -> None:
        """Open database connection and ensure schema exists."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        """Create required tables/indexes if they do not exist."""
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    source_image TEXT,
                    camera_id INTEGER,
                    captured_at TEXT,
                    quality_score REAL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    face_box TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id)
                );

                CREATE TABLE IF NOT EXISTS detector_reviews (
                    sample_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL CHECK(label IN ('person', 'no_person')),
                    notes TEXT,
                    reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(sample_id) REFERENCES face_samples(sample_id)
                );

                CREATE INDEX IF NOT EXISTS idx_face_samples_person_id ON face_samples(person_id);
                CREATE INDEX IF NOT EXISTS idx_detector_reviews_label ON detector_reviews(label);
                """
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def upsert_person(self, person_id: str, display_name: Optional[str] = None) -> None:
        """Insert/update person metadata while keeping existing names when absent."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO persons (person_id, display_name) VALUES (?, ?)
                ON CONFLICT(person_id) DO UPDATE SET display_name = COALESCE(excluded.display_name, persons.display_name)
                """,
                (person_id, display_name),
            )
            self._conn.commit()

    def add_face_sample(
        self,
        person_id: str,
        embedding: np.ndarray,
        source_image: str,
        camera_id: Optional[int],
        captured_at: Optional[str],
        quality_score: float,
        face_box: str,
    ) -> None:
        """Persist one normalized face embedding with source metadata."""
        emb = embedding.astype(np.float32)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO face_samples (
                    person_id, source_image, camera_id, captured_at, quality_score,
                    embedding, embedding_dim, face_box
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    source_image,
                    camera_id,
                    captured_at,
                    quality_score,
                    sqlite3.Binary(emb.tobytes()),
                    int(emb.shape[0]),
                    face_box,
                ),
            )
            self._conn.commit()

    def person_sample_counts(self) -> Dict[str, int]:
        """Return `{person_id: sample_count}` for reporting and QA."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, COUNT(*) FROM face_samples GROUP BY person_id"
            ).fetchall()
        return {str(person_id): int(count) for person_id, count in rows}

    def person_centroids(self, min_samples: int = 1) -> Dict[str, np.ndarray]:
        """Compute normalized centroid embedding per person."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, embedding, embedding_dim FROM face_samples"
            ).fetchall()

        grouped: Dict[str, list[np.ndarray]] = {}
        for person_id, embedding_blob, dim in rows:
            vec = np.frombuffer(embedding_blob, dtype=np.float32)
            if vec.shape[0] != int(dim):
                continue
            grouped.setdefault(str(person_id), []).append(vec)

        centroids: Dict[str, np.ndarray] = {}
        for person_id, vectors in grouped.items():
            if len(vectors) < min_samples:
                continue
            centroid = np.mean(np.stack(vectors, axis=0), axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                continue
            centroids[person_id] = centroid / norm
        return centroids

    def match_embedding(self, embedding: np.ndarray, threshold: float, min_samples: int = 1) -> FaceMatch:
        """Match an embedding to the closest centroid using cosine similarity."""
        centroids = self.person_centroids(min_samples=min_samples)
        if not centroids:
            return FaceMatch(person_id=None, similarity=0.0)

        query = embedding.astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return FaceMatch(person_id=None, similarity=0.0)
        query = query / query_norm

        best_person: Optional[str] = None
        best_sim = -1.0

        for person_id, centroid in centroids.items():
            sim = float(np.dot(query, centroid))
            if sim > best_sim:
                best_sim = sim
                best_person = person_id

        if best_person is None or best_sim < threshold:
            return FaceMatch(person_id=None, similarity=max(0.0, best_sim))
        return FaceMatch(person_id=best_person, similarity=best_sim)

from __future__ import annotations

"""Face embedding, identity matching, and notification-policy helpers."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

import cv2
import numpy as np

from app.face_db import FaceDatabase

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Identity lookup result for one frame."""

    person_id: Optional[str]
    confidence: float


class InsightFaceEngine:
    """InsightFace/ArcFace-based face detection and embedding engine.
    
    Uses RetinaFace for detection and ArcFace for 512-dimensional embeddings.
    Much more accurate than baseline, especially for varying angles/lighting.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1) -> None:
        """Initialize InsightFace model.
        
        Args:
            model_name: Model pack name (buffalo_l, buffalo_s, etc.)
            ctx_id: -1 for CPU, 0+ for GPU
        """
        self._app = None
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._init_attempted = False
        self._init_error: Optional[str] = None

    def _lazy_init(self) -> bool:
        """Lazy-load InsightFace models on first use."""
        if self._app is not None:
            return True
        if self._init_attempted:
            return False
            
        self._init_attempted = True
        try:
            from insightface.app import FaceAnalysis
            
            self._app = FaceAnalysis(
                name=self._model_name,
                providers=['CPUExecutionProvider'] if self._ctx_id < 0 else ['CUDAExecutionProvider'],
            )
            self._app.prepare(ctx_id=self._ctx_id, det_size=(640, 640))
            logger.info("InsightFace engine initialized with model=%s", self._model_name)
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.warning("InsightFace initialization failed: %s. Falling back to baseline.", e)
            return False

    def _quality(self, face) -> float:
        """Compute quality score from InsightFace face object."""
        # Use detection score and face size as quality indicators
        det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.5
        bbox = face.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # Normalize area (assume 640x640 input, face area up to ~40% is good)
        area_score = min(1.0, area / (640 * 640 * 0.2))
        return det_score * 50.0 + area_score * 50.0

    def best_face_embedding(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        """Return the highest-quality face embedding from a frame, if any."""
        if not self._lazy_init():
            return None, 0.0

        try:
            faces = self._app.get(frame_bgr)
            if not faces:
                return None, 0.0

            best_face = None
            best_score = -1.0

            for face in faces:
                score = self._quality(face)
                if score > best_score:
                    best_score = score
                    best_face = face

            if best_face is None or not hasattr(best_face, 'embedding'):
                return None, 0.0

            embedding = best_face.embedding.astype(np.float32)
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding, best_score

        except Exception as e:
            logger.warning("InsightFace inference failed: %s", e)
            return None, 0.0


class BaselineFaceEngine:
    """Baseline OpenCV-based face extraction and lightweight embedding engine.
    
    Uses Haar Cascade for detection and 16x16 grayscale resize for embeddings.
    Fast but low accuracy - use as fallback only.
    """

    def __init__(self) -> None:
        """Load cascade detector once for reuse across frame evaluations."""
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _embedding_from_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Convert a face crop into a normalized fixed-length embedding vector."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        vec = resized.astype(np.float32).reshape(-1)
        vec = vec - float(np.mean(vec))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def _quality(self, gray_face: np.ndarray, frame_area: float, box_area: float) -> float:
        """Compute a heuristic quality score (size + sharpness)."""
        blur = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
        area_ratio = box_area / max(frame_area, 1.0)
        return (area_ratio * 100.0) + min(blur / 220.0, 1.0) * 20.0

    def best_face_embedding(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        """Return the highest-quality face embedding from a frame, if any."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None, 0.0

        h, w = gray.shape
        frame_area = float(h * w)

        best_embedding: Optional[np.ndarray] = None
        best_score = -1.0

        for (x, y, fw, fh) in faces:
            face_crop = frame_bgr[y : y + fh, x : x + fw]
            if face_crop.size == 0:
                continue
            face_gray = gray[y : y + fh, x : x + fw]
            score = self._quality(face_gray, frame_area, float(fw * fh))
            if score > best_score:
                best_score = score
                best_embedding = self._embedding_from_face(face_crop)

        if best_embedding is None:
            return None, 0.0
        return best_embedding, best_score


# Backwards compatibility alias
FaceEmbeddingEngine = BaselineFaceEngine


class HybridFaceEngine:
    """Face engine that tries InsightFace first, falls back to baseline."""

    def __init__(self, prefer_insightface: bool = True) -> None:
        self._insightface: Optional[InsightFaceEngine] = None
        self._baseline: Optional[BaselineFaceEngine] = None
        self._prefer_insightface = prefer_insightface
        
        if prefer_insightface:
            self._insightface = InsightFaceEngine()

    def best_face_embedding(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        """Return the highest-quality face embedding from a frame."""
        # Try InsightFace first
        if self._insightface is not None:
            embedding, score = self._insightface.best_face_embedding(frame_bgr)
            if embedding is not None:
                return embedding, score
            # If InsightFace failed to init, fall through to baseline
            if self._insightface._init_attempted and self._insightface._app is None:
                logger.info("InsightFace unavailable, using baseline engine")
                self._insightface = None  # Stop trying

        # Fallback to baseline
        if self._baseline is None:
            self._baseline = BaselineFaceEngine()
        return self._baseline.best_face_embedding(frame_bgr)


class FaceRecognizer:
    """High-level recognizer backed by local embedding database."""

    def __init__(
        self,
        db_path: str = "data/faces.db",
        match_threshold: float = 0.80,
        min_samples: int = 3,
        use_insightface: bool = True,
    ) -> None:
        """Initialize embedding engine and DB-backed matching parameters."""
        self.db = FaceDatabase(Path(db_path))
        self.engine = HybridFaceEngine(prefer_insightface=use_insightface)
        self.match_threshold = match_threshold
        self.min_samples = min_samples

    def identify(self, frame) -> RecognitionResult:
        """Identify best matching known person for a frame."""
        embedding, _ = self.engine.best_face_embedding(frame)
        if embedding is None:
            return RecognitionResult(person_id=None, confidence=0.0)

        match = self.db.match_embedding(
            embedding=embedding,
            threshold=self.match_threshold,
            min_samples=self.min_samples,
        )
        return RecognitionResult(person_id=match.person_id, confidence=match.similarity)

    def remember_sample(
        self,
        frame,
        source_image: str,
        camera_id: int,
        captured_at: Optional[str],
        person_id: Optional[str],
    ) -> bool:
        """Persist a frame-derived face sample for future recognition."""
        embedding, quality = self.engine.best_face_embedding(frame)
        if embedding is None:
            return False

        target_person = (person_id or "unknown").strip() or "unknown"
        self.db.upsert_person(person_id=target_person, display_name=target_person)
        self.db.add_face_sample(
            person_id=target_person,
            embedding=embedding,
            source_image=source_image,
            camera_id=camera_id,
            captured_at=captured_at,
            quality_score=quality,
            face_box="",
        )
        return True

    def close(self) -> None:
        """Release DB resources."""
        self.db.close()


class NotificationRules:
    """
    In-memory rules by person id and weekday (0=Monday, 6=Sunday).
    """

    def __init__(self) -> None:
        self.blocked_days_by_person: Dict[str, Set[int]] = {}

    def should_notify(self, person_id: Optional[str], now: datetime) -> bool:
        if person_id is None:
            return True
        blocked_days = self.blocked_days_by_person.get(person_id, set())
        return now.weekday() not in blocked_days

    def block_day(self, person_id: str, weekday: int) -> None:
        self.blocked_days_by_person.setdefault(person_id, set()).add(weekday)

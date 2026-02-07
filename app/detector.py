from __future__ import annotations

"""Person detection wrapper around Ultralytics YOLO."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """Detect class-0 (person) instances and annotate confident boxes."""

    def __init__(self, model_name: str, confidence_threshold: float) -> None:
        """Load the YOLO model once per worker/thread."""
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _is_box_ignored(
        box: Tuple[int, int, int, int],
        ignored_boxes: List[Tuple[int, int, int, int]],
    ) -> bool:
        """Treat a detection as ignored when its center falls in any ignore box."""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        for ix1, iy1, ix2, iy2 in ignored_boxes:
            if ix1 <= center_x <= ix2 and iy1 <= center_y <= iy2:
                return True
        return False

    def detect(
        self,
        frame: np.ndarray,
        ignored_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[bool, np.ndarray, float, Optional[Tuple[int, int, int, int]]]:
        """Run inference and return `(has_person, annotated_frame, max_confidence, max_conf_box_xyxy)`."""
        ignored_boxes = ignored_boxes or []
        results = self.model.predict(frame, verbose=False)
        if not results:
            return False, frame, 0.0, None

        result = results[0]
        max_person_conf = 0.0
        max_person_box: Optional[Tuple[int, int, int, int]] = None

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_box = (x1, y1, x2, y2)
                if self._is_box_ignored(person_box, ignored_boxes):
                    continue

                if confidence > max_person_conf:
                    max_person_conf = confidence
                    max_person_box = person_box

                if confidence < self.confidence_threshold:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    frame,
                    f"person {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

        return max_person_conf >= self.confidence_threshold, frame, max_person_conf, max_person_box

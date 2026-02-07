from __future__ import annotations

"""Core surveillance orchestration loop.

This module coordinates:
- camera ingestion workers
- person detection and burst selection
- face recognition lookup/persistence
- Telegram notifications and status reporting
"""

import logging
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict, Optional

import cv2

from app.camera import CameraClient, CameraFrame, IsapiSnapshotCamera, RtspCamera
from app.config import CameraConfig, Settings, build_camera_map
from app.detector import PersonDetector
from app.face_rules import FaceRecognizer, NotificationRules
from app.notifier import TelegramEvent, TelegramNotifier
from ultralytics import YOLO

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "detection_models"
BASE_MODEL_PATH = MODEL_DIR / "yolov8n_base.pt"
ACTIVE_MODEL_PATH = MODEL_DIR / "yolov8n.pt"


@dataclass
class BurstState:
    """In-flight detection burst state for one camera."""

    camera: CameraConfig
    started_at: float
    ends_at: float
    frames_seen: int
    best_frame: any
    best_score: float
    best_face_count: int
    trigger_confidence: float
    trigger_annotated_frame: any


class RuntimeStats:
    """Thread-safe counters used for telemetry and `/status` responses."""

    def __init__(self, high_conf_cutoff: float) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.high_conf_cutoff = high_conf_cutoff

        self.frames_downloaded = 0
        self.snapshots_saved = 0
        self.people_detected = 0
        self.high_conf_people_detected = 0
        self.alerts_sent = 0
        self.errors = 0
        self.last_event_at = self.started_at

        self._last_report = {
            "frames_downloaded": 0,
            "snapshots_saved": 0,
            "people_detected": 0,
            "high_conf_people_detected": 0,
            "alerts_sent": 0,
            "errors": 0,
        }

    def _touch(self) -> None:
        self.last_event_at = time.time()

    def inc_frames_downloaded(self) -> None:
        with self._lock:
            self.frames_downloaded += 1
            self._touch()

    def inc_snapshots_saved(self) -> None:
        with self._lock:
            self.snapshots_saved += 1
            self._touch()

    def record_person_detection(self, confidence: float) -> None:
        with self._lock:
            self.people_detected += 1
            if confidence >= self.high_conf_cutoff:
                self.high_conf_people_detected += 1
            self._touch()

    def inc_alerts_sent(self) -> None:
        with self._lock:
            self.alerts_sent += 1
            self._touch()

    def inc_errors(self) -> None:
        with self._lock:
            self.errors += 1
            self._touch()

    def status_report(self, cameras: int) -> str:
        """Build status string and reset `since last report` checkpoint."""
        with self._lock:
            now = time.time()
            uptime = int(now - self.started_at)
            since_last = {
                "frames_downloaded": self.frames_downloaded - self._last_report["frames_downloaded"],
                "snapshots_saved": self.snapshots_saved - self._last_report["snapshots_saved"],
                "people_detected": self.people_detected - self._last_report["people_detected"],
                "high_conf_people_detected": self.high_conf_people_detected - self._last_report["high_conf_people_detected"],
                "alerts_sent": self.alerts_sent - self._last_report["alerts_sent"],
                "errors": self.errors - self._last_report["errors"],
            }
            self._last_report = {
                "frames_downloaded": self.frames_downloaded,
                "snapshots_saved": self.snapshots_saved,
                "people_detected": self.people_detected,
                "high_conf_people_detected": self.high_conf_people_detected,
                "alerts_sent": self.alerts_sent,
                "errors": self.errors,
            }

            last_event = int(now - self.last_event_at)

        return (
            "App status: running\n"
            f"Uptime: {uptime}s | Cameras: {cameras} | Last activity: {last_event}s ago\n"
            f"Since last report: frames={since_last['frames_downloaded']}, snapshots={since_last['snapshots_saved']}, "
            f"detections={since_last['people_detected']}, high_conf={since_last['high_conf_people_detected']} "
            f"(> {self.high_conf_cutoff:.2f}), alerts={since_last['alerts_sent']}, errors={since_last['errors']}\n"
            f"Totals: frames={self.frames_downloaded}, snapshots={self.snapshots_saved}, detections={self.people_detected}, "
            f"high_conf={self.high_conf_people_detected}, alerts={self.alerts_sent}, errors={self.errors}"
        )


class SurveillanceApp:
    """Top-level service object controlling worker lifecycle and pipelines."""

    def __init__(self, settings: Settings) -> None:
        """Initialize dependencies, workers state, and command listener."""
        self.settings = settings
        self.settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_target_ratio = self._parse_aspect_ratio(settings.snapshot_target_aspect_ratio)

        self.cameras = self._build_cameras(settings)
        self.notifier = TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self.face_recognizer = FaceRecognizer(
            db_path=str(self.settings.face_db_path),
            match_threshold=self.settings.face_match_threshold,
            min_samples=self.settings.face_min_samples,
        )
        self.rules = NotificationRules()
        self.stats = RuntimeStats(high_conf_cutoff=self.settings.confidence_threshold * 0.8)

        self.frame_queues: Dict[int, queue.Queue[CameraFrame]] = {
            camera_id: queue.Queue(maxsize=self.settings.frame_queue_size)
            for camera_id in self.cameras
        }
        self.stop_event = threading.Event()

        self.poller_threads: Dict[int, threading.Thread] = {}
        self.processor_threads: Dict[int, threading.Thread] = {}
        self.status_report_thread: threading.Thread | None = None
        self.notify_lock = threading.Lock()

        self.last_alert_by_camera: Dict[int, float] = {}
        self.last_periodic_by_camera: Dict[int, float] = {}
        self.burst_by_camera: Dict[int, BurstState] = {}
        self.status_report_interval_seconds = max(0.0, float(settings.status_report_interval_hours) * 3600.0)
        self._status_interval_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._notification_pause_lock = threading.Lock()
        self._notifications_paused_until: Optional[float] = None
        self._notifications_paused_indefinite = False
        self._model_generation_lock = threading.Lock()
        self._model_generation = 0
        self._model_name_lock = threading.Lock()
        self._model_dir = MODEL_DIR
        self._base_model_path = BASE_MODEL_PATH
        self._active_model_path = ACTIVE_MODEL_PATH
        self._active_yolo_model = str(self._active_model_path)
        self._model_update_lock = threading.Lock()
        self._model_update_thread: Optional[threading.Thread] = None

        self._ensure_managed_detector_models(configured_model=settings.yolo_model)
        self._init_feedback_store()
        self.notifier.start_command_listener(self._handle_telegram_event)

    def _init_feedback_store(self) -> None:
        """Ensure DB tables used by Telegram-driven detector feedback exist."""
        with sqlite3.connect(str(self.settings.face_db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshot_reviews (
                    source_image TEXT PRIMARY KEY,
                    detector_label TEXT CHECK(detector_label IN ('person', 'no_person') OR detector_label IS NULL),
                    person_id TEXT,
                    notes TEXT,
                    reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS telegram_snapshot_map (
                    message_id INTEGER PRIMARY KEY,
                    source_image TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def _ensure_managed_detector_models(self, configured_model: str) -> None:
        """Initialize managed detector model files under `detection_models/`."""
        self._model_dir.mkdir(parents=True, exist_ok=True)

        configured_raw = (configured_model or "").strip()
        configured_path: Optional[Path] = None
        if configured_raw:
            maybe_path = Path(configured_raw)
            configured_path = maybe_path if maybe_path.is_absolute() else (PROJECT_ROOT / maybe_path).resolve()
        legacy_root_model = PROJECT_ROOT / "yolov8n.pt"

        if not self._base_model_path.exists():
            source: Optional[Path] = None
            if configured_path and configured_path.exists() and configured_path != self._active_model_path:
                source = configured_path
            elif legacy_root_model.exists():
                source = legacy_root_model
            elif self._active_model_path.exists():
                source = self._active_model_path

            if source is None:
                raise FileNotFoundError(
                    "No base YOLO model found. Expected one of: "
                    f"{configured_raw or '<unset>'}, {legacy_root_model}, {self._active_model_path}"
                )

            # Move legacy top-level model to managed base name; copy otherwise.
            if source == legacy_root_model:
                shutil.move(str(source), str(self._base_model_path))
                logger.info("Moved legacy base model to %s", self._base_model_path)
            elif source == self._active_model_path:
                shutil.copy2(str(source), str(self._base_model_path))
            else:
                shutil.copy2(str(source), str(self._base_model_path))

        if not self._active_model_path.exists():
            shutil.copy2(str(self._base_model_path), str(self._active_model_path))
            logger.info("Initialized active YOLO model at %s", self._active_model_path)

        self._set_model_name(str(self._active_model_path))

    def _remember_telegram_snapshot_message(self, message_id: int, image_path: Path) -> None:
        """Store Telegram message -> snapshot mapping for later user feedback."""
        if message_id <= 0:
            return
        with self._feedback_lock:
            with sqlite3.connect(str(self.settings.face_db_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO telegram_snapshot_map (message_id, source_image)
                    VALUES (?, ?)
                    ON CONFLICT(message_id) DO UPDATE SET
                        source_image = excluded.source_image,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (message_id, str(image_path.resolve())),
                )
                conn.commit()

    def _mark_snapshot_no_person_from_telegram_message(self, message_id: int) -> Optional[str]:
        """Mark mapped snapshot as `no_person` and return source path when found."""
        if message_id <= 0:
            return None
        with self._feedback_lock:
            with sqlite3.connect(str(self.settings.face_db_path)) as conn:
                row = conn.execute(
                    "SELECT source_image FROM telegram_snapshot_map WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if row is None or not row[0]:
                    return None
                source_image = str(row[0])
                existing = conn.execute(
                    "SELECT person_id FROM snapshot_reviews WHERE source_image = ?",
                    (source_image,),
                ).fetchone()
                existing_person = str(existing[0]) if existing is not None and existing[0] else None
                conn.execute(
                    """
                    INSERT INTO snapshot_reviews (source_image, detector_label, person_id, reviewed_at)
                    VALUES (?, 'no_person', ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(source_image) DO UPDATE SET
                        detector_label = 'no_person',
                        person_id = COALESCE(snapshot_reviews.person_id, excluded.person_id),
                        reviewed_at = CURRENT_TIMESTAMP
                    """,
                    (source_image, existing_person),
                )
                conn.commit()
                return source_image

    @staticmethod
    def _has_thumbs_down(emoji_values: tuple[str, ...]) -> bool:
        """Return True when any emoji reaction is a thumbs-down variant."""
        return any(str(emoji).startswith("👎") for emoji in emoji_values)

    def _current_status_interval_seconds(self) -> float:
        with self._status_interval_lock:
            return self.status_report_interval_seconds

    def _set_status_interval_seconds(self, seconds: float) -> None:
        with self._status_interval_lock:
            self.status_report_interval_seconds = max(0.0, seconds)

    def _pause_notifications(self, hours: Optional[float]) -> None:
        """Pause Telegram notifications indefinitely or for a fixed duration."""
        with self._notification_pause_lock:
            if hours is None:
                self._notifications_paused_indefinite = True
                self._notifications_paused_until = None
                return
            self._notifications_paused_indefinite = False
            self._notifications_paused_until = time.time() + max(0.0, float(hours) * 3600.0)

    def _resume_notifications(self) -> None:
        """Resume Telegram notifications immediately."""
        with self._notification_pause_lock:
            self._notifications_paused_indefinite = False
            self._notifications_paused_until = None

    def _notifications_paused(self) -> bool:
        """Return whether Telegram notifications are currently paused."""
        with self._notification_pause_lock:
            if self._notifications_paused_indefinite:
                return True
            if self._notifications_paused_until is None:
                return False
            if time.time() < self._notifications_paused_until:
                return True
            self._notifications_paused_until = None
            return False

    def _notifications_pause_summary(self) -> str:
        """Human-readable pause state for status/help responses."""
        with self._notification_pause_lock:
            if self._notifications_paused_indefinite:
                return "paused indefinitely"
            if self._notifications_paused_until is None:
                return "active"
            remaining = max(0, int(self._notifications_paused_until - time.time()))
            if remaining <= 0:
                return "active"
            return f"paused ({remaining}s remaining)"

    def _current_model_generation(self) -> int:
        """Return current detector model generation."""
        with self._model_generation_lock:
            return self._model_generation

    def _bump_model_generation(self) -> int:
        """Increment detector model generation and return new value."""
        with self._model_generation_lock:
            self._model_generation += 1
            return self._model_generation

    def _current_model_name(self) -> str:
        """Return currently configured YOLO model path/name."""
        with self._model_name_lock:
            return self._active_yolo_model

    def _set_model_name(self, model_name: str) -> None:
        """Update active YOLO model path/name used by worker reloads."""
        value = (model_name or "").strip()
        if not value:
            return
        with self._model_name_lock:
            self._active_yolo_model = value

    def _model_update_running(self) -> bool:
        """Return whether retrain/reload workflow is currently running."""
        with self._model_update_lock:
            return self._model_update_thread is not None and self._model_update_thread.is_alive()

    def _start_model_update(self, requested_model: str) -> bool:
        """Start background export+train+reload workflow."""
        with self._model_update_lock:
            if self._model_update_thread is not None and self._model_update_thread.is_alive():
                return False
            thread = threading.Thread(
                target=self._run_model_update_pipeline,
                args=(requested_model,),
                name="model-update",
                daemon=True,
            )
            self._model_update_thread = thread
            thread.start()
            return True

    def _run_model_update_pipeline(self, requested_model: str) -> None:
        """Export reviewed dataset, train detector, then reload workers."""
        base_model = (requested_model or "").strip() or str(self._base_model_path)
        dataset_dir = PROJECT_ROOT / "data" / "detector_dataset"
        dataset_yaml = dataset_dir / "dataset.yaml"
        train_project = PROJECT_ROOT / "data" / "detector_training"
        run_name = f"runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        best_weights = train_project / run_name / "weights" / "best.pt"
        promoted_tagged = self._model_dir / f"yolov8n_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

        try:
            logger.info("Starting model update pipeline | base_model=%s", base_model)
            export_cmd = [
                sys.executable,
                "scripts/export_detector_dataset.py",
                "--db-path",
                str(self.settings.face_db_path),
                "--output-dir",
                str(dataset_dir),
                "--model",
                base_model,
                "--confidence",
                "0.35",
                "--clean",
            ]
            export_proc = subprocess.run(
                export_cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            if export_proc.returncode != 0:
                raise RuntimeError(
                    f"dataset export failed (code={export_proc.returncode}): {export_proc.stderr.strip() or export_proc.stdout.strip()}"
                )
            if not dataset_yaml.exists():
                raise RuntimeError(f"dataset.yaml not found after export: {dataset_yaml}")

            logger.info("Dataset export complete. Starting YOLO training | run=%s", run_name)
            model = YOLO(base_model)
            model.train(
                data=str(dataset_yaml),
                epochs=20,
                imgsz=640,
                batch=8,
                project=str(train_project),
                name=run_name,
                exist_ok=False,
                verbose=False,
            )
            if not best_weights.exists():
                raise RuntimeError(f"training finished but best weights not found: {best_weights}")

            shutil.copy2(str(best_weights), str(promoted_tagged))
            shutil.copy2(str(best_weights), str(self._active_model_path))
            self._set_model_name(str(self._active_model_path))
            generation = self._bump_model_generation()
            logger.info(
                "Model update complete | new_model=%s | active_model=%s | generation=%d",
                promoted_tagged,
                self._active_model_path,
                generation,
            )
            with self.notify_lock:
                self.notifier.send_text(
                    "Model update complete. "
                    f"Loaded weights: {self._active_model_path} (generation {generation}). "
                    f"Archived as: {promoted_tagged.name}"
                )
        except Exception as exc:
            logger.exception("Model update pipeline failed: %s", exc)
            with self.notify_lock:
                self.notifier.send_text(f"Model update failed: {exc}")

    def _handle_telegram_event(self, event: TelegramEvent) -> Optional[str]:
        """Handle Telegram commands and thumbs-down reaction feedback."""
        if event.reaction_message_id and self._has_thumbs_down(event.reaction_emojis):
            source = self._mark_snapshot_no_person_from_telegram_message(event.reaction_message_id)
            if source:
                return f"Marked as no_person from reaction: {Path(source).name}"
            return "I could not map this reaction to a snapshot message."

        text = (event.text or "").strip()
        if not text:
            return None

        parts = text.split()
        command = parts[0].lower()

        if command in {"ping", "/ping"}:
            return "pong"
        if command in {"status", "/status", "/report"}:
            return (
                self.stats.status_report(cameras=len(self.cameras))
                + f"\nNotifications: {self._notifications_pause_summary()}"
                + f"\nModel generation: {self._current_model_generation()}"
                + f"\nYOLO model: {self._current_model_name()}"
                + f"\nModel update running: {self._model_update_running()}"
            )
        if command in {"help", "/help", "?"}:
            interval_h = self._current_status_interval_seconds() / 3600.0
            return (
                "Commands:\n"
                "/ping\n"
                "/status\n"
                "/noperson (reply to a snapshot)\n"
                "/pause or /pause <hours>\n"
                "/resume\n"
                "/reload_model\n"
                f"/setstatus <hours|off>  (current: {interval_h:.2f}h)"
            )
        if command in {"reload_model", "/reload_model"}:
            requested = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
            if self._model_update_running():
                return "Model update already running. Please wait for completion."

            if requested:
                base_model = requested
                model_source = "command argument"
            else:
                base_model = str(self._base_model_path)
                model_source = "managed base model"

            started = self._start_model_update(base_model)
            if not started:
                return "Model update already running. Please wait for completion."
            logger.info(
                "Telegram command queued model update pipeline | source=%s | base_model=%s",
                model_source,
                base_model,
            )
            return (
                f"Model update started (source={model_source}, base_model={base_model}). "
                "I will export dataset, train, then reload workers."
            )
        if command == "/pause":
            if len(parts) < 2:
                self._pause_notifications(hours=None)
                logger.info("Telegram notifications paused indefinitely by command")
                return "Notifications paused indefinitely. Use /resume to re-enable."
            try:
                hours = float(parts[1].strip())
            except ValueError:
                return "Usage: /pause or /pause <hours> (example: /pause 6)"
            if hours <= 0:
                return "Pause hours must be > 0. Use /resume to re-enable notifications."
            self._pause_notifications(hours=hours)
            logger.info("Telegram notifications paused for %.2f hours by command", hours)
            return f"Notifications paused for {hours:.2f} hours."
        if command == "/resume":
            self._resume_notifications()
            logger.info("Telegram notifications resumed by command")
            return "Notifications resumed."
        if command in {"noperson", "/noperson"}:
            if not event.reply_to_message_id:
                return "Reply to a snapshot alert with /noperson, or react with 👎."
            source = self._mark_snapshot_no_person_from_telegram_message(event.reply_to_message_id)
            if source:
                return f"Marked as no_person: {Path(source).name}"
            return "I could not map that replied message to a snapshot."
        if command == "/setstatus":
            if len(parts) < 2:
                interval_h = self._current_status_interval_seconds() / 3600.0
                return f"Current periodic status interval: {interval_h:.2f}h. Usage: /setstatus <hours|off>"

            raw = parts[1].strip().lower()
            if raw in {"off", "0"}:
                self._set_status_interval_seconds(0.0)
                logger.info("Telegram command updated periodic status interval: disabled")
                return "Periodic status reports disabled."
            try:
                hours = float(raw)
            except ValueError:
                return "Invalid value. Usage: /setstatus <hours|off> (example: /setstatus 6)"
            if hours <= 0:
                return "Hours must be > 0, or use /setstatus off."

            self._set_status_interval_seconds(hours * 3600.0)
            logger.info("Telegram command updated periodic status interval: %.2f hours", hours)
            return f"Periodic status reports set to every {hours:.2f} hours."
        return None

    def _run_periodic_status_reports(self) -> None:
        """Send periodic Telegram status reports at configurable intervals."""
        if not self.notifier.enabled:
            return

        next_report_at = time.time() + self._current_status_interval_seconds()
        while not self.stop_event.is_set():
            interval = self._current_status_interval_seconds()
            if interval <= 0:
                self.stop_event.wait(timeout=1.0)
                next_report_at = time.time() + max(1.0, self._current_status_interval_seconds())
                continue

            now = time.time()
            if now >= next_report_at:
                report = self.stats.status_report(cameras=len(self.cameras))
                with self.notify_lock:
                    sent = self.notifier.send_text(report)
                if sent:
                    self.stats.inc_alerts_sent()
                    logger.info("Periodic status report sent to Telegram")
                else:
                    self.stats.inc_errors()
                    logger.error("Periodic status report failed")
                next_report_at = now + interval
                continue

            self.stop_event.wait(timeout=min(1.0, max(0.1, next_report_at - now)))

    def _build_cameras(self, settings: Settings) -> Dict[int, CameraClient]:
        """Create camera clients according to configured capture mode."""
        camera_map = build_camera_map(settings)

        if settings.capture_mode == "isapi":
            return {
                channel_id: IsapiSnapshotCamera(
                    camera=config,
                    username=settings.dvr_username,
                    password=settings.dvr_password,
                    timeout_seconds=settings.isapi_timeout_seconds,
                    reconnect_seconds=settings.camera_reconnect_seconds,
                    auth_mode=settings.isapi_auth_mode,
                )
                for channel_id, config in camera_map.items()
            }

        return {
            channel_id: RtspCamera(
                camera=config,
                reconnect_seconds=settings.camera_reconnect_seconds,
                rtsp_transport=settings.rtsp_transport,
            )
            for channel_id, config in camera_map.items()
        }

    def _parse_aspect_ratio(self, raw: str) -> Optional[float]:
        """Parse aspect ratio values (`16:9`, float, or disabled flags)."""
        value = raw.strip().lower()
        if not value or value in {"off", "none", "-1"}:
            return None
        if ":" in value:
            left, right = value.split(":", 1)
            width = float(left)
            height = float(right)
            if width <= 0 or height <= 0:
                return None
            return width / height
        ratio = float(value)
        if ratio <= 0:
            return None
        return ratio

    def _correct_snapshot_aspect(self, frame):
        """Apply optional output aspect correction before persistence."""
        if self.snapshot_target_ratio is None:
            return frame
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return frame
        current_ratio = width / height
        if abs(current_ratio - self.snapshot_target_ratio) < 0.01:
            return frame
        target_width = max(1, int(round(height * self.snapshot_target_ratio)))
        return cv2.resize(frame, (target_width, height), interpolation=cv2.INTER_LINEAR)

    def _save_snapshot(self, frame, camera: CameraConfig) -> Path:
        """Persist one snapshot and update runtime stats."""
        now = datetime.now()
        filename = f"{camera.channel_key}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = self.settings.snapshot_dir / filename
        corrected = self._correct_snapshot_aspect(frame)
        cv2.imwrite(str(output_path), corrected)
        self.stats.inc_snapshots_saved()
        return output_path

    def _camera_on_cooldown(self, channel_id: int) -> bool:
        """Return whether detection alerts are in cooldown for this camera."""
        last_alert = self.last_alert_by_camera.get(channel_id)
        if last_alert is None:
            return False
        return monotonic() - last_alert < self.settings.alert_cooldown_seconds

    def _periodic_due(self, channel_id: int) -> bool:
        """Return whether periodic snapshot schedule is due for this camera."""
        interval = self.settings.periodic_alert_seconds
        if interval <= 0:
            return False
        last = self.last_periodic_by_camera.get(channel_id)
        if last is None:
            return True
        return monotonic() - last >= interval

    def _score_face_quality(self, face_detector: cv2.CascadeClassifier, frame) -> tuple[float, int]:
        """Score frame quality for face usability (size + sharpness heuristic)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return 0.0, 0

        height, width = gray.shape
        best = 0.0
        for (x, y, w, h) in faces:
            face_area_ratio = (w * h) / float(width * height)
            roi = gray[y : y + h, x : x + w]
            blur_score = cv2.Laplacian(roi, cv2.CV_64F).var()
            quality = (face_area_ratio * 100.0) + min(blur_score / 220.0, 1.0) * 20.0
            if quality > best:
                best = quality
        return best, len(faces)

    def _notify_sync(self, image_path: Path, caption: str, camera_name: str) -> None:
        """Send alert synchronously (or log local-only mode) with stats updates."""
        if self.notifier.enabled:
            if self._notifications_paused():
                logger.info("Notification suppressed while paused for %s (%s)", camera_name, image_path.name)
                return
            with self.notify_lock:
                message_id = self.notifier.send_snapshot(image_path=image_path, caption=caption)
            if message_id:
                self._remember_telegram_snapshot_message(message_id=message_id, image_path=image_path)
                logger.info("Alert sent for %s", camera_name)
                self.stats.inc_alerts_sent()
            else:
                logger.error("Alert failed for %s", camera_name)
                self.stats.inc_errors()
            return
        logger.info("Snapshot saved for %s at %s", camera_name, image_path)

    def _send_periodic_snapshot(self, detector: PersonDetector, frame_packet: CameraFrame) -> None:
        """Execute unconditional periodic snapshot path for one frame."""
        ignored_boxes = self.settings.ignored_person_bboxes.get(frame_packet.camera.channel_key, [])
        has_person, annotated, confidence, _ = detector.detect(
            frame_packet.frame,
            ignored_boxes=ignored_boxes,
        )
        if has_person:
            self.stats.record_person_detection(confidence=confidence)
        image_path = self._save_snapshot(annotated, frame_packet.camera)
        caption = (
            f"Periodic snapshot from {frame_packet.camera.name} "
            f"(channel {frame_packet.camera.channel_key}, person={has_person}, conf={confidence:.2f})"
        )
        self._notify_sync(image_path, caption, frame_packet.camera.name)

    def _start_burst(
        self,
        face_detector: cv2.CascadeClassifier,
        camera_id: int,
        frame_packet: CameraFrame,
        trigger_confidence: float,
        trigger_annotated_frame,
    ) -> None:
        """Start burst window after initial person detection on a camera."""
        quality, face_count = self._score_face_quality(face_detector, frame_packet.frame)
        now = monotonic()
        self.burst_by_camera[camera_id] = BurstState(
            camera=frame_packet.camera,
            started_at=now,
            ends_at=now + self.settings.burst_capture_seconds,
            frames_seen=1,
            best_frame=frame_packet.frame.copy(),
            best_score=quality,
            best_face_count=face_count,
            trigger_confidence=trigger_confidence,
            trigger_annotated_frame=trigger_annotated_frame.copy(),
        )
        logger.info(
            "Burst started for %s (faces=%d score=%.2f)",
            frame_packet.camera.name,
            face_count,
            quality,
        )

    def _finalize_burst(self, detector: PersonDetector, camera_id: int, burst: BurstState) -> None:
        """Finalize burst, emit alert, and persist face sample when possible."""
        try:
            recognition = self.face_recognizer.identify(burst.best_frame)
            if not self.rules.should_notify(recognition.person_id, datetime.now()):
                logger.info("Suppressed alert for %s by rule", recognition.person_id)
                self.last_alert_by_camera[camera_id] = monotonic()
                return

            # Re-run person detection on the selected burst frame; if no person is found
            # at finalize time, fall back to the trigger frame annotation so alerts
            # consistently contain a person box.
            ignored_boxes = self.settings.ignored_person_bboxes.get(camera_id, [])
            has_person, annotated, _, _ = detector.detect(
                burst.best_frame.copy(),
                ignored_boxes=ignored_boxes,
            )
            if not has_person:
                annotated = burst.trigger_annotated_frame.copy()
            image_path = self._save_snapshot(annotated, burst.camera)
            who = recognition.person_id or "unknown"
            caption = (
                f"Person detected on {burst.camera.name} "
                f"(channel {burst.camera.channel_key}, trigger_conf={burst.trigger_confidence:.2f}, "
                f"faces={burst.best_face_count}, face_score={burst.best_score:.1f}, id={who})"
            )
            self._notify_sync(image_path, caption, burst.camera.name)
            remembered = self.face_recognizer.remember_sample(
                frame=burst.best_frame,
                source_image=str(image_path),
                camera_id=burst.camera.channel_key,
                captured_at=datetime.now().isoformat(timespec="seconds"),
                person_id=recognition.person_id,
            )
            if not remembered:
                logger.info("No usable face embedding to store for %s", burst.camera.name)
            self.last_alert_by_camera[camera_id] = monotonic()
        except Exception:
            logger.exception("Failed finalizing burst for %s", burst.camera.name)
            self.stats.inc_errors()
            self.last_alert_by_camera[camera_id] = monotonic()
        finally:
            self.burst_by_camera.pop(camera_id, None)

    def _handle_detection_flow(
        self,
        detector: PersonDetector,
        face_detector: cv2.CascadeClassifier,
        camera_id: int,
        frame_packet: CameraFrame,
    ) -> None:
        """Process detection state machine (start/update/finalize burst)."""
        burst = self.burst_by_camera.get(camera_id)
        if burst is not None:
            quality, face_count = self._score_face_quality(face_detector, frame_packet.frame)
            burst.frames_seen += 1
            if quality >= burst.best_score:
                burst.best_score = quality
                burst.best_face_count = face_count
                burst.best_frame = frame_packet.frame.copy()

            if monotonic() >= burst.ends_at or burst.frames_seen >= self.settings.burst_max_frames:
                self._finalize_burst(detector, camera_id, burst)
            return

        if self._camera_on_cooldown(camera_id):
            return

        ignored_boxes = self.settings.ignored_person_bboxes.get(camera_id, [])
        has_person, annotated, confidence, max_box = detector.detect(
            frame_packet.frame,
            ignored_boxes=ignored_boxes,
        )
        if not has_person:
            if confidence > 0:
                box_text = "none"
                if max_box is not None:
                    box_text = f"{max_box[0]},{max_box[1]},{max_box[2]},{max_box[3]}"
                logger.info(
                    "Low-confidence person detection suppressed | camera=%s | channel=%s | confidence=%.3f | threshold=%.3f | bbox_xyxy=%s | captured_at=%s",
                    frame_packet.camera.name,
                    frame_packet.camera.channel_key,
                    confidence,
                    self.settings.confidence_threshold,
                    box_text,
                    datetime.fromtimestamp(frame_packet.captured_at).isoformat(timespec="seconds"),
                )
            return
        self.stats.record_person_detection(confidence=confidence)

        self._start_burst(
            face_detector,
            camera_id,
            frame_packet,
            confidence,
            annotated,
        )

    def _process_frame(
        self,
        detector: PersonDetector,
        face_detector: cv2.CascadeClassifier,
        camera_id: int,
        frame_packet: CameraFrame,
    ) -> None:
        """Run periodic and detection flows for one frame packet."""
        if self._periodic_due(camera_id):
            self._send_periodic_snapshot(detector, frame_packet)
            self.last_periodic_by_camera[camera_id] = monotonic()

        self._handle_detection_flow(detector, face_detector, camera_id, frame_packet)

    def _poll_camera(self, camera_id: int, camera: CameraClient) -> None:
        """Producer loop: read frames and push into bounded per-camera queue."""
        frame_queue = self.frame_queues[camera_id]
        while not self.stop_event.is_set():
            frame_packet = camera.read()
            if frame_packet is not None:
                self.stats.inc_frames_downloaded()
                try:
                    frame_queue.put(frame_packet, timeout=0.2)
                except queue.Full:
                    # Drop oldest frame to keep latency bounded under load.
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put(frame_packet, timeout=0.2)
                    except queue.Full:
                        pass
            time.sleep(self.settings.detect_every_seconds)

    def _process_camera(self, camera_id: int) -> None:
        """Consumer loop: run inference pipeline for one camera queue."""
        detector: PersonDetector | None = None
        detector_generation = -1
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        frame_queue = self.frame_queues[camera_id]
        camera_name = self.cameras[camera_id].camera.name

        while not self.stop_event.is_set():
            current_generation = self._current_model_generation()
            if detector is None or current_generation != detector_generation:
                try:
                    detector = PersonDetector(
                        model_name=self._current_model_name(),
                        confidence_threshold=self.settings.confidence_threshold,
                    )
                    detector_generation = current_generation
                    logger.info(
                        "Loaded YOLO model for %s (generation=%d, model=%s)",
                        camera_name,
                        detector_generation,
                        self._current_model_name(),
                    )
                except Exception:
                    logger.exception(
                        "Failed loading YOLO model for %s (generation=%d)",
                        camera_name,
                        current_generation,
                    )
                    self.stats.inc_errors()
                    time.sleep(1.0)
                    continue

            try:
                frame_packet = frame_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                assert detector is not None
                self._process_frame(detector, face_detector, camera_id, frame_packet)
            except Exception:
                logger.exception("Failed processing frame from %s", camera_name)
                self.stats.inc_errors()

    def _start_workers(self) -> None:
        """Spawn paired poller/processor threads per camera."""
        if self.notifier.enabled and self.status_report_thread is None:
            self.status_report_thread = threading.Thread(
                target=self._run_periodic_status_reports,
                name="status-reporter",
                daemon=True,
            )
            self.status_report_thread.start()
            logger.info("Started periodic status reporter thread")

        for camera_id, camera in self.cameras.items():
            poller = threading.Thread(
                target=self._poll_camera,
                args=(camera_id, camera),
                name=f"poller-{camera_id}",
                daemon=True,
            )
            processor = threading.Thread(
                target=self._process_camera,
                args=(camera_id,),
                name=f"processor-{camera_id}",
                daemon=True,
            )
            poller.start()
            processor.start()
            self.poller_threads[camera_id] = poller
            self.processor_threads[camera_id] = processor
            logger.info("Started worker threads for camera=%s channel=%s", camera.camera.name, camera_id)

    def _stop_workers(self) -> None:
        """Stop worker threads and close external resources."""
        self.stop_event.set()
        if self.status_report_thread is not None:
            self.status_report_thread.join(timeout=2)
            logger.info("Stopped periodic status reporter thread")
        for thread in self.poller_threads.values():
            thread.join(timeout=2)
        for thread in self.processor_threads.values():
            thread.join(timeout=2)
        for camera in self.cameras.values():
            camera.release()
        self.notifier.close()
        self.face_recognizer.close()
        logger.info("All workers and resources stopped")

    def run(self) -> None:
        """Run service until interrupted from main thread."""
        logger.info(
            "Starting surveillance with %d cameras in %s mode (poller+processor per camera)",
            len(self.cameras),
            self.settings.capture_mode,
        )
        self._start_workers()
        try:
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("Interrupted by user, stopping workers...")
            self.stop_event.set()
        finally:
            self._stop_workers()

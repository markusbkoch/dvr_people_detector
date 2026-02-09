from __future__ import annotations

"""Configuration loading and camera URL construction.

This module centralizes environment/.secrets parsing and maps static channel
metadata to concrete stream URLs used by the runtime.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_secrets_file(path: Path) -> Dict[str, str]:
    """Parse KEY=VALUE lines from the local secrets file.

    The parser is intentionally permissive:
    - ignores blank lines/comments
    - accepts surrounding whitespace around keys/values
    - strips both single and double wrapping quotes
    """
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class CameraConfig:
    """Resolved camera endpoints and display metadata."""

    channel_key: int
    name: str
    rtsp_url: str
    isapi_url: str


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment and/or `.secrets`."""

    dvr_username: str
    dvr_password: str
    dvr_ip: str
    telegram_bot_token: str
    telegram_chat_id: str
    status_report_interval_hours: float
    detect_every_seconds: float
    rtsp_detection_fps: float
    frame_queue_size: int
    periodic_alert_seconds: float
    confidence_threshold: float
    low_conf_log_min_confidence: float
    low_conf_review_cooldown_seconds: float
    person_min_box_area_px: int
    person_min_movement_px: float
    detection_confirmation_frames: int
    detection_confirmation_window_seconds: float
    alert_cooldown_seconds: int
    burst_capture_seconds: float
    burst_max_frames: int
    snapshot_target_aspect_ratio: str
    snapshot_dir: Path
    live_preview_fps: float
    face_db_path: Path
    face_match_threshold: float
    face_min_samples: int
    camera_reconnect_seconds: float
    capture_mode: str
    rtsp_transport: str
    person_tracker: str
    isapi_timeout_seconds: float
    isapi_auth_mode: str
    yolo_model: str
    ignored_person_bboxes: Dict[int, List[Tuple[int, int, int, int]]]
    camera_channels: Dict[int, str]


def _get_env(name: str, file_values: Dict[str, str], default: str = "") -> str:
    """Read a setting from env first, then file, then default."""
    return os.getenv(name, file_values.get(name, default)).strip()


def _parse_ignored_person_bboxes(raw: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """Parse camera-specific ignore boxes.

    Format:
    - `channel:x1,y1,x2,y2`
    - multiple boxes per camera: `channel:x1,y1,x2,y2|x1,y1,x2,y2`
    - multiple cameras: separate entries with `;`

    Example:
    `501:304,180,316,205;401:10,20,30,40|100,120,180,220`
    """
    parsed: Dict[int, List[Tuple[int, int, int, int]]] = {}
    text = raw.strip()
    if not text:
        return parsed

    for entry in text.split(";"):
        chunk = entry.strip()
        if not chunk or ":" not in chunk:
            continue
        channel_raw, boxes_raw = chunk.split(":", 1)
        try:
            channel_id = int(channel_raw.strip())
        except ValueError:
            continue

        boxes: List[Tuple[int, int, int, int]] = []
        for box_raw in boxes_raw.split("|"):
            coords = [part.strip() for part in box_raw.split(",")]
            if len(coords) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(int, coords)
            except ValueError:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
        if boxes:
            parsed[channel_id] = boxes

    return parsed


def _parse_camera_channels(raw: str) -> Dict[int, str]:
    """Parse camera channel map from `.secrets`.

    Format:
    - `channel_id:name` entries separated by `;`
    - Example: `101:Camera1;201:Camera2`
    """
    parsed: Dict[int, str] = {}
    text = raw.strip()
    if not text:
        return parsed

    for entry in text.split(";"):
        chunk = entry.strip()
        if not chunk or ":" not in chunk:
            continue
        channel_raw, name_raw = chunk.split(":", 1)
        try:
            channel_id = int(channel_raw.strip())
        except ValueError:
            continue
        name = name_raw.strip()
        if not name:
            continue
        parsed[channel_id] = name
    return parsed


def load_settings(secrets_path: str = ".secrets") -> Settings:
    """Load and validate app settings.

    Required DVR credentials raise `ValueError` when missing, while operational
    tuning flags fall back to safe defaults.
    """
    file_values = _parse_secrets_file(Path(secrets_path))

    dvr_username = _get_env("DVR_USERNAME", file_values)
    dvr_password = _get_env("DVR_PASSWORD", file_values)
    dvr_ip = _get_env("DVR_IP", file_values)
    camera_channels_raw = _get_env("CAMERA_CHANNELS", file_values)
    camera_channels = _parse_camera_channels(camera_channels_raw)

    if not dvr_username or not dvr_password or not dvr_ip:
        raise ValueError(
            "Missing DVR credentials. Set DVR_USERNAME, DVR_PASSWORD and DVR_IP in .secrets or environment."
        )
    if not camera_channels:
        raise ValueError(
            "Missing CAMERA_CHANNELS. Set CAMERA_CHANNELS in .secrets or environment "
            "(format: channel_id:name;channel_id:name)."
        )

    snapshot_dir = Path(_get_env("SNAPSHOT_DIR", file_values, "data/snapshots"))
    low_conf_review_cooldown_seconds = _get_env("LOW_CONF_REVIEW_COOLDOWN_SECONDS", file_values, "15")
    return Settings(
        dvr_username=dvr_username,
        dvr_password=dvr_password,
        dvr_ip=dvr_ip,
        telegram_bot_token=_get_env("TELEGRAM_BOT_TOKEN", file_values),
        telegram_chat_id=_get_env("TELEGRAM_CHAT_ID", file_values),
        status_report_interval_hours=float(_get_env("STATUS_REPORT_INTERVAL_HOURS", file_values, "12")),
        detect_every_seconds=float(_get_env("DETECT_EVERY_SECONDS", file_values, "1.2")),
        rtsp_detection_fps=float(_get_env("RTSP_DETECTION_FPS", file_values, "6.0")),
        frame_queue_size=int(_get_env("FRAME_QUEUE_SIZE", file_values, "120")),
        periodic_alert_seconds=float(_get_env("PERIODIC_ALERT_SECONDS", file_values, "-1")),
        confidence_threshold=float(_get_env("MIN_PERSON_CONFIDENCE_FOR_ALERT", file_values, "0.65")),
        low_conf_log_min_confidence=float(_get_env("MIN_PERSON_CONFIDENCE_FOR_LOW_CONF_REVIEW", file_values, "0.05")),
        low_conf_review_cooldown_seconds=float(low_conf_review_cooldown_seconds),
        person_min_box_area_px=int(_get_env("PERSON_MIN_BOX_AREA_PX", file_values, "0")),
        person_min_movement_px=float(_get_env("PERSON_MIN_MOVEMENT_PX", file_values, "0")),
        detection_confirmation_frames=int(_get_env("DETECTION_CONFIRMATION_FRAMES", file_values, "3")),
        detection_confirmation_window_seconds=float(
            _get_env("DETECTION_CONFIRMATION_WINDOW_SECONDS", file_values, "1.5")
        ),
        alert_cooldown_seconds=int(_get_env("ALERT_COOLDOWN_SECONDS", file_values, "40")),
        burst_capture_seconds=float(_get_env("BURST_CAPTURE_SECONDS", file_values, "4.0")),
        burst_max_frames=int(_get_env("BURST_MAX_FRAMES", file_values, "12")),
        snapshot_target_aspect_ratio=_get_env("SNAPSHOT_TARGET_ASPECT_RATIO", file_values, "16:9"),
        snapshot_dir=snapshot_dir,
        live_preview_fps=float(_get_env("LIVE_PREVIEW_FPS", file_values, "4.0")),
        face_db_path=Path(_get_env("FACE_DB_PATH", file_values, "data/faces.db")),
        face_match_threshold=float(_get_env("FACE_MATCH_THRESHOLD", file_values, "0.80")),
        face_min_samples=int(_get_env("FACE_MIN_SAMPLES", file_values, "3")),
        camera_reconnect_seconds=float(_get_env("CAMERA_RECONNECT_SECONDS", file_values, "5")),
        capture_mode=_get_env("CAPTURE_MODE", file_values, "isapi").lower(),
        rtsp_transport=_get_env("RTSP_TRANSPORT", file_values, "tcp").lower(),
        person_tracker=_get_env("PERSON_TRACKER", file_values, "bytetrack").lower(),
        isapi_timeout_seconds=float(_get_env("ISAPI_TIMEOUT_SECONDS", file_values, "4")),
        isapi_auth_mode=_get_env("ISAPI_AUTH_MODE", file_values, "auto").lower(),
        yolo_model=_get_env("YOLO_MODEL", file_values, "detection_models/yolov8n.pt"),
        ignored_person_bboxes=_parse_ignored_person_bboxes(_get_env("IGNORED_PERSON_BBOXES", file_values, "")),
        camera_channels=camera_channels,
    )


def build_camera_map(settings: Settings) -> Dict[int, CameraConfig]:
    """Build static camera channel definitions for this deployment."""
    channels = settings.camera_channels

    camera_map: Dict[int, CameraConfig] = {}
    for channel_id, name in channels.items():
        rtsp_url = (
            f"rtsp://{settings.dvr_username}:{settings.dvr_password}@{settings.dvr_ip}:554/"
            f"Streaming/Channels/{channel_id}"
        )
        isapi_url = f"http://{settings.dvr_ip}/ISAPI/Streaming/channels/{channel_id}/picture"
        camera_map[channel_id] = CameraConfig(
            channel_key=channel_id,
            name=name,
            rtsp_url=rtsp_url,
            isapi_url=isapi_url,
        )

    return camera_map

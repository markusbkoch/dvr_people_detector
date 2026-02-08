from __future__ import annotations

"""Camera client implementations for RTSP streams and ISAPI snapshots."""

import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from app.config import CameraConfig


@dataclass
class CameraFrame:
    """Single decoded frame plus source camera metadata and capture timestamp."""

    camera: CameraConfig
    frame: np.ndarray
    captured_at: float


class CameraClient(Protocol):
    """Common interface used by the app regardless of transport/backend."""

    camera: CameraConfig

    def read(self) -> Optional[CameraFrame]:
        ...

    def release(self) -> None:
        ...


class RtspCamera:
    """OpenCV/FFmpeg RTSP reader with reconnect and low-buffer settings."""

    def __init__(self, camera: CameraConfig, reconnect_seconds: float, rtsp_transport: str = "tcp") -> None:
        self.camera = camera
        self.reconnect_seconds = reconnect_seconds
        self.rtsp_transport = rtsp_transport
        self._capture: Optional[cv2.VideoCapture] = None

    def _open(self) -> None:
        """Open the RTSP capture handle with transport-specific FFmpeg options."""
        # Prefer decode stability over ultra-low-latency buffering, which can
        # increase HEVC reference-frame decode errors on noisy links.
        options = f"rtsp_transport;{self.rtsp_transport}"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
        try:
            cv2.setLogLevel(2)
        except Exception:
            pass
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        self._capture = cv2.VideoCapture(self.camera.rtsp_url, cv2.CAP_FFMPEG)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _ensure_open(self) -> None:
        """Lazily open the capture handle if it is not available."""
        if self._capture is not None and self._capture.isOpened():
            return
        self._open()

    def read(self) -> Optional[CameraFrame]:
        """Read one frame, or trigger reconnect on failure."""
        self._ensure_open()
        if self._capture is None:
            return None

        ok, frame = self._capture.read()
        if ok and frame is not None:
            return CameraFrame(camera=self.camera, frame=frame, captured_at=time.time())

        self.release()
        time.sleep(self.reconnect_seconds)
        return None

    def release(self) -> None:
        """Release underlying OpenCV resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class IsapiSnapshotCamera:
    """HTTP snapshot client for DVR/NVR ISAPI endpoints."""

    def __init__(
        self,
        camera: CameraConfig,
        username: str,
        password: str,
        timeout_seconds: float,
        reconnect_seconds: float,
        auth_mode: str = "auto",
    ) -> None:
        self.camera = camera
        self.timeout_seconds = timeout_seconds
        self.reconnect_seconds = reconnect_seconds
        self._session = requests.Session()
        self._auth_mode = auth_mode

        if auth_mode == "basic":
            self._auth = HTTPBasicAuth(username, password)
        elif auth_mode == "digest":
            self._auth = HTTPDigestAuth(username, password)
        else:
            # Hikvision typically uses digest auth.
            self._auth = HTTPDigestAuth(username, password)

    def read(self) -> Optional[CameraFrame]:
        """Fetch and decode one JPEG snapshot from ISAPI."""
        if not self.camera.isapi_url:
            return None

        try:
            response = self._session.get(
                self.camera.isapi_url,
                auth=self._auth,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            data = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            return CameraFrame(camera=self.camera, frame=frame, captured_at=time.time())
        except requests.RequestException:
            time.sleep(self.reconnect_seconds)
            return None

    def release(self) -> None:
        """Close the underlying requests session."""
        self._session.close()

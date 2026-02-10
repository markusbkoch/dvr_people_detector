from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.app import SurveillanceApp
from app.config import build_camera_map, load_settings
from app.web_ui import start_gallery_server


class _SensitiveDataFilter(logging.Filter):
    """Redact sensitive Telegram bot token fragments from log messages."""

    _TELEGRAM_BOT_PATH_RE = re.compile(r"(https://api\.telegram\.org/bot)[^/\s]+")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = self._TELEGRAM_BOT_PATH_RE.sub(r"\1<redacted>", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging() -> None:
    """Configure console + rotating file logging."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "surveillance.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_SensitiveDataFilter())

    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_SensitiveDataFilter())

    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # Avoid third-party HTTP request logging that can leak full Telegram URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    settings = load_settings(".secrets")
    app = SurveillanceApp(settings)
    gallery_host = os.getenv("GALLERY_HOST", "127.0.0.1")
    gallery_port = int(os.getenv("GALLERY_PORT", "8765"))
    gallery_db_path = Path(os.getenv("GALLERY_DB_PATH", str(settings.face_db_path)))
    gallery_snapshot_dir = Path(os.getenv("GALLERY_SNAPSHOT_DIR", str(settings.snapshot_dir)))
    gallery_server, _ = start_gallery_server(
        host=gallery_host,
        port=gallery_port,
        db_path=gallery_db_path,
        snapshot_dir=gallery_snapshot_dir,
        camera_map=build_camera_map(settings),
        settings=settings,
        live_frame_provider=app.get_live_preview_frame,
        model_importer=app.import_model_file,
        model_status_provider=app.get_model_management_status,
        stats_provider=app.get_stats,
    )
    logging.info("Face gallery: http://%s:%d", gallery_host, gallery_port)
    try:
        app.run()
    finally:
        gallery_server.shutdown()
        gallery_server.server_close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting.")

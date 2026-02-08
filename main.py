from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.app import SurveillanceApp
from app.config import build_camera_map, load_settings
from scripts.face_gallery import start_gallery_server


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

    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


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
        live_preview_dir=settings.live_preview_dir,
        camera_map=build_camera_map(settings),
        settings=settings,
        live_frame_provider=app.get_live_preview_frame,
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

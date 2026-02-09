from __future__ import annotations

"""Local web UI for reviewing snapshots and managing face/person datasets."""

import html
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]

from app.face_rules import FaceEmbeddingEngine
from app.config import CameraConfig, Settings


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class FaceGalleryHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing review and moderation routes."""

    db_path: Path = Path("data/faces.db")
    snapshot_dir: Path = Path("data/snapshots")
    live_frame_provider: Optional[Callable[[int], Optional[tuple[int, bytes]]]] = None
    camera_map: dict[int, CameraConfig] = {}
    settings: Settings | None = None

    def _connect(self) -> sqlite3.Connection:
        """Open SQLite connection and ensure required tables exist."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS detector_reviews (
                sample_id INTEGER PRIMARY KEY,
                label TEXT NOT NULL CHECK(label IN ('person', 'no_person')),
                notes TEXT,
                reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS snapshot_reviews (
                source_image TEXT PRIMARY KEY,
                detector_label TEXT CHECK(detector_label IN ('person', 'no_person') OR detector_label IS NULL),
                person_id TEXT,
                notes TEXT,
                reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        return conn

    def _send_html(self, body: str, status: int = 200) -> None:
        """Send UTF-8 HTML response."""
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_binary(self, payload: bytes, content_type: str) -> None:
        """Send binary payload (e.g., image bytes)."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        """Issue HTTP redirect response."""
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _guess_type(self, path: Path) -> str:
        """Infer media content-type from file extension."""
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        return "application/octet-stream"

    def _all_snapshots(self) -> list[Path]:
        """Return all snapshot image files sorted newest-first by file timestamp."""
        if not self.snapshot_dir.exists():
            return []
        files = [p for p in self.snapshot_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        files.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        return files

    def _resolve_person_id(self, conn: sqlite3.Connection, identifier: str) -> str | None:
        """Resolve user input to a person_id via id or unique display name."""
        value = (identifier or "").strip()
        if not value:
            return None

        row = conn.execute("SELECT person_id FROM persons WHERE person_id = ?", (value,)).fetchone()
        if row is not None:
            return str(row["person_id"])

        rows = conn.execute(
            "SELECT person_id FROM persons WHERE lower(COALESCE(display_name, '')) = lower(?)",
            (value,),
        ).fetchall()
        if len(rows) == 1:
            return str(rows[0]["person_id"])
        return None

    def _create_person_from_input(self, conn: sqlite3.Connection, raw: str) -> str:
        """Create a new person entry from free-form input."""
        value = (raw or "").strip()
        if not value:
            raise ValueError("empty person identifier")

        if re.fullmatch(r"[A-Za-z0-9_\-]+", value):
            person_id = value
            display_name = value
        else:
            base = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "person"
            person_id = base
            suffix = 1
            while conn.execute("SELECT 1 FROM persons WHERE person_id = ?", (person_id,)).fetchone() is not None:
                suffix += 1
                person_id = f"{base}_{suffix}"
            display_name = value

        conn.execute(
            """
            INSERT INTO persons (person_id, display_name)
            VALUES (?, ?)
            ON CONFLICT(person_id) DO UPDATE SET display_name = COALESCE(persons.display_name, excluded.display_name)
            """,
            (person_id, display_name),
        )
        return person_id

    def _resolve_image_path(self, path_param: str) -> Path | None:
        """Resolve and validate image path under workspace root."""
        if not path_param:
            return None
        raw = Path(unquote(path_param))
        if raw.is_absolute():
            resolved = raw.resolve()
        else:
            resolved = (ROOT / raw).resolve()

        root_resolved = ROOT.resolve()
        if root_resolved not in resolved.parents and resolved != root_resolved:
            return None
        if not resolved.exists() or not resolved.is_file():
            return None
        return resolved

    def _render_root(self) -> str:
        """Render landing page with links to main review workflows."""
        with self._connect() as conn:
            person_count = int(conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0])
            face_sample_count = int(conn.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0])
            reviewed_snapshots = int(
                conn.execute("SELECT COUNT(*) FROM snapshot_reviews WHERE detector_label IS NOT NULL").fetchone()[0]
            )
        snapshot_count = len(self._all_snapshots())

        return f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Face & Snapshot Review</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #111; }}
            .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
            .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; text-decoration: none; color: inherit; }}
            .title {{ font-weight: 700; margin-bottom: 6px; }}
            .sub {{ color: #555; }}
          </style>
        </head>
        <body>
          <h1>Face & Snapshot Review</h1>
          <div class="cards">
            <a class="card" href="/snapshots">
              <div class="title">Snapshot Review</div>
              <div class="sub">{snapshot_count} snapshots · {reviewed_snapshots} reviewed</div>
            </a>
            <a class="card" href="/people">
              <div class="title">People Gallery</div>
              <div class="sub">{person_count} people · {face_sample_count} face samples</div>
            </a>
            <a class="card" href="/live">
              <div class="title">Live Feed</div>
              <div class="sub">{len(self.camera_map)} camera feeds in browser</div>
            </a>
            <a class="card" href="/guide">
              <div class="title">User Guide</div>
              <div class="sub">System setup, parameters, review workflow and training export</div>
            </a>
          </div>
        </body>
        </html>
        """

    def _render_live(self) -> str:
        """Render browser page with live MJPEG streams from shared preview frames."""
        tiles = []
        camera_items: list[tuple[int, str]] = []
        for channel_id in sorted(self.camera_map):
            camera = self.camera_map[channel_id]
            thumb_src = f"/live/frame?channel={channel_id}"
            camera_items.append((channel_id, camera.name))
            tiles.append(
                f"""
                <button class="thumb" type="button" data-channel="{channel_id}" data-name="{html.escape(camera.name)}">
                  <img src="{thumb_src}" alt="live {html.escape(camera.name)}" loading="lazy" />
                  <span class="thumb-label">{html.escape(camera.name)} ({channel_id})</span>
                </button>
                """
            )
        camera_list_js = ",".join(
            f"{{channel:{channel_id},name:{camera_name!r}}}" for channel_id, camera_name in camera_items
        )
        return f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Live Feed</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #111; }}
            a {{ color: #2563eb; text-decoration: none; }}
            .viewer {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }}
            .viewer img {{ width: 100%; max-height: 70vh; object-fit: contain; background: #111; border-radius: 10px; }}
            .viewer-top {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 8px; }}
            .viewer-title {{ font-weight: 700; }}
            .viewer-help {{ color: #555; font-size: 0.9rem; }}
            .thumbs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; margin-top: 12px; }}
            .thumb {{ border: 2px solid #d1d5db; background: white; border-radius: 10px; padding: 6px; cursor: pointer; text-align: left; }}
            .thumb.active {{ border-color: #2563eb; box-shadow: 0 0 0 2px #bfdbfe; }}
            .thumb img {{ width: 100%; height: 108px; object-fit: cover; border-radius: 6px; background: #111; display: block; }}
            .thumb-label {{ display: block; margin-top: 5px; font-size: 0.88rem; color: #333; }}
            .title {{ font-weight: 700; margin-bottom: 8px; }}
            .sub {{ font-weight: 400; color: #555; }}
          </style>
        </head>
        <body>
          <p><a href="/">← Home</a> · <a href="/snapshots">Snapshot Review</a> · <a href="/people">People Gallery</a></p>
          <h1>Live Feed</h1>
          <p>Feeds are served from the running surveillance process.</p>
          {'' if camera_items else '<p>No cameras configured.</p>'}
          <section class="viewer" id="viewer" style="display:{'block' if camera_items else 'none'};">
            <div class="viewer-top">
              <div class="viewer-title" id="viewer-title"></div>
              <div class="viewer-help">Use ← / → to switch cameras</div>
            </div>
            <img id="viewer-image" src="" alt="live focused camera" loading="eager" />
          </section>
          <div class="thumbs">{''.join(tiles)}</div>
          <script>
            const cameras = [{camera_list_js}];
            let currentIndex = 0;
            let streamNonce = 0;
            const titleEl = document.getElementById("viewer-title");
            const imageEl = document.getElementById("viewer-image");
            const thumbEls = Array.from(document.querySelectorAll(".thumb"));

            function streamUrl(channel) {{
              streamNonce += 1;
              return "/live/stream?channel=" + encodeURIComponent(channel) + "&v=" + streamNonce;
            }}
            function frameUrl(channel) {{
              streamNonce += 1;
              return "/live/frame?channel=" + encodeURIComponent(channel) + "&v=" + streamNonce;
            }}
            function refreshThumbs() {{
              thumbEls.forEach((el) => {{
                const img = el.querySelector("img");
                const channel = Number(el.dataset.channel || "0");
                if (!img || !channel) return;
                img.src = frameUrl(channel);
              }});
            }}

            function setCurrent(index) {{
              if (!cameras.length) return;
              currentIndex = (index + cameras.length) % cameras.length;
              const camera = cameras[currentIndex];
              titleEl.textContent = camera.name + " (" + camera.channel + ")";
              // Force stream teardown/reconnect to guarantee camera switch.
              imageEl.src = "";
              requestAnimationFrame(() => {{
                imageEl.src = streamUrl(camera.channel);
              }});
              thumbEls.forEach((el, idx) => el.classList.toggle("active", idx === currentIndex));
            }}

            thumbEls.forEach((el, idx) => {{
              el.addEventListener("click", () => setCurrent(idx));
            }});

            document.addEventListener("keydown", (event) => {{
              if (event.key === "ArrowRight") {{
                event.preventDefault();
                setCurrent(currentIndex + 1);
              }} else if (event.key === "ArrowLeft") {{
                event.preventDefault();
                setCurrent(currentIndex - 1);
              }}
            }});

            setCurrent(0);
            setInterval(refreshThumbs, 1500);
          </script>
        </body>
        </html>
        """

    def _send_live_frame(self, channel_id: int) -> None:
        """Send a single JPEG frame for one camera."""
        camera = self.camera_map.get(channel_id)
        if camera is None:
            self.send_error(404, "Unknown camera channel")
            return
        provider = self.live_frame_provider
        if provider is None:
            self.send_error(503, "Live frame provider unavailable")
            return
        packet = provider(channel_id)
        if packet is None:
            self.send_response(204)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            return
        _, payload = packet
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def _send_mjpeg_stream(self, channel_id: int) -> None:
        """Stream one camera preview as MJPEG multipart response."""
        camera = self.camera_map.get(channel_id)
        if camera is None:
            self.send_error(404, "Unknown camera channel")
            return

        boundary = "frame"
        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        last_seq = -1
        provider = self.live_frame_provider
        if provider is None:
            self.send_error(503, "Live frame provider unavailable")
            return
        while True:
            packet = provider(channel_id)
            if packet is None:
                time.sleep(0.05)
                continue
            seq, payload = packet
            if seq == last_seq:
                time.sleep(0.05)
                continue
            last_seq = seq

            try:
                self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def _render_guide(self) -> str:
        """Render HTML user guide for surveillance and review workflows."""
        return """
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>User Guide</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #111; line-height: 1.45; }
            h1, h2 { margin: 0 0 10px 0; }
            h2 { margin-top: 18px; }
            p { margin: 8px 0; color: #333; }
            ul { margin: 8px 0 12px 20px; }
            li { margin: 4px 0; }
            code { background: #eef2f7; padding: 2px 6px; border-radius: 6px; }
            pre { background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 10px; overflow-x: auto; }
            .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; margin-top: 10px; }
            table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; }
            th, td { text-align: left; padding: 10px; border-bottom: 1px solid #eef2f7; vertical-align: top; }
            th { background: #f8fafc; font-weight: 600; }
            tr:last-child td { border-bottom: none; }
            a { color: #2563eb; text-decoration: none; }
          </style>
        </head>
        <body>
          <p><a href="/">← Home</a> · <a href="/snapshots">Snapshot Review</a> · <a href="/people">People Gallery</a></p>
          <h1>Surveillance User Guide</h1>
          <p>This page documents how to run and operate the surveillance system, the snapshot review workflow, and person classification tools.</p>

          <div class="card">
            <h2>1. Start The System</h2>
            <p>Run the surveillance pipeline (this also serves the gallery UI):</p>
            <pre>source .venv/bin/activate
python main.py</pre>
          </div>

          <h2>2. Important Parameters (.secrets)</h2>
          <table>
            <tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr>
            <tr><td><code>DVR_USERNAME</code>, <code>DVR_PASSWORD</code>, <code>DVR_IP</code></td><td><code>(required)</code></td><td>DVR access credentials and host.</td></tr>
            <tr><td><code>CAMERA_CHANNELS</code></td><td><code>(required)</code></td><td>Runtime channel map in <code>channel_id:name</code> format separated by <code>;</code>.</td></tr>
            <tr><td><code>CAPTURE_MODE</code></td><td><code>rtsp</code></td><td><code>isapi</code> or <code>rtsp</code>.</td></tr>
            <tr><td><code>PERSON_TRACKER</code></td><td><code>bytetrack</code></td><td>RTSP tracker backend: <code>bytetrack</code>, <code>botsort</code>, or <code>none</code>.</td></tr>
            <tr><td><code>MIN_PERSON_CONFIDENCE_FOR_ALERT</code></td><td><code>0.5</code></td><td>Minimum confidence to treat a person detection as positive for alerting.</td></tr>
            <tr><td><code>MIN_PERSON_CONFIDENCE_FOR_LOW_CONF_REVIEW</code></td><td><code>0.15</code></td><td>Minimum confidence for low-confidence suppression log lines and low-confidence review snapshots.</td></tr>
            <tr><td><code>LOW_CONF_REVIEW_COOLDOWN_SECONDS</code></td><td><code>15</code></td><td>Minimum seconds between low-confidence snapshot saves per camera. Saved to <code>SNAPSHOT_DIR</code> as <code>lowconf_*</code>.</td></tr>
            <tr><td><code>PERSON_MIN_BOX_AREA_PX</code></td><td><code>1200</code></td><td>Minimum person box area in pixels. Set <code>0</code> to disable.</td></tr>
            <tr><td><code>PERSON_MIN_MOVEMENT_PX</code></td><td><code>5</code></td><td>Minimum person-box center movement across confirmation window. Set <code>0</code> to disable.</td></tr>
            <tr><td><code>DETECTION_CONFIRMATION_FRAMES</code></td><td><code>3</code></td><td>Positive detection frames required before alert flow starts.</td></tr>
            <tr><td><code>DETECTION_CONFIRMATION_WINDOW_SECONDS</code></td><td><code>1.5</code></td><td>Max gap between confirmation hits for one camera.</td></tr>
            <tr><td><code>IGNORED_PERSON_BBOXES</code></td><td><code>(empty)</code></td><td>Optional camera ignore zones, format <code>channel:x1,y1,x2,y2;channel:x1,y1,x2,y2|x1,y1,x2,y2</code>.</td></tr>
            <tr><td><code>DETECT_EVERY_SECONDS</code></td><td><code>1.2</code></td><td>Polling interval per camera.</td></tr>
            <tr><td><code>RTSP_DETECTION_FPS</code></td><td><code>12.0</code></td><td>Target per-camera inference rate in RTSP mode (continuous ingest + sampled detection).</td></tr>
            <tr><td><code>ALERT_COOLDOWN_SECONDS</code></td><td><code>40</code></td><td>Detection alert cooldown per camera.</td></tr>
            <tr><td><code>PERIODIC_ALERT_SECONDS</code></td><td><code>-1</code></td><td><code>-1</code> means detect-only alerts; positive values trigger periodic snapshots.</td></tr>
            <tr><td><code>BURST_CAPTURE_SECONDS</code></td><td><code>4.0</code></td><td>Burst time window used to select better frames before alerting.</td></tr>
            <tr><td><code>BURST_MAX_FRAMES</code></td><td><code>12</code></td><td>Maximum frames considered in one burst window.</td></tr>
            <tr><td><code>FRAME_QUEUE_SIZE</code></td><td><code>120</code></td><td>Per-camera queue capacity between polling and processing.</td></tr>
            <tr><td><code>SNAPSHOT_TARGET_ASPECT_RATIO</code></td><td><code>16:9</code></td><td>Optional aspect correction before saving snapshots.</td></tr>
            <tr><td><code>SNAPSHOT_DIR</code></td><td><code>data/snapshots</code></td><td>Where snapshots are saved.</td></tr>
            <tr><td><code>RTSP_TRANSPORT</code></td><td><code>tcp</code></td><td>RTSP transport mode for OpenCV/FFmpeg capture.</td></tr>
            <tr><td><code>LIVE_PREVIEW_FPS</code></td><td><code>4.0</code></td><td>Max in-memory publish rate (per camera) for web live previews in <code>main.py</code>. Set <code>0</code> to disable.</td></tr>
            <tr><td><code>CAMERA_RECONNECT_SECONDS</code></td><td><code>5</code></td><td>Delay before retrying failed camera reads.</td></tr>
            <tr><td><code>ISAPI_AUTH_MODE</code></td><td><code>auto</code></td><td>ISAPI auth strategy (<code>auto|digest|basic</code>).</td></tr>
            <tr><td><code>ISAPI_TIMEOUT_SECONDS</code></td><td><code>4</code></td><td>ISAPI HTTP timeout per request.</td></tr>
            <tr><td><code>YOLO_MODEL</code></td><td><code>detection_models/yolov8n.pt</code></td><td>Detector model used by the surveillance pipeline.</td></tr>
            <tr><td><code>FACE_DB_PATH</code></td><td><code>data/faces.db</code></td><td>SQLite database for face identities and samples.</td></tr>
            <tr><td><code>FACE_MATCH_THRESHOLD</code></td><td><code>0.80</code></td><td>Minimum cosine similarity to consider a face match.</td></tr>
            <tr><td><code>FACE_MIN_SAMPLES</code></td><td><code>3</code></td><td>Minimum samples per person before matching is considered valid.</td></tr>
            <tr><td><code>TELEGRAM_BOT_TOKEN</code>, <code>TELEGRAM_CHAT_ID</code></td><td><code>(optional)</code></td><td>Enables Telegram alerts and bot commands (<code>/status</code>, <code>/ping</code>, <code>/setstatus</code>).</td></tr>
            <tr><td><code>STATUS_REPORT_INTERVAL_HOURS</code></td><td><code>12</code></td><td>Periodic status-report interval in hours. Set <code>0</code> to disable.</td></tr>
          </table>

          <h2>3. Snapshot Review Workflow (Loop 1)</h2>
          <div class="card">
            <ul>
              <li>Go to <a href="/snapshots">Snapshot Review</a>.</li>
              <li>For each snapshot, classify detector label as <code>Person</code> or <code>No Person</code>.</li>
              <li>Use bulk multi-select to label many thumbnails at once.</li>
              <li>If a face is visible, assign a person name/id to improve identity data.</li>
            </ul>
          </div>

          <h2>4. Person Gallery Workflow (Loop 2)</h2>
          <div class="card">
            <ul>
              <li>Go to <a href="/people">People Gallery</a> and open a person.</li>
              <li>Rename the person display name.</li>
              <li>Move misclassified samples to another person.</li>
              <li>Delete incorrect samples or delete a person entirely.</li>
            </ul>
          </div>

          <h2>5. Face Data Flow</h2>
          <div class="card">
            <ul>
              <li>When alerts are generated, the runtime already extracts and stores face samples in <code>face_samples</code>.</li>
              <li>Use <a href="/snapshots">Snapshot Review</a> to label detector outcomes (<code>person</code>/<code>no_person</code>).</li>
              <li>Use <a href="/people">People Gallery</a> to rename identities, move misclassified samples, and delete bad samples.</li>
              <li>No separate bootstrap step is required for normal operation.</li>
            </ul>
          </div>

          <h2>6. Export Detector Training Dataset</h2>
          <pre>python scripts/export_detector_dataset.py \\
  --db-path data/faces.db \\
  --output-dir data/detector_dataset \\
  --model detection_models/yolov8n.pt \\
  --confidence 0.35 \\
  --clean</pre>
          <p>Output includes YOLO-compatible <code>images/</code>, <code>labels/</code>, and <code>dataset.yaml</code>.</p>

          <h2>7. Telegram Runtime Commands</h2>
          <div class="card">
            <ul>
              <li><code>/ping</code>: liveness check</li>
              <li><code>/status</code>: current app status and since-last-report counters</li>
              <li><code>/noperson</code> (reply to a snapshot): marks that snapshot as <code>no_person</code></li>
              <li><code>/pause</code>: pause snapshot/photo alerts indefinitely (status reports still run)</li>
              <li><code>/pause &lt;hours&gt;</code>: pause snapshot/photo alerts for a fixed duration</li>
              <li><code>/resume</code>: resume Telegram notifications</li>
              <li><code>/reload_model</code>: export reviewed data, train from <code>detection_models/yolov8n_base.pt</code>, promote to managed model files, then reload workers</li>
              <li><code>/reload_model &lt;model_path&gt;</code>: same pipeline using the provided base model path</li>
              <li><code>/setstatus &lt;hours|off&gt;</code>: update periodic status interval at runtime</li>
              <li>React with <code>👎</code> on a snapshot alert to mark it as <code>no_person</code> directly from Telegram.</li>
            </ul>
            <p>Example: <code>/setstatus 6</code> (every 6 hours), <code>/setstatus off</code> (disable periodic status).</p>
          </div>
        </body>
        </html>
        """

    def _render_people(self) -> str:
        """Render people-centric face sample gallery."""
        with self._connect() as conn:
            persons = conn.execute(
                """
                SELECT p.person_id, COALESCE(p.display_name, p.person_id) AS display_name,
                       COUNT(fs.sample_id) AS sample_count,
                       MAX(fs.source_image) AS preview_image
                FROM persons p
                LEFT JOIN face_samples fs ON fs.person_id = p.person_id
                GROUP BY p.person_id, display_name
                ORDER BY sample_count DESC, p.person_id ASC
                """
            ).fetchall()

        cards = []
        for row in persons:
            person_id = str(row["person_id"])
            display_name = html.escape(str(row["display_name"]))
            sample_count = int(row["sample_count"])
            preview = row["preview_image"]
            if preview:
                src = f"/img?path={quote(str(preview))}"
                thumb = f'<img class="thumb" src="{src}" alt="{display_name}" loading="lazy" />'
            else:
                thumb = '<div class="thumb empty">No image</div>'
            link = f"/person/{quote(person_id)}"
            cards.append(
                f"""
                <a class="card" href="{link}">
                  {thumb}
                  <div class="meta">
                    <div class="title">{display_name}</div>
                    <div class="sub">{person_id} · {sample_count} samples</div>
                  </div>
                </a>
                """
            )

        return f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>People Gallery</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #111; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
            .card {{ text-decoration: none; color: inherit; background: white; border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb; }}
            .thumb {{ width: 100%; height: 180px; object-fit: cover; display: block; background: #eee; }}
            .thumb.empty {{ display:flex; align-items:center; justify-content:center; color:#666; font-size:14px; }}
            .meta {{ padding: 10px 12px; }}
            .title {{ font-weight: 600; }}
            .sub {{ margin-top: 2px; color: #666; font-size: 13px; }}
          </style>
        </head>
        <body>
          <p><a href="/">← Home</a> · <a href="/guide">Guide</a></p>
          <h1>People Gallery</h1>
          <div class="grid">{''.join(cards) if cards else '<p>No persons found.</p>'}</div>
        </body>
        </html>
        """

    def _render_snapshots(self) -> str:
        """Render snapshot review page with single and bulk label controls."""
        files = self._all_snapshots()

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_image, detector_label, person_id FROM snapshot_reviews"
            ).fetchall()
        review_by_image = {
            str(Path(str(r["source_image"])).resolve()): {
                "detector_label": (r["detector_label"] or ""),
                "person_id": (r["person_id"] or ""),
            }
            for r in rows
        }

        buckets: dict[str, list[str]] = {
            "unreviewed": [],
            "person": [],
            "no_person": [],
        }

        for path in files:
            source = str(path.resolve())
            src = f"/img?path={quote(source)}"
            info = review_by_image.get(source, {"detector_label": "", "person_id": ""})
            lbl = str(info["detector_label"])
            pid = str(info["person_id"])

            section = "unreviewed"
            if lbl == "person":
                section = "person"
            elif lbl == "no_person":
                section = "no_person"

            buckets[section].append(
                f"""
                <button class="thumb" data-src="{src}" data-source="{html.escape(source)}" data-label="{html.escape(lbl)}" data-person="{html.escape(pid)}">
                  <img src="{src}" alt="snapshot" loading="lazy" />
                </button>
                """
            )

        ordered_sources = [str(p.resolve()) for p in files]
        preferred_sources = [
            s for s in ordered_sources if (review_by_image.get(s, {}).get("detector_label") or "") == ""
        ] + [
            s for s in ordered_sources if (review_by_image.get(s, {}).get("detector_label") or "") == "person"
        ] + [
            s for s in ordered_sources if (review_by_image.get(s, {}).get("detector_label") or "") == "no_person"
        ]
        first_source = preferred_sources[0] if preferred_sources else ""
        first_src = f"/img?path={quote(first_source)}" if files else "/"
        first_review = review_by_image.get(first_source, {"detector_label": "", "person_id": ""})

        return f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Snapshot Review</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #111; }}
            a {{ color: #2563eb; text-decoration: none; }}
            .viewer {{ background: white; border-radius: 12px; border: 1px solid #e5e7eb; padding: 12px; }}
            .viewer img {{ width: 100%; max-height: 72vh; object-fit: contain; display: block; background: #eee; border-radius: 8px; }}
            .meta {{ margin-top: 8px; font-size: 13px; color: #555; word-break: break-all; }}
            .actions {{ margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
            .actions button {{ padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; cursor: pointer; }}
            .actions input {{ padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; min-width: 220px; }}
            .state {{ font-size: 13px; color: #555; }}
            .thumbs {{ margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }}
            .sections {{ margin-top: 14px; display: grid; gap: 18px; }}
            .section {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }}
            .section h2 {{ margin: 0; font-size: 16px; }}
            .section .sub {{ margin-top: 3px; font-size: 13px; color: #666; }}
            .thumb {{ border: 2px solid #d1d5db; border-radius: 8px; background: white; padding: 0; cursor: pointer; overflow: hidden; }}
            .thumb img {{ width: 100%; height: 100px; object-fit: cover; display: block; background: #eee; }}
            .thumb.active {{ border-color: #2563eb; }}
            .thumb.selected {{ outline: 3px solid #2563eb; outline-offset: -3px; }}
            .bulk {{ margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
            .bulk button {{ padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; cursor: pointer; }}
            .bulk .count {{ font-size: 13px; color: #555; min-width: 120px; }}
          </style>
        </head>
        <body>
          <p><a href="/">← Home</a> · <a href="/people">People Gallery</a> · <a href="/guide">Guide</a></p>
          <h1>Snapshot Review</h1>
          <p>{len(files)} snapshots</p>
          {"<p>No snapshots found.</p>" if not files else f'''
          <section class="viewer">
            <a id="viewer-link" href="{first_src}" target="_blank" rel="noopener noreferrer">
              <img id="viewer-image" src="{first_src}" alt="Selected snapshot" />
            </a>
            <div class="meta" id="viewer-source">{html.escape(first_source)}</div>
            <div class="state">Detector label: <span id="detector-state">{html.escape(first_review.get('detector_label') or 'unreviewed')}</span> · Person: <span id="person-state">{html.escape(first_review.get('person_id') or 'none')}</span></div>
            <div class="actions">
              <form method="post" action="/snapshot/review-detector">
                <input type="hidden" name="source_image" id="person-source-1" value="{html.escape(first_source)}" />
                <input type="hidden" name="label" value="person" />
                <button type="submit">Mark Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector">
                <input type="hidden" name="source_image" id="person-source-2" value="{html.escape(first_source)}" />
                <input type="hidden" name="label" value="no_person" />
                <button type="submit">Mark No Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector">
                <input type="hidden" name="source_image" id="person-source-3" value="{html.escape(first_source)}" />
                <input type="hidden" name="label" value="clear" />
                <button type="submit">Clear Label</button>
              </form>
            </div>
            <div class="actions">
              <form method="post" action="/snapshot/assign-person">
                <input type="hidden" name="source_image" id="assign-source" value="{html.escape(first_source)}" />
                <input type="text" name="target_person" placeholder="person_id or person name" />
                <button type="submit">Assign Person</button>
              </form>
              <form method="post" action="/snapshot/assign-person">
                <input type="hidden" name="source_image" id="clear-person-source" value="{html.escape(first_source)}" />
                <input type="hidden" name="target_person" value="" />
                <button type="submit">Clear Assignment</button>
              </form>
            </div>
            <div class="bulk">
              <span class="count">Selected: <span id="selected-count">0</span></span>
              <button type="button" id="select-all">Select All</button>
              <button type="button" id="clear-selected">Clear Selection</button>
              <form method="post" action="/snapshot/review-detector-bulk">
                <input type="hidden" name="selected_sources" id="bulk-sources" value="" />
                <input type="hidden" name="label" value="person" />
                <button type="submit">Bulk Mark Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector-bulk">
                <input type="hidden" name="selected_sources" id="bulk-sources-2" value="" />
                <input type="hidden" name="label" value="no_person" />
                <button type="submit">Bulk Mark No Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector-bulk">
                <input type="hidden" name="selected_sources" id="bulk-sources-3" value="" />
                <input type="hidden" name="label" value="clear" />
                <button type="submit">Bulk Clear Label</button>
              </form>
            </div>
          </section>
          <section class="sections" id="thumbs">
            <div class="section">
              <h2>Not Reviewed</h2>
              <div class="sub">{len(buckets["unreviewed"])} snapshots</div>
              <div class="thumbs">{''.join(buckets["unreviewed"]) if buckets["unreviewed"] else '<p>No snapshots in this section.</p>'}</div>
            </div>
            <div class="section">
              <h2>Person</h2>
              <div class="sub">{len(buckets["person"])} snapshots</div>
              <div class="thumbs">{''.join(buckets["person"]) if buckets["person"] else '<p>No snapshots in this section.</p>'}</div>
            </div>
            <div class="section">
              <h2>No Person</h2>
              <div class="sub">{len(buckets["no_person"])} snapshots</div>
              <div class="thumbs">{''.join(buckets["no_person"]) if buckets["no_person"] else '<p>No snapshots in this section.</p>'}</div>
            </div>
          </section>
          <script>
            (() => {{
              const thumbs = Array.from(document.querySelectorAll('.thumb'));
              const image = document.getElementById('viewer-image');
              const link = document.getElementById('viewer-link');
              const sourceText = document.getElementById('viewer-source');
              const detectorState = document.getElementById('detector-state');
              const personState = document.getElementById('person-state');
              const selectedCount = document.getElementById('selected-count');
              const selectAllBtn = document.getElementById('select-all');
              const clearSelectedBtn = document.getElementById('clear-selected');
              const bulkSources = document.getElementById('bulk-sources');
              const bulkSources2 = document.getElementById('bulk-sources-2');
              const bulkSources3 = document.getElementById('bulk-sources-3');

              const sourceInputs = [
                document.getElementById('person-source-1'),
                document.getElementById('person-source-2'),
                document.getElementById('person-source-3'),
                document.getElementById('assign-source'),
                document.getElementById('clear-person-source'),
              ];

              let idx = 0;
              const selected = new Set();
              let rangeAnchorIndex = null;

              function refreshSelectionUi() {{
                thumbs.forEach((el) => {{
                  const source = el.dataset.source || '';
                  el.classList.toggle('selected', selected.has(source));
                }});
                const joined = Array.from(selected).join(String.fromCharCode(10));
                selectedCount.textContent = String(selected.size);
                bulkSources.value = joined;
                bulkSources2.value = joined;
                bulkSources3.value = joined;
              }}

              function activate(i) {{
                if (i < 0 || i >= thumbs.length) return;
                idx = i;
                thumbs.forEach((el, j) => el.classList.toggle('active', j === idx));
                const selected = thumbs[idx];
                const src = selected.dataset.src || '';
                const source = selected.dataset.source || '';
                const label = selected.dataset.label || 'unreviewed';
                const person = selected.dataset.person || 'none';
                image.src = src;
                link.href = src;
                sourceText.textContent = source;
                detectorState.textContent = label || 'unreviewed';
                personState.textContent = person || 'none';
                sourceInputs.forEach((inp) => inp.value = source);
                selected.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'nearest' }});
              }}

              thumbs.forEach((btn, i) => btn.addEventListener('click', (ev) => {{
                activate(i);
                const source = btn.dataset.source || '';
                if (!source) return;

                if (ev.shiftKey && rangeAnchorIndex !== null) {{
                  const from = Math.min(rangeAnchorIndex, i);
                  const to = Math.max(rangeAnchorIndex, i);
                  for (let j = from; j <= to; j += 1) {{
                    const rangeSource = thumbs[j].dataset.source || '';
                    if (rangeSource) selected.add(rangeSource);
                  }}
                }} else if (ev.metaKey || ev.ctrlKey) {{
                  if (selected.has(source)) selected.delete(source);
                  else selected.add(source);
                }} else {{
                  if (selected.has(source)) selected.delete(source);
                  else selected.add(source);
                }}
                rangeAnchorIndex = i;
                refreshSelectionUi();
              }}));

              selectAllBtn.addEventListener('click', () => {{
                selected.clear();
                thumbs.forEach((el) => {{
                  const source = el.dataset.source || '';
                  if (source) selected.add(source);
                }});
                refreshSelectionUi();
              }});

              clearSelectedBtn.addEventListener('click', () => {{
                selected.clear();
                refreshSelectionUi();
              }});

              document.addEventListener('keydown', (ev) => {{
                const target = ev.target;
                if (target instanceof HTMLElement) {{
                  const tag = target.tagName;
                  const isTypingContext =
                    target.isContentEditable ||
                    tag === 'INPUT' ||
                    tag === 'TEXTAREA' ||
                    tag === 'SELECT';
                  if (isTypingContext) return;
                }}
                if (ev.key === 'ArrowRight') {{
                  ev.preventDefault();
                  activate(Math.min(thumbs.length - 1, idx + 1));
                }} else if (ev.key === 'ArrowLeft') {{
                  ev.preventDefault();
                  activate(Math.max(0, idx - 1));
                }}
              }});
              activate(0);
              refreshSelectionUi();
            }})();
          </script>
          '''}
        </body>
        </html>
        """

    def _render_person(self, person_id: str) -> str:
        """Render detail page for one person and associated face samples."""
        with self._connect() as conn:
            person = conn.execute(
                "SELECT person_id, COALESCE(display_name, person_id) AS display_name FROM persons WHERE person_id = ?",
                (person_id,),
            ).fetchone()
            samples = conn.execute(
                """
                SELECT fs.sample_id, fs.source_image, fs.quality_score, fs.created_at, dr.label AS detector_label
                FROM face_samples fs
                LEFT JOIN detector_reviews dr ON dr.sample_id = fs.sample_id
                WHERE fs.person_id = ?
                ORDER BY fs.quality_score DESC, fs.sample_id DESC
                """,
                (person_id,),
            ).fetchall()

        if person is None:
            return "<h1>Not found</h1>"

        display_name_raw = str(person["display_name"])
        display_name = html.escape(display_name_raw)
        person_id_esc = html.escape(person_id)

        items = []
        for row in samples:
            sample_id = int(row["sample_id"])
            path = str(row["source_image"] or "")
            score = float(row["quality_score"] or 0.0)
            created_at = html.escape(str(row["created_at"] or ""))
            src = f"/img?path={quote(path)}"
            meta = f"score={score:.2f} · {created_at}"
            detector_label = str(row["detector_label"] or "")
            items.append(
                f"""
                <button class="thumb" data-src="{src}" data-path="{html.escape(path)}" data-meta="{html.escape(meta)}" data-sample-id="{sample_id}" data-detector-label="{html.escape(detector_label)}">
                  <img src="{src}" alt="sample" loading="lazy" />
                </button>
                """
            )

        first_src = "/"
        first_path = ""
        first_meta = ""
        if samples:
            p0 = str(samples[0]["source_image"] or "")
            first_src = f"/img?path={quote(p0)}"
            first_path = html.escape(p0)
            first_meta = html.escape(
                f"score={float(samples[0]['quality_score'] or 0.0):.2f} · {str(samples[0]['created_at'] or '')}"
            )

        return f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>{display_name}</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #111; }}
            a {{ color: #2563eb; text-decoration: none; }}
            .rename {{ display: flex; gap: 8px; margin: 8px 0 16px 0; }}
            .rename input {{ flex: 1; max-width: 380px; padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; }}
            .rename button {{ padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; cursor: pointer; }}
            .person-actions {{ margin: 0 0 16px 0; }}
            .person-actions button {{ padding: 8px 12px; border: 1px solid #ef4444; border-radius: 8px; color: #b91c1c; background: #fff; cursor: pointer; }}
            .viewer {{ background: white; border-radius: 12px; border: 1px solid #e5e7eb; padding: 12px; }}
            .viewer img {{ width: 100%; max-height: 72vh; object-fit: contain; display: block; background: #eee; border-radius: 8px; }}
            .meta {{ margin-top: 8px; font-size: 13px; color: #555; }}
            .path {{ word-break: break-all; margin-top: 4px; }}
            .thumbs {{ margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }}
            .thumb {{ border: 2px solid transparent; border-radius: 8px; background: white; padding: 0; cursor: pointer; overflow: hidden; }}
            .thumb img {{ width: 100%; height: 100px; object-fit: cover; display: block; background: #eee; }}
            .thumb.active {{ border-color: #2563eb; }}
            .moderation {{ margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
            .moderation input {{ padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; min-width: 220px; }}
            .moderation button {{ padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; cursor: pointer; }}
            .moderation .danger {{ border-color: #ef4444; color: #b91c1c; }}
            .selected-id {{ font-size: 13px; color: #555; }}
          </style>
        </head>
        <body>
          <p><a href="/">← Home</a> · <a href="/people">People Gallery</a></p>
          <h1>{display_name}</h1>
          <p><code>{person_id_esc}</code> · {len(samples)} samples</p>
          <form class="rename" method="post" action="/person/{quote(person_id)}/rename">
            <input type="text" name="display_name" value="{html.escape(display_name_raw)}" placeholder="Display name" />
            <button type="submit">Save Name</button>
          </form>
          <form class="person-actions" method="post" action="/person/{quote(person_id)}/delete" onsubmit="return confirm('Delete this person and all associated samples?');">
            <button type="submit">Delete Person</button>
          </form>
          {"<p>No samples.</p>" if not items else f'''
          <section class="viewer">
            <a id="viewer-link" href="{first_src}" target="_blank" rel="noopener noreferrer">
              <img id="viewer-image" src="{first_src}" alt="Selected full-size snapshot" />
            </a>
            <div class="meta" id="viewer-meta">{first_meta}</div>
            <div class="path" id="viewer-path">{first_path}</div>
            <div class="moderation">
              <div class="selected-id">Sample ID: <span id="selected-sample-id"></span></div>
              <form method="post" action="/sample/move">
                <input type="hidden" name="sample_id" id="move-sample-id" value="" />
                <input type="hidden" name="return_person" value="{person_id_esc}" />
                <input type="text" name="target_person_id" placeholder="target person_id or name" required />
                <button type="submit">Move Sample</button>
              </form>
              <form method="post" action="/sample/delete" onsubmit="return confirm('Delete this sample from DB?');">
                <input type="hidden" name="sample_id" id="delete-sample-id" value="" />
                <input type="hidden" name="return_person" value="{person_id_esc}" />
                <button class="danger" type="submit">Delete Sample</button>
              </form>
            </div>
          </section>
          <section class="thumbs" id="thumbs">{''.join(items)}</section>
          <script>
            (() => {{
              const thumbs = Array.from(document.querySelectorAll('.thumb'));
              const image = document.getElementById('viewer-image');
              const link = document.getElementById('viewer-link');
              const meta = document.getElementById('viewer-meta');
              const path = document.getElementById('viewer-path');
              const selectedSample = document.getElementById('selected-sample-id');
              const moveSample = document.getElementById('move-sample-id');
              const deleteSample = document.getElementById('delete-sample-id');
              let idx = 0;

              function activate(i) {{
                if (i < 0 || i >= thumbs.length) return;
                idx = i;
                thumbs.forEach((el, j) => el.classList.toggle('active', j === idx));
                const selected = thumbs[idx];
                const src = selected.dataset.src || '';
                image.src = src;
                link.href = src;
                meta.textContent = selected.dataset.meta || '';
                path.textContent = selected.dataset.path || '';
                const sampleId = selected.dataset.sampleId || '';
                selectedSample.textContent = sampleId;
                moveSample.value = sampleId;
                deleteSample.value = sampleId;
              }}

              thumbs.forEach((btn, i) => btn.addEventListener('click', () => activate(i)));
              document.addEventListener('keydown', (ev) => {{
                if (ev.key === 'ArrowRight') activate(Math.min(thumbs.length - 1, idx + 1));
                if (ev.key === 'ArrowLeft') activate(Math.max(0, idx - 1));
              }});
              activate(0);
            }})();
          </script>
          '''}
        </body>
        </html>
        """

    def _assign_snapshot_person(self, conn: sqlite3.Connection, source_image: str, target_person_id: str) -> None:
        """Attempt to extract and store a face sample from assigned snapshot."""
        image_path = Path(source_image)
        if not image_path.exists():
            return

        frame = cv2.imread(str(image_path))
        if frame is None:
            return

        engine = FaceEmbeddingEngine()
        embedding, quality = engine.best_face_embedding(frame)
        if embedding is None:
            return

        exists = conn.execute(
            "SELECT 1 FROM face_samples WHERE source_image = ? AND person_id = ? LIMIT 1",
            (source_image, target_person_id),
        ).fetchone()
        if exists is not None:
            return

        emb = embedding.astype("float32")
        conn.execute(
            """
            INSERT INTO face_samples (
                person_id, source_image, camera_id, captured_at, quality_score,
                embedding, embedding_dim, face_box
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target_person_id,
                source_image,
                None,
                None,
                float(quality),
                sqlite3.Binary(emb.tobytes()),
                int(emb.shape[0]),
                "",
            ),
        )

    def do_GET(self) -> None:
        """Handle HTTP GET routes."""
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_html(self._render_root())
            return

        if parsed.path == "/people":
            self._send_html(self._render_people())
            return

        if parsed.path == "/guide":
            self._send_html(self._render_guide())
            return

        if parsed.path == "/live":
            self._send_html(self._render_live())
            return

        if parsed.path == "/live/stream":
            query = parse_qs(parsed.query)
            channel_raw = (query.get("channel", [""])[0] or "").strip()
            if not channel_raw.isdigit():
                self.send_error(400, "Missing or invalid channel")
                return
            self._send_mjpeg_stream(int(channel_raw))
            return

        if parsed.path == "/live/frame":
            query = parse_qs(parsed.query)
            channel_raw = (query.get("channel", [""])[0] or "").strip()
            if not channel_raw.isdigit():
                self.send_error(400, "Missing or invalid channel")
                return
            self._send_live_frame(int(channel_raw))
            return

        if parsed.path == "/snapshots":
            self._send_html(self._render_snapshots())
            return

        if parsed.path.startswith("/person/"):
            person_id = unquote(parsed.path.split("/person/", 1)[1])
            self._send_html(self._render_person(person_id))
            return

        if parsed.path == "/img":
            query = parse_qs(parsed.query)
            path_param = query.get("path", [""])[0]
            image_path = self._resolve_image_path(path_param)
            if image_path is None:
                self.send_error(404, "Image not found")
                return
            self._send_binary(image_path.read_bytes(), self._guess_type(image_path))
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        """Handle HTTP POST routes for reviews and dataset moderation."""
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body)

        if parsed.path.startswith("/person/") and parsed.path.endswith("/rename"):
            parts = parsed.path.split("/")
            if len(parts) < 4:
                self.send_error(400, "Bad request")
                return
            person_id = unquote(parts[2])
            display_name = (form.get("display_name", [""])[0] or "").strip() or person_id
            with self._connect() as conn:
                conn.execute("UPDATE persons SET display_name = ? WHERE person_id = ?", (display_name, person_id))
                conn.commit()
            self._redirect(f"/person/{quote(person_id)}")
            return

        if parsed.path.startswith("/person/") and parsed.path.endswith("/delete"):
            parts = parsed.path.split("/")
            if len(parts) < 4:
                self.send_error(400, "Bad request")
                return
            person_id = unquote(parts[2])
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM detector_reviews WHERE sample_id IN (SELECT sample_id FROM face_samples WHERE person_id = ?)",
                    (person_id,),
                )
                conn.execute("DELETE FROM face_samples WHERE person_id = ?", (person_id,))
                conn.execute("UPDATE snapshot_reviews SET person_id = NULL WHERE person_id = ?", (person_id,))
                conn.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))
                conn.commit()
            self._redirect("/people")
            return

        if parsed.path == "/sample/move":
            sample_id_raw = (form.get("sample_id", [""])[0] or "").strip()
            target_identifier = (form.get("target_person_id", [""])[0] or "").strip()
            return_person = (form.get("return_person", [""])[0] or "").strip()
            if not sample_id_raw.isdigit() or not target_identifier:
                self.send_error(400, "Invalid sample move request")
                return
            sample_id = int(sample_id_raw)

            with self._connect() as conn:
                target_person_id = self._resolve_person_id(conn, target_identifier)
                if target_person_id is None:
                    target_person_id = self._create_person_from_input(conn, target_identifier)
                conn.execute("UPDATE face_samples SET person_id = ? WHERE sample_id = ?", (target_person_id, sample_id))
                conn.commit()

            self._redirect(f"/person/{quote(return_person or target_person_id)}")
            return

        if parsed.path == "/sample/delete":
            sample_id_raw = (form.get("sample_id", [""])[0] or "").strip()
            return_person = (form.get("return_person", [""])[0] or "").strip()
            if not sample_id_raw.isdigit():
                self.send_error(400, "Invalid sample delete request")
                return
            sample_id = int(sample_id_raw)

            with self._connect() as conn:
                conn.execute("DELETE FROM detector_reviews WHERE sample_id = ?", (sample_id,))
                conn.execute("DELETE FROM face_samples WHERE sample_id = ?", (sample_id,))
                conn.commit()

            self._redirect(f"/person/{quote(return_person)}")
            return

        if parsed.path == "/snapshot/review-detector":
            source_image = (form.get("source_image", [""])[0] or "").strip()
            label = (form.get("label", [""])[0] or "").strip()
            if not source_image:
                self.send_error(400, "Missing source_image")
                return

            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT person_id FROM snapshot_reviews WHERE source_image = ?",
                    (source_image,),
                ).fetchone()
                existing_person = existing["person_id"] if existing is not None else None

                if label == "clear":
                    if existing_person:
                        conn.execute(
                            "UPDATE snapshot_reviews SET detector_label = NULL, reviewed_at = CURRENT_TIMESTAMP WHERE source_image = ?",
                            (source_image,),
                        )
                    else:
                        conn.execute("DELETE FROM snapshot_reviews WHERE source_image = ?", (source_image,))
                elif label in {"person", "no_person"}:
                    conn.execute(
                        """
                        INSERT INTO snapshot_reviews (source_image, detector_label, person_id, reviewed_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(source_image) DO UPDATE SET
                            detector_label = excluded.detector_label,
                            reviewed_at = CURRENT_TIMESTAMP
                        """,
                        (source_image, label, existing_person),
                    )
                else:
                    self.send_error(400, "Invalid detector label")
                    return
                conn.commit()

            self._redirect("/snapshots")
            return

        if parsed.path == "/snapshot/review-detector-bulk":
            selected_sources_raw = (form.get("selected_sources", [""])[0] or "").strip()
            label = (form.get("label", [""])[0] or "").strip()
            selected_sources = [s.strip() for s in selected_sources_raw.splitlines() if s.strip()]
            if not selected_sources:
                self._redirect("/snapshots")
                return

            if label not in {"person", "no_person", "clear"}:
                self.send_error(400, "Invalid detector label")
                return

            with self._connect() as conn:
                for source_image in selected_sources:
                    existing = conn.execute(
                        "SELECT person_id FROM snapshot_reviews WHERE source_image = ?",
                        (source_image,),
                    ).fetchone()
                    existing_person = existing["person_id"] if existing is not None else None

                    if label == "clear":
                        if existing_person:
                            conn.execute(
                                "UPDATE snapshot_reviews SET detector_label = NULL, reviewed_at = CURRENT_TIMESTAMP WHERE source_image = ?",
                                (source_image,),
                            )
                        else:
                            conn.execute("DELETE FROM snapshot_reviews WHERE source_image = ?", (source_image,))
                    else:
                        conn.execute(
                            """
                            INSERT INTO snapshot_reviews (source_image, detector_label, person_id, reviewed_at)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                            ON CONFLICT(source_image) DO UPDATE SET
                                detector_label = excluded.detector_label,
                                reviewed_at = CURRENT_TIMESTAMP
                            """,
                            (source_image, label, existing_person),
                        )
                conn.commit()

            self._redirect("/snapshots")
            return

        if parsed.path == "/snapshot/assign-person":
            source_image = (form.get("source_image", [""])[0] or "").strip()
            target_raw = (form.get("target_person", [""])[0] or "").strip()
            if not source_image:
                self.send_error(400, "Missing source_image")
                return

            with self._connect() as conn:
                if not target_raw:
                    conn.execute(
                        """
                        INSERT INTO snapshot_reviews (source_image, detector_label, person_id, reviewed_at)
                        VALUES (?, NULL, NULL, CURRENT_TIMESTAMP)
                        ON CONFLICT(source_image) DO UPDATE SET
                            person_id = NULL,
                            reviewed_at = CURRENT_TIMESTAMP
                        """,
                        (source_image,),
                    )
                    conn.commit()
                    self._redirect("/snapshots")
                    return

                person_id = self._resolve_person_id(conn, target_raw)
                if person_id is None:
                    person_id = self._create_person_from_input(conn, target_raw)

                conn.execute(
                    """
                    INSERT INTO snapshot_reviews (source_image, detector_label, person_id, reviewed_at)
                    VALUES (?, 'person', ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(source_image) DO UPDATE SET
                        person_id = excluded.person_id,
                        detector_label = COALESCE(snapshot_reviews.detector_label, 'person'),
                        reviewed_at = CURRENT_TIMESTAMP
                    """,
                    (source_image, person_id),
                )
                self._assign_snapshot_person(conn, source_image, person_id)
                conn.commit()

            self._redirect("/snapshots")
            return

        self.send_error(404, "Not found")


def create_gallery_server(
    *,
    host: str,
    port: int,
    db_path: Path,
    snapshot_dir: Path,
    camera_map: dict[int, CameraConfig],
    settings: Settings | None,
    live_frame_provider: Callable[[int], Optional[tuple[int, bytes]]],
) -> ThreadingHTTPServer:
    """Create configured gallery HTTP server instance."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    FaceGalleryHandler.db_path = db_path
    FaceGalleryHandler.snapshot_dir = snapshot_dir
    FaceGalleryHandler.camera_map = camera_map
    FaceGalleryHandler.settings = settings
    FaceGalleryHandler.live_frame_provider = live_frame_provider
    return ThreadingHTTPServer((host, port), FaceGalleryHandler)


def start_gallery_server(
    *,
    host: str,
    port: int,
    db_path: Path,
    snapshot_dir: Path,
    camera_map: dict[int, CameraConfig],
    settings: Settings | None,
    live_frame_provider: Callable[[int], Optional[tuple[int, bytes]]],
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Create and start gallery server in a background thread."""
    server = create_gallery_server(
        host=host,
        port=port,
        db_path=db_path,
        snapshot_dir=snapshot_dir,
        camera_map=camera_map,
        settings=settings,
        live_frame_provider=live_frame_provider,
    )
    thread = threading.Thread(target=server.serve_forever, name="face-gallery", daemon=True)
    thread.start()
    return server, thread

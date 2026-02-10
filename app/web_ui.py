from __future__ import annotations

"""Local web UI for reviewing snapshots and managing face/person datasets."""

import html
import cgi
import re
import shutil
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
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
    model_importer: Optional[Callable[[Path, bool], tuple[bool, str]]] = None
    model_status_provider: Optional[Callable[[], dict[str, object]]] = None
    stats_provider: Optional[Callable[[], dict[str, object]]] = None
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

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send JSON response."""
        import json
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _render_nav(self, active: str = "") -> str:
        """Render unified navigation bar HTML."""
        def nav_item(href: str, label: str, key: str, icon: str) -> str:
            active_class = " active" if active == key else ""
            return f'<a class="nav-item{active_class}" href="{href}">{icon}<span>{label}</span></a>'

        return f"""
        <nav class="main-nav">
          <div class="nav-brand">
            <svg class="nav-logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            <span class="nav-title">DVR Console</span>
          </div>
          <div class="nav-links">
            {nav_item("/", "Dashboard", "home", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>')}
            {nav_item("/live", "Live", "live", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>')}
            {nav_item("/snapshots", "Snapshots", "snapshots", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>')}
            {nav_item("/people", "People", "people", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>')}
            {nav_item("/models", "Models", "models", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>')}
            {nav_item("/guide", "Guide", "guide", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>')}
          </div>
        </nav>
        """

    def _shared_styles(self) -> str:
        """Return shared CSS styles for all pages."""
        return """
        <style>
          :root {
            --bg-primary: #f3f7fe;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f8fafc;
            --ink-primary: #0f172a;
            --ink-secondary: #475569;
            --ink-muted: #64748b;
            --accent: #2563eb;
            --accent-light: #3b82f6;
            --accent-bg: rgba(37, 99, 235, 0.08);
            --stroke: rgba(148, 163, 184, 0.35);
            --stroke-strong: #cbd5e1;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --card-bg: rgba(255, 255, 255, 0.92);
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --nav-height: 60px;
          }
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(ellipse at top left, #dbeafe 0%, transparent 50%),
                        radial-gradient(ellipse at bottom right, #fef3c7 0%, transparent 50%),
                        var(--bg-primary);
            color: var(--ink-primary);
            min-height: 100vh;
            padding-top: var(--nav-height);
          }
          .main-nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: var(--nav-height);
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--stroke);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            z-index: 1000;
          }
          .nav-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--ink-primary);
          }
          .nav-logo {
            width: 28px;
            height: 28px;
            color: var(--accent);
          }
          .nav-links {
            display: flex;
            gap: 4px;
          }
          .nav-item {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            border-radius: 8px;
            text-decoration: none;
            color: var(--ink-secondary);
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 150ms ease;
          }
          .nav-item:hover {
            background: var(--accent-bg);
            color: var(--accent);
          }
          .nav-item.active {
            background: var(--accent);
            color: white;
          }
          .nav-item svg {
            width: 18px;
            height: 18px;
          }
          .shell {
            max-width: 1280px;
            margin: 0 auto;
            padding: 24px;
          }
          .page-header {
            margin-bottom: 24px;
          }
          .page-header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 4px;
          }
          .page-header p {
            color: var(--ink-muted);
            font-size: 0.95rem;
          }
          .card {
            background: var(--card-bg);
            border: 1px solid var(--stroke);
            border-radius: 12px;
            padding: 20px;
            box-shadow: var(--card-shadow);
          }
          .btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            border: 1px solid var(--stroke-strong);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--ink-primary);
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 150ms ease;
          }
          .btn:hover {
            border-color: var(--accent);
            color: var(--accent);
          }
          .btn-primary {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
          }
          .btn-primary:hover {
            background: var(--accent-light);
            border-color: var(--accent-light);
            color: white;
          }
          .btn-danger {
            border-color: var(--error);
            color: var(--error);
          }
          .btn-danger:hover {
            background: var(--error);
            color: white;
          }
          @media (max-width: 768px) {
            .nav-item span { display: none; }
            .nav-item { padding: 8px 10px; }
            .shell { padding: 16px; }
          }
        </style>
        """

    def _send_health(self) -> None:
        """Return health check response with system status."""
        # Count snapshots
        snapshot_count = len(self._all_snapshots())
        
        # Get camera status from live frame provider
        cameras_ok = 0
        cameras_total = len(self.camera_map)
        for cam_id in self.camera_map:
            if self.live_frame_provider:
                frame = self.live_frame_provider(cam_id)
                if frame is not None:
                    cameras_ok += 1
        
        # Get model status (mask absolute paths)
        model_status = {}
        if self.model_status_provider:
            raw_status = self.model_status_provider()
            for key, value in raw_status.items():
                if isinstance(value, str) and "/" in value:
                    # Only show filename, not full path
                    model_status[key] = Path(value).name
                else:
                    model_status[key] = value
        
        health = {
            "status": "ok" if cameras_ok > 0 else "degraded",
            "cameras": {
                "total": cameras_total,
                "active": cameras_ok,
            },
            "snapshots": snapshot_count,
            "model": model_status,
        }
        self._send_json(health)

    def _send_metrics(self) -> None:
        """Return Prometheus-format metrics."""
        lines = []
        lines.append("# HELP dvr_cameras_total Total number of configured cameras")
        lines.append("# TYPE dvr_cameras_total gauge")
        lines.append(f"dvr_cameras_total {len(self.camera_map)}")
        
        lines.append("# HELP dvr_snapshots_total Total snapshots on disk")
        lines.append("# TYPE dvr_snapshots_total gauge")
        lines.append(f"dvr_snapshots_total {len(self._all_snapshots())}")
        
        if self.stats_provider:
            stats = self.stats_provider()
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    metric_name = f"dvr_{key}"
                    lines.append(f"# TYPE {metric_name} counter")
                    lines.append(f"{metric_name} {value}")
        
        body = "\n".join(lines) + "\n"
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
          <title>DVR Console - Dashboard</title>
          {self._shared_styles()}
          <style>
            .hero {{
              display: grid;
              grid-template-columns: 1.3fr 1fr;
              gap: 16px;
              margin-bottom: 24px;
            }}
            .hero-text h1 {{ font-size: clamp(1.5rem, 2.8vw, 2rem); margin-bottom: 8px; }}
            .hero-text p {{ color: var(--ink-muted); }}
            .stats-grid {{
              display: grid;
              grid-template-columns: repeat(2, 1fr);
              gap: 12px;
            }}
            .stat-card {{
              background: var(--accent-bg);
              border: 1px solid rgba(37, 99, 235, 0.2);
              border-radius: 10px;
              padding: 14px;
              text-align: center;
            }}
            .stat-value {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
            .stat-label {{ font-size: 0.8rem; color: var(--ink-muted); margin-top: 2px; }}
            .quick-links {{
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
              gap: 16px;
            }}
            .link-card {{
              display: flex;
              align-items: flex-start;
              gap: 14px;
              padding: 20px;
              text-decoration: none;
              color: inherit;
              transition: transform 150ms ease, box-shadow 150ms ease;
            }}
            .link-card:hover {{
              transform: translateY(-2px);
              box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
            }}
            .link-icon {{
              width: 44px;
              height: 44px;
              border-radius: 10px;
              background: var(--accent-bg);
              display: flex;
              align-items: center;
              justify-content: center;
              flex-shrink: 0;
            }}
            .link-icon svg {{ width: 22px; height: 22px; color: var(--accent); }}
            .link-content h3 {{ font-size: 1rem; font-weight: 600; margin-bottom: 4px; }}
            .link-content p {{ font-size: 0.875rem; color: var(--ink-muted); line-height: 1.4; }}
            .link-tag {{
              display: inline-block;
              margin-top: 8px;
              padding: 3px 10px;
              background: var(--bg-tertiary);
              border-radius: 999px;
              font-size: 0.7rem;
              font-weight: 600;
              color: var(--ink-secondary);
              text-transform: uppercase;
              letter-spacing: 0.03em;
            }}
            @media (max-width: 900px) {{
              .hero {{ grid-template-columns: 1fr; }}
            }}
          </style>
        </head>
        <body>
          {self._render_nav("home")}
          <div class="shell">
            <section class="hero">
              <div class="card hero-text">
                <h1>Surveillance Dashboard</h1>
                <p>Monitor live feeds, review detections, manage identities, and control model lifecycle.</p>
              </div>
              <div class="card">
                <div class="stats-grid">
                  <div class="stat-card">
                    <div class="stat-value">{snapshot_count}</div>
                    <div class="stat-label">Snapshots</div>
                  </div>
                  <div class="stat-card">
                    <div class="stat-value">{reviewed_snapshots}</div>
                    <div class="stat-label">Reviewed</div>
                  </div>
                  <div class="stat-card">
                    <div class="stat-value">{person_count}</div>
                    <div class="stat-label">People</div>
                  </div>
                  <div class="stat-card">
                    <div class="stat-value">{len(self.camera_map)}</div>
                    <div class="stat-label">Cameras</div>
                  </div>
                </div>
              </div>
            </section>
            <div class="quick-links">
              <a class="card link-card" href="/live">
                <div class="link-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="23 7 16 12 23 17 23 7"/>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                  </svg>
                </div>
                <div class="link-content">
                  <h3>Live Feed</h3>
                  <p>{len(self.camera_map)} camera streams with real-time monitoring</p>
                  <span class="link-tag">Operations</span>
                </div>
              </a>
              <a class="card link-card" href="/snapshots">
                <div class="link-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                </div>
                <div class="link-content">
                  <h3>Snapshot Review</h3>
                  <p>{snapshot_count} snapshots · {reviewed_snapshots} reviewed</p>
                  <span class="link-tag">Feedback Loop</span>
                </div>
              </a>
              <a class="card link-card" href="/people">
                <div class="link-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                    <circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                  </svg>
                </div>
                <div class="link-content">
                  <h3>People Gallery</h3>
                  <p>{person_count} people · {face_sample_count} face samples</p>
                  <span class="link-tag">Identity Curation</span>
                </div>
              </a>
              <a class="card link-card" href="/models">
                <div class="link-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                    <polyline points="2 17 12 22 22 17"/>
                    <polyline points="2 12 12 17 22 12"/>
                  </svg>
                </div>
                <div class="link-content">
                  <h3>Model Management</h3>
                  <p>Import YOLO models and manage active/base model</p>
                  <span class="link-tag">Model Ops</span>
                </div>
              </a>
              <a class="card link-card" href="/guide">
                <div class="link-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                  </svg>
                </div>
                <div class="link-content">
                  <h3>User Guide</h3>
                  <p>System setup, parameters, and review workflow documentation</p>
                  <span class="link-tag">Docs</span>
                </div>
              </a>
            </div>
          </div>
        </body>
        </html>
        """

    def _render_models(self, message: str = "", ok: bool = True) -> str:
        """Render model import/management page."""
        status_provider = self.model_status_provider
        status: dict[str, object] = {}
        if status_provider is not None:
            try:
                status = status_provider() or {}
            except Exception:
                status = {}

        active_model = html.escape(str(status.get("active_model", "n/a")))
        base_model = html.escape(str(status.get("base_model", "n/a")))
        loaded_model = html.escape(str(status.get("loaded_model", "n/a")))
        generation = html.escape(str(status.get("generation", "n/a")))
        update_running = "yes" if bool(status.get("update_running", False)) else "no"
        update_running = html.escape(update_running)
        message_html = (
            f'<p class="msg {"ok" if ok else "err"}">{html.escape(message)}</p>' if message else ""
        )

        return f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>DVR Console - Models</title>
          {self._shared_styles()}
          <style>
            .status-grid {{
              display: grid;
              grid-template-columns: 180px 1fr;
              gap: 12px 16px;
              align-items: center;
            }}
            .status-grid dt {{ font-weight: 600; color: var(--ink-secondary); }}
            .status-grid dd {{ margin: 0; }}
            .status-grid code {{
              background: var(--bg-tertiary);
              padding: 4px 8px;
              border-radius: 6px;
              font-size: 0.875rem;
              word-break: break-all;
            }}
            .section {{ margin-bottom: 20px; }}
            .section h2 {{ font-size: 1.1rem; margin-bottom: 12px; }}
            .section p {{ color: var(--ink-muted); font-size: 0.9rem; margin-bottom: 12px; }}
            .upload-form {{ display: flex; flex-direction: column; gap: 12px; }}
            .upload-form input[type="file"] {{
              padding: 12px;
              border: 2px dashed var(--stroke-strong);
              border-radius: 10px;
              background: var(--bg-tertiary);
              cursor: pointer;
            }}
            .upload-form input[type="file"]:hover {{ border-color: var(--accent); }}
            .upload-form label {{
              display: flex;
              align-items: center;
              gap: 8px;
              font-size: 0.9rem;
              color: var(--ink-secondary);
            }}
            .msg {{
              border-radius: 8px;
              padding: 12px 16px;
              margin-top: 16px;
              font-size: 0.9rem;
            }}
            .msg.ok {{
              background: rgba(16, 185, 129, 0.1);
              border: 1px solid var(--success);
              color: #065f46;
            }}
            .msg.err {{
              background: rgba(239, 68, 68, 0.1);
              border: 1px solid var(--error);
              color: #991b1b;
            }}
          </style>
        </head>
        <body>
          {self._render_nav("models")}
          <div class="shell">
            <div class="page-header">
              <h1>Model Management</h1>
              <p>Import and manage YOLO detection models</p>
            </div>
            <div class="card section">
              <h2>Current Status</h2>
              <dl class="status-grid">
                <dt>Active model</dt><dd><code>{active_model}</code></dd>
                <dt>Loaded by workers</dt><dd><code>{loaded_model}</code></dd>
                <dt>Base model</dt><dd><code>{base_model}</code></dd>
                <dt>Generation</dt><dd><code>{generation}</code></dd>
                <dt>Update running</dt><dd><code>{update_running}</code></dd>
              </dl>
            </div>
            <div class="card section">
              <h2>Import New Model</h2>
              <p>Upload a <code>.pt</code> file to replace the active model immediately. Optionally also replace the base model used by <code>/reload_model</code>.</p>
              <form class="upload-form" method="post" action="/models/import" enctype="multipart/form-data">
                <input type="file" name="model_file" accept=".pt" required />
                <label>
                  <input type="checkbox" name="replace_base" value="1" />
                  Also replace base model
                </label>
                <button class="btn btn-primary" type="submit">Import Model</button>
              </form>
              {message_html}
            </div>
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
          <title>DVR Console - Live</title>
          {self._shared_styles()}
          <style>
            .viewer {{
              background: var(--card-bg);
              border: 1px solid var(--stroke);
              border-radius: 12px;
              padding: 16px;
              margin-bottom: 16px;
              box-shadow: var(--card-shadow);
            }}
            .viewer img {{
              width: 100%;
              max-height: 65vh;
              object-fit: contain;
              background: #0a0a0a;
              border-radius: 8px;
            }}
            .viewer-top {{
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 12px;
            }}
            .viewer-title {{
              font-weight: 700;
              font-size: 1.1rem;
            }}
            .viewer-help {{
              color: var(--ink-muted);
              font-size: 0.85rem;
              display: flex;
              align-items: center;
              gap: 6px;
            }}
            .viewer-help kbd {{
              background: var(--bg-tertiary);
              border: 1px solid var(--stroke);
              border-radius: 4px;
              padding: 2px 6px;
              font-size: 0.75rem;
              font-family: inherit;
            }}
            .thumbs {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
              gap: 12px;
            }}
            .thumb {{
              border: 2px solid transparent;
              background: var(--card-bg);
              border-radius: 10px;
              padding: 8px;
              cursor: pointer;
              text-align: left;
              transition: all 150ms ease;
              box-shadow: var(--card-shadow);
            }}
            .thumb:hover {{
              transform: translateY(-2px);
              box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            }}
            .thumb.active {{
              border-color: var(--accent);
              box-shadow: 0 0 0 3px var(--accent-bg);
            }}
            .thumb img {{
              width: 100%;
              height: 120px;
              object-fit: cover;
              border-radius: 6px;
              background: #1a1a1a;
              display: block;
            }}
            .thumb-label {{
              display: block;
              margin-top: 8px;
              font-size: 0.85rem;
              font-weight: 600;
              color: var(--ink-primary);
            }}
          </style>
        </head>
        <body>
          {self._render_nav("live")}
          <div class="shell">
            <div class="page-header">
              <h1>Live Feed</h1>
              <p>Real-time camera monitoring with keyboard navigation</p>
            </div>
          {'' if camera_items else '<p class="card">No cameras configured.</p>'}
          <section class="viewer" id="viewer" style="display:{'block' if camera_items else 'none'};">
            <div class="viewer-top">
              <div class="viewer-title" id="viewer-title"></div>
              <div class="viewer-help"><kbd>←</kbd> <kbd>→</kbd> to switch cameras</div>
            </div>
            <img id="viewer-image" src="" alt="live focused camera" loading="eager" />
          </section>
          <div class="thumbs">{''.join(tiles)}</div>
          </div>
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
        return f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>DVR Console - Guide</title>
          {self._shared_styles()}
          <style>
            .guide-content h2 {{ margin: 28px 0 12px; font-size: 1.15rem; }}
            .guide-content h2:first-child {{ margin-top: 0; }}
            .guide-content p {{ color: var(--ink-secondary); line-height: 1.6; margin: 8px 0; }}
            .guide-content ul {{ margin: 8px 0 16px 24px; color: var(--ink-secondary); }}
            .guide-content li {{ margin: 6px 0; line-height: 1.5; }}
            .guide-content code {{ background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px; font-size: 0.875em; }}
            .guide-content pre {{
              background: #1e293b;
              color: #e2e8f0;
              padding: 16px;
              border-radius: 8px;
              overflow-x: auto;
              font-size: 0.875rem;
              margin: 12px 0;
            }}
            .guide-content .card {{ margin: 12px 0 20px; }}
            .guide-content table {{
              width: 100%;
              border-collapse: collapse;
              background: var(--bg-secondary);
              border: 1px solid var(--stroke);
              border-radius: 8px;
              overflow: hidden;
              font-size: 0.875rem;
              margin: 12px 0;
            }}
            .guide-content th, .guide-content td {{
              text-align: left;
              padding: 10px 12px;
              border-bottom: 1px solid var(--stroke);
              vertical-align: top;
            }}
            .guide-content th {{ background: var(--bg-tertiary); font-weight: 600; }}
            .guide-content tr:last-child td {{ border-bottom: none; }}
            a {{ color: var(--accent); text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
          </style>
        </head>
        <body>
          {self._render_nav("guide")}
          <div class="shell">
            <div class="page-header">
              <h1>User Guide</h1>
              <p>System setup, parameters, and review workflow documentation</p>
            </div>
            <div class="card guide-content">
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

          <h2>8. Model Management (Web UI)</h2>
          <div class="card">
            <ul>
              <li>Open <a href="/models">Model Management</a>.</li>
              <li>Upload a YOLO <code>.pt</code> model to make it active immediately.</li>
              <li>Enable "replace base model" if you also want <code>/reload_model</code> to train from this imported model.</li>
              <li>The imported model is archived under <code>detection_models/yolov8n_imported_&lt;timestamp&gt;.pt</code>.</li>
            </ul>
          </div>
            </div>
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
                <a class="person-card" href="{link}">
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
          <title>DVR Console - People</title>
          {self._shared_styles()}
          <style>
            .people-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
              gap: 16px;
            }}
            .person-card {{
              text-decoration: none;
              color: inherit;
              background: var(--card-bg);
              border-radius: 12px;
              overflow: hidden;
              border: 1px solid var(--stroke);
              box-shadow: var(--card-shadow);
              transition: transform 150ms ease, box-shadow 150ms ease;
            }}
            .person-card:hover {{
              transform: translateY(-3px);
              box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
            }}
            .person-card .thumb {{
              width: 100%;
              height: 180px;
              object-fit: cover;
              display: block;
              background: var(--bg-tertiary);
            }}
            .person-card .thumb.empty {{
              display: flex;
              align-items: center;
              justify-content: center;
              color: var(--ink-muted);
              font-size: 0.875rem;
            }}
            .person-card .meta {{
              padding: 14px;
            }}
            .person-card .title {{
              font-weight: 600;
              font-size: 1rem;
            }}
            .person-card .sub {{
              margin-top: 4px;
              color: var(--ink-muted);
              font-size: 0.8rem;
            }}
          </style>
        </head>
        <body>
          {self._render_nav("people")}
          <div class="shell">
            <div class="page-header">
              <h1>People Gallery</h1>
              <p>Manage identities and face samples</p>
            </div>
            <div class="people-grid">{''.join(cards) if cards else '<p class="card">No persons found.</p>'}</div>
          </div>
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
          <title>DVR Console - Snapshots</title>
          {self._shared_styles()}
          <style>
            .viewer {{
              position: sticky;
              top: calc(var(--nav-height) + 16px);
              background: var(--card-bg);
              border-radius: 12px;
              border: 1px solid var(--stroke);
              padding: 16px;
              box-shadow: var(--card-shadow);
              margin-bottom: 20px;
            }}
            .viewer img {{
              width: 100%;
              max-height: 65vh;
              object-fit: contain;
              display: block;
              background: var(--bg-tertiary);
              border-radius: 8px;
            }}
            .viewer-meta {{
              margin-top: 10px;
              font-size: 0.8rem;
              color: var(--ink-muted);
              word-break: break-all;
            }}
            .viewer-state {{
              font-size: 0.85rem;
              color: var(--ink-secondary);
              margin-top: 6px;
            }}
            .viewer-state strong {{ color: var(--ink-primary); }}
            .actions {{
              margin-top: 12px;
              display: flex;
              gap: 8px;
              flex-wrap: wrap;
              align-items: center;
            }}
            .actions input {{
              padding: 8px 12px;
              border: 1px solid var(--stroke-strong);
              border-radius: 8px;
              min-width: 200px;
              font-size: 0.875rem;
            }}
            .sections {{ display: grid; gap: 20px; }}
            .section {{
              background: var(--card-bg);
              border: 1px solid var(--stroke);
              border-radius: 12px;
              padding: 16px;
              box-shadow: var(--card-shadow);
            }}
            .section h2 {{ margin: 0 0 4px; font-size: 1rem; }}
            .section .sub {{ font-size: 0.8rem; color: var(--ink-muted); margin-bottom: 12px; }}
            .thumbs {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
              gap: 8px;
            }}
            .thumb {{
              border: 2px solid var(--stroke-strong);
              border-radius: 8px;
              background: var(--bg-secondary);
              padding: 0;
              cursor: pointer;
              overflow: hidden;
              transition: all 120ms ease;
            }}
            .thumb:hover {{
              transform: translateY(-2px);
              box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            }}
            .thumb img {{
              width: 100%;
              height: 90px;
              object-fit: cover;
              display: block;
              background: var(--bg-tertiary);
            }}
            .thumb.active {{ border-color: var(--accent); }}
            .thumb.selected {{ outline: 3px solid var(--accent); outline-offset: -3px; }}
            .bulk {{
              margin-top: 12px;
              display: flex;
              gap: 8px;
              flex-wrap: wrap;
              align-items: center;
              padding-top: 12px;
              border-top: 1px solid var(--stroke);
            }}
            .bulk .count {{
              font-size: 0.85rem;
              color: var(--ink-secondary);
              min-width: 100px;
              font-weight: 600;
            }}
          </style>
        </head>
        <body>
          {self._render_nav("snapshots")}
          <div class="shell">
            <div class="page-header">
              <h1>Snapshot Review</h1>
              <p>{len(files)} snapshots to review</p>
            </div>
          {"<p class='card'>No snapshots found.</p>" if not files else f'''
          <section class="viewer">
            <a id="viewer-link" href="{first_src}" target="_blank" rel="noopener noreferrer">
              <img id="viewer-image" src="{first_src}" alt="Selected snapshot" />
            </a>
            <div class="viewer-meta" id="viewer-source">{html.escape(first_source)}</div>
            <div class="viewer-state"><strong>Label:</strong> <span id="detector-state">{html.escape(first_review.get('detector_label') or 'unreviewed')}</span> · <strong>Person:</strong> <span id="person-state">{html.escape(first_review.get('person_id') or 'none')}</span></div>
            <div class="actions">
              <form method="post" action="/snapshot/review-detector">
                <input type="hidden" name="source_image" id="person-source-1" value="{html.escape(first_source)}" />
                <input type="hidden" name="label" value="person" />
                <button class="btn btn-primary" type="submit">Mark Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector">
                <input type="hidden" name="source_image" id="person-source-2" value="{html.escape(first_source)}" />
                <input type="hidden" name="label" value="no_person" />
                <button class="btn" type="submit">Mark No Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector">
                <input type="hidden" name="source_image" id="person-source-3" value="{html.escape(first_source)}" />
                <input type="hidden" name="label" value="clear" />
                <button class="btn" type="submit">Clear Label</button>
              </form>
            </div>
            <div class="actions">
              <form method="post" action="/snapshot/assign-person">
                <input type="hidden" name="source_image" id="assign-source" value="{html.escape(first_source)}" />
                <input type="text" name="target_person" placeholder="person_id or person name" />
                <button class="btn" type="submit">Assign Person</button>
              </form>
              <form method="post" action="/snapshot/assign-person">
                <input type="hidden" name="source_image" id="clear-person-source" value="{html.escape(first_source)}" />
                <input type="hidden" name="target_person" value="" />
                <button class="btn" type="submit">Clear</button>
              </form>
            </div>
            <div class="bulk">
              <span class="count">Selected: <span id="selected-count">0</span></span>
              <button class="btn" type="button" id="select-all">Select All</button>
              <button class="btn" type="button" id="clear-selected">Clear Selection</button>
              <form method="post" action="/snapshot/review-detector-bulk">
                <input type="hidden" name="selected_sources" id="bulk-sources" value="" />
                <input type="hidden" name="label" value="person" />
                <button class="btn btn-primary" type="submit">Bulk Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector-bulk">
                <input type="hidden" name="selected_sources" id="bulk-sources-2" value="" />
                <input type="hidden" name="label" value="no_person" />
                <button class="btn" type="submit">Bulk No Person</button>
              </form>
              <form method="post" action="/snapshot/review-detector-bulk">
                <input type="hidden" name="selected_sources" id="bulk-sources-3" value="" />
                <input type="hidden" name="label" value="clear" />
                <button class="btn" type="submit">Bulk Clear</button>
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
          </div>
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
          <title>DVR Console - {display_name}</title>
          {self._shared_styles()}
          <style>
            .person-header {{
              display: flex;
              align-items: center;
              gap: 16px;
              margin-bottom: 20px;
            }}
            .person-header h1 {{ margin: 0; }}
            .person-header .badge {{
              background: var(--accent-bg);
              color: var(--accent);
              padding: 4px 10px;
              border-radius: 999px;
              font-size: 0.8rem;
              font-weight: 600;
            }}
            .person-actions {{
              display: flex;
              gap: 12px;
              margin-bottom: 20px;
              flex-wrap: wrap;
            }}
            .rename-form {{
              display: flex;
              gap: 8px;
              flex: 1;
              max-width: 400px;
            }}
            .rename-form input {{
              flex: 1;
              padding: 8px 12px;
              border: 1px solid var(--stroke-strong);
              border-radius: 8px;
              font-size: 0.875rem;
            }}
            .viewer {{
              background: var(--card-bg);
              border-radius: 12px;
              border: 1px solid var(--stroke);
              padding: 16px;
              box-shadow: var(--card-shadow);
              margin-bottom: 16px;
            }}
            .viewer img {{
              width: 100%;
              max-height: 65vh;
              object-fit: contain;
              display: block;
              background: var(--bg-tertiary);
              border-radius: 8px;
            }}
            .viewer-info {{
              margin-top: 12px;
              font-size: 0.85rem;
              color: var(--ink-muted);
            }}
            .viewer-path {{
              word-break: break-all;
              margin-top: 4px;
              font-size: 0.8rem;
            }}
            .moderation {{
              margin-top: 12px;
              display: flex;
              gap: 8px;
              flex-wrap: wrap;
              align-items: center;
              padding-top: 12px;
              border-top: 1px solid var(--stroke);
            }}
            .moderation input {{
              padding: 8px 12px;
              border: 1px solid var(--stroke-strong);
              border-radius: 8px;
              min-width: 200px;
              font-size: 0.875rem;
            }}
            .sample-info {{
              font-size: 0.85rem;
              color: var(--ink-secondary);
            }}
            .thumbs {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
              gap: 10px;
            }}
            .thumb {{
              border: 2px solid transparent;
              border-radius: 8px;
              background: var(--bg-secondary);
              padding: 0;
              cursor: pointer;
              overflow: hidden;
              transition: all 120ms ease;
            }}
            .thumb:hover {{
              transform: translateY(-2px);
              box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            }}
            .thumb img {{
              width: 100%;
              height: 90px;
              object-fit: cover;
              display: block;
              background: var(--bg-tertiary);
            }}
            .thumb.active {{ border-color: var(--accent); }}
          </style>
        </head>
        <body>
          {self._render_nav("people")}
          <div class="shell">
            <div class="person-header">
              <h1>{display_name}</h1>
              <span class="badge">{len(samples)} samples</span>
            </div>
            <div class="person-actions">
              <form class="rename-form" method="post" action="/person/{quote(person_id)}/rename">
                <input type="text" name="display_name" value="{html.escape(display_name_raw)}" placeholder="Display name" />
                <button class="btn btn-primary" type="submit">Save</button>
              </form>
              <form method="post" action="/person/{quote(person_id)}/delete" onsubmit="return confirm('Delete this person and all associated samples?');">
                <button class="btn btn-danger" type="submit">Delete Person</button>
              </form>
            </div>
          {"<p class='card'>No samples.</p>" if not items else f'''
          <section class="viewer">
            <a id="viewer-link" href="{first_src}" target="_blank" rel="noopener noreferrer">
              <img id="viewer-image" src="{first_src}" alt="Selected full-size snapshot" />
            </a>
            <div class="viewer-info" id="viewer-meta">{first_meta}</div>
            <div class="viewer-path" id="viewer-path">{first_path}</div>
            <div class="moderation">
              <span class="sample-info">Sample: <strong id="selected-sample-id"></strong></span>
              <form method="post" action="/sample/move">
                <input type="hidden" name="sample_id" id="move-sample-id" value="" />
                <input type="hidden" name="return_person" value="{person_id_esc}" />
                <input type="text" name="target_person_id" placeholder="Move to person..." required />
                <button class="btn" type="submit">Move</button>
              </form>
              <form method="post" action="/sample/delete" onsubmit="return confirm('Delete this sample from DB?');">
                <input type="hidden" name="sample_id" id="delete-sample-id" value="" />
                <input type="hidden" name="return_person" value="{person_id_esc}" />
                <button class="btn btn-danger" type="submit">Delete Sample</button>
              </form>
            </div>
          </section>
          <div class="thumbs" id="thumbs">{''.join(items)}</div>
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
          </div>
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

        if parsed.path == "/health":
            self._send_health()
            return

        if parsed.path == "/metrics":
            self._send_metrics()
            return

        if parsed.path == "/people":
            self._send_html(self._render_people())
            return

        if parsed.path == "/guide":
            self._send_html(self._render_guide())
            return

        if parsed.path == "/models":
            query = parse_qs(parsed.query)
            message = (query.get("msg", [""])[0] or "").strip()
            ok_raw = (query.get("ok", ["1"])[0] or "1").strip().lower()
            ok = ok_raw not in {"0", "false", "no"}
            self._send_html(self._render_models(message=message, ok=ok))
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

        if parsed.path == "/models/import":
            importer = self.model_importer
            if importer is None:
                self.send_error(503, "Model importer unavailable")
                return

            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_error(400, "Expected multipart/form-data")
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

            file_field = form["model_file"] if "model_file" in form else None
            if file_field is None or not getattr(file_field, "file", None):
                self._redirect("/models?ok=0&msg=" + quote("Missing model file"))
                return

            filename = Path(getattr(file_field, "filename", "") or "upload.pt").name
            if filename.lower().endswith(".pt") is False:
                self._redirect("/models?ok=0&msg=" + quote("Only .pt files are supported"))
                return

            replace_base = bool(form.getfirst("replace_base", ""))
            try:
                tmp_dir = ROOT / "data" / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(delete=False, suffix=".pt", prefix="model_upload_", dir=str(tmp_dir)) as tmp:
                    tmp_path = Path(tmp.name)
                    shutil.copyfileobj(file_field.file, tmp)
                ok, message = importer(tmp_path, replace_base)
            except Exception as exc:
                ok = False
                message = f"Model import failed: {exc}"
            finally:
                try:
                    if "tmp_path" in locals() and tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

            self._redirect("/models?ok=" + ("1" if ok else "0") + "&msg=" + quote(message))
            return

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
    model_importer: Optional[Callable[[Path, bool], tuple[bool, str]]] = None,
    model_status_provider: Optional[Callable[[], dict[str, object]]] = None,
    stats_provider: Optional[Callable[[], dict[str, object]]] = None,
) -> ThreadingHTTPServer:
    """Create configured gallery HTTP server instance."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    FaceGalleryHandler.db_path = db_path
    FaceGalleryHandler.snapshot_dir = snapshot_dir
    FaceGalleryHandler.camera_map = camera_map
    FaceGalleryHandler.settings = settings
    FaceGalleryHandler.live_frame_provider = live_frame_provider
    FaceGalleryHandler.model_importer = model_importer
    FaceGalleryHandler.model_status_provider = model_status_provider
    FaceGalleryHandler.stats_provider = stats_provider
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
    model_importer: Optional[Callable[[Path, bool], tuple[bool, str]]] = None,
    model_status_provider: Optional[Callable[[], dict[str, object]]] = None,
    stats_provider: Optional[Callable[[], dict[str, object]]] = None,
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
        model_importer=model_importer,
        model_status_provider=model_status_provider,
        stats_provider=stats_provider,
    )
    thread = threading.Thread(target=server.serve_forever, name="face-gallery", daemon=True)
    thread.start()
    return server, thread

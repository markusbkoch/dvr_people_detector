# RTSP or ISAPI Person Detection + Alerts

This app reads credentials from `.secrets`, captures frames from your DVR channels, detects people, saves snapshots locally, and optionally sends Telegram alerts.

## Cameras and URL patterns

Configured channel IDs:

- `101`: Camera1
- `201`: Camera2
...

Capture endpoints used by the app:

- RTSP: `rtsp://<DVR_USERNAME>:<DVR_PASSWORD>@<DVR_IP>:554/Streaming/Channels/<CHANNEL_ID>`
- ISAPI snapshot: `http://<DVR_IP>/ISAPI/Streaming/channels/<CHANNEL_ID>/picture`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure `.secrets` (example):

```dotenv
DVR_USERNAME=admin
DVR_PASSWORD=***
DVR_IP=***
CAMERA_CHANNELS=101:Camera1;201:Camera2

CAPTURE_MODE=isapi
ISAPI_AUTH_MODE=auto
ISAPI_TIMEOUT_SECONDS=4

PERIODIC_ALERT_SECONDS=-1
MIN_PERSON_CONFIDENCE_FOR_ALERT=0.65
MIN_PERSON_CONFIDENCE_FOR_LOW_CONF_REVIEW=0.05
LOW_CONF_REVIEW_COOLDOWN_SECONDS=15
PERSON_MIN_BOX_AREA_PX=0
PERSON_MIN_MOVEMENT_PX=0
DETECTION_CONFIRMATION_FRAMES=3
DETECTION_CONFIRMATION_WINDOW_SECONDS=1.5
IGNORED_PERSON_BBOXES=501:304,180,316,205
DETECT_EVERY_SECONDS=1.2
RTSP_DETECTION_FPS=6.0
ALERT_COOLDOWN_SECONDS=40
BURST_CAPTURE_SECONDS=4.0
BURST_MAX_FRAMES=12
FRAME_QUEUE_SIZE=120

SNAPSHOT_TARGET_ASPECT_RATIO=16:9
SNAPSHOT_DIR=data/snapshots
LIVE_PREVIEW_FPS=4.0

FACE_DB_PATH=data/faces.db
FACE_MATCH_THRESHOLD=0.80
FACE_MIN_SAMPLES=3

RTSP_TRANSPORT=tcp
PERSON_TRACKER=bytetrack
CAMERA_RECONNECT_SECONDS=5
YOLO_MODEL=detection_models/yolov8n.pt

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
STATUS_REPORT_INTERVAL_HOURS=12
```

4. Run:

```bash
./run.sh
```

Alternative:

```bash
source .venv/bin/activate
python main.py
```

Logs are written to:

- console (stdout)
- rotating file: `logs/surveillance.log` (10MB per file, 5 backups)


## Face Gallery Viewer

`main.py` now serves the gallery UI in the same process.

```bash
source .venv/bin/activate
python main.py
```

Then open `http://127.0.0.1:8765`.

- Built-in guide: `http://127.0.0.1:8765/guide`
- `Snapshot Review` page shows all snapshots (not just face-detected samples).
- `Live Feed` page reads in-memory frames from the running surveillance process (no extra RTSP sessions and no disk handoff).
- For each snapshot you can:
  - mark detector feedback (`Mark Person` / `Mark No Person`)
  - optionally assign a `person_id/name` when a face is visible
  - bulk select multiple thumbnails and apply `Person` / `No Person` / `Clear Label` in one action
- `People Gallery` remains available for managing clustered face samples.

## Detector Feedback Export (YOLO)

After reviewing samples in the gallery, export detector training data:

```bash
python scripts/export_detector_dataset.py \
  --db-path data/faces.db \
  --output-dir data/detector_dataset \
  --model detection_models/yolov8n.pt \
  --confidence 0.35 \
  --clean
```

This generates:

- `data/detector_dataset/images/train`, `images/val`
- `data/detector_dataset/labels/train`, `labels/val`
- `data/detector_dataset/dataset.yaml`

Notes:

- Uses labels from `snapshot_reviews` (`Mark Person` / `Mark No Person`) across all reviewed snapshots.
- `no_person` becomes a negative example (empty label file).
- `person` uses YOLO person boxes from the current model.
- If a `person` review has no detected box, it is skipped and reported.


## Telegram Commands

When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured, you can send commands to the bot from that chat:

- `/ping`: quick liveness check (`pong`).
- `/status` (or `/report`): app health + counters.
- `/noperson` (reply to a snapshot alert): mark that snapshot as `no_person` in `snapshot_reviews`.
- `/pause`: pause Telegram notifications indefinitely.
- `/pause <hours>`: pause Telegram notifications for a fixed duration.
- `/resume`: resume Telegram notifications immediately.
- `/reload_model`: export reviewed detector dataset, train from `detection_models/yolov8n_base.pt`, copy result to managed model files, then reload workers.
- `/reload_model <model_path>`: same pipeline, but use `<model_path>` as base model.
- `/setstatus <hours|off>`: change periodic status-report interval at runtime.

Detector feedback from Telegram:

- React with `👎` on a snapshot alert to mark it as `no_person`.
- If reactions are unavailable in your chat type, reply to the snapshot with `/noperson`.

Pause behavior:

- While paused, detection and snapshot processing continue normally.
- Snapshot/photo alerts are suppressed.
- Periodic status reports continue to be sent.

Model update behavior (`/reload_model`):

- Runs in background and may take minutes.
- Step 1: runs `scripts/export_detector_dataset.py` with `--clean`.
- Step 2: trains YOLO and writes a new run under `data/detector_training/`.
- Step 3: copies `best.pt` to `detection_models/yolov8n_<timestamp>.pt` and to `detection_models/yolov8n.pt`.
- Step 4: reloads workers with `detection_models/yolov8n.pt`.

`/status` includes:

- uptime and camera count
- since last report: frames downloaded, snapshots saved, detections, high-confidence detections
- alerts sent and errors
- running totals

High-confidence means confidence `> 0.8 * MIN_PERSON_CONFIDENCE_FOR_ALERT`.

Periodic status reports are enabled by default every 12 hours and can be configured with `STATUS_REPORT_INTERVAL_HOURS`.

## Configuration Reference

### Required DVR credentials

- `DVR_USERNAME`
  - DVR/NVR login username.
  - Required.
- `DVR_PASSWORD`
  - DVR/NVR login password.
  - Required.
- `DVR_IP`
  - DVR/NVR IP or hostname used to build RTSP/ISAPI URLs.
  - Required.
- `CAMERA_CHANNELS`
  - Channel map used to build camera list at runtime.
  - Required.
  - Format: `channel_id:name` entries separated by `;`.
  - Example: `101:Camera1;201:Camera2`.

### Capture mode and camera access

- `CAPTURE_MODE` (default: `isapi`)
  - `isapi`: poll JPEG snapshots via ISAPI.
  - `rtsp`: read stream frames continuously.
- `CAMERA_RECONNECT_SECONDS` (default: `5`)
  - Wait time before retry when a camera read fails.
- `RTSP_TRANSPORT` (default: `tcp`)
  - RTSP transport for OpenCV/FFmpeg.
  - Typical values: `tcp`, `udp`.
- `PERSON_TRACKER` (default: `bytetrack`)
  - Tracker backend used in RTSP mode.
  - Supported values: `bytetrack`, `botsort`, `none`.
  - `none` disables tracker integration and uses per-frame detection only.
- `ISAPI_AUTH_MODE` (default: `auto`)
  - Auth mode for ISAPI requests.
  - Supported values: `auto`, `digest`, `basic`.
  - `auto` currently uses digest auth.
- `ISAPI_TIMEOUT_SECONDS` (default: `4`)
  - Timeout per ISAPI HTTP request.

### Detection and polling

- `YOLO_MODEL` (default: `detection_models/yolov8n.pt`)
  - YOLO model file used by `ultralytics`.
  - You can switch to larger models (`yolov8s.pt`, etc.) for accuracy at higher CPU/GPU cost.
- `MIN_PERSON_CONFIDENCE_FOR_ALERT` (default: `0.65`)
  - Minimum confidence to treat class `person` as detected.
  - Lower value increases sensitivity and false positives.
- `MIN_PERSON_CONFIDENCE_FOR_LOW_CONF_REVIEW` (default: `0.05`)
  - Minimum confidence required to emit "Low-confidence person detection suppressed" log lines.
  - Helps reduce log noise from ultra-low-confidence tracker candidates.
- `LOW_CONF_REVIEW_COOLDOWN_SECONDS` (default: `15`)
  - Minimum time between low-confidence snapshot writes per camera to avoid disk flooding.
  - Low-confidence snapshots are saved in `SNAPSHOT_DIR` with `lowconf_` filename prefix.
- `PERSON_MIN_BOX_AREA_PX` (default: `0`)
  - Minimum person bounding-box area in pixels to consider detection valid.
  - Set `0` to disable area filtering.
- `PERSON_MIN_MOVEMENT_PX` (default: `0`)
  - Minimum track-box center movement (pixels) required across confirmation window.
  - Useful to suppress static-object false positives. Set `0` to disable.
- `DETECTION_CONFIRMATION_FRAMES` (default: `3`)
  - Number of positive detection frames required before starting alert flow.
- `DETECTION_CONFIRMATION_WINDOW_SECONDS` (default: `1.5`)
  - Max time gap allowed between confirmation hits for the same camera.
- `IGNORED_PERSON_BBOXES` (default: empty)
  - Camera-specific ignore zones for person detections.
  - Format: `channel:x1,y1,x2,y2;channel:x1,y1,x2,y2|x1,y1,x2,y2`
  - Example: `501:304,180,316,205` ignores detections centered in that box on camera 501.
- `DETECT_EVERY_SECONDS` (default: `1.2`)
  - Per-camera polling interval used by ISAPI mode poller threads.
  - Lower values increase CPU/network load.
- `RTSP_DETECTION_FPS` (default: `6.0`)
  - Target detection rate per camera in RTSP mode.
  - RTSP ingest is continuous; frames are sampled for inference at this FPS.
  - Increase for responsiveness, decrease to reduce CPU/GPU load.
- `FRAME_QUEUE_SIZE` (default: `120`)
  - Max buffered frames per camera between polling and processing.
  - If full, oldest frames are dropped to keep latency low.

### Notification behavior

- `PERIODIC_ALERT_SECONDS` (default: `-1`)
  - `-1`: detection-only mode (notify/save only when person detected).
  - `>0`: always notify/save every `X` seconds per camera, regardless of detection.
- `ALERT_COOLDOWN_SECONDS` (default: `40`)
  - Cooldown between completed detection-triggered alerts per camera.
  - Applies after burst selection finishes, not to periodic alerts.
- `BURST_CAPTURE_SECONDS` (default: `4.0`)
  - After first person detection, keep collecting frames for this many seconds to find a better face shot.
- `BURST_MAX_FRAMES` (default: `12`)
  - Hard cap on frames evaluated during a burst window.
  - Burst closes when either time or frame cap is reached.
- `TELEGRAM_BOT_TOKEN`
  - Telegram bot token from BotFather.
  - Optional.
- `TELEGRAM_CHAT_ID`
  - Destination chat/user ID for Telegram alerts.
  - Optional.
- `STATUS_REPORT_INTERVAL_HOURS` (default: `12`)
  - Periodic Telegram status-report interval in hours.
  - Set to `0` to disable periodic reports.
- Telegram behavior when unset
  - If token/chat ID are missing, the app still detects and saves snapshots locally, but skips Telegram messages.

### Face identification

- `FACE_DB_PATH` (default: `data/faces.db`)
  - SQLite database with identities and face samples.
- `FACE_MATCH_THRESHOLD` (default: `0.80`)
  - Cosine similarity threshold for matching a detected face to a known person.
  - Increase for stricter matching; decrease for more aggressive matching.
- `FACE_MIN_SAMPLES` (default: `3`)
  - Minimum samples required for a person to be considered matchable.
  - Helps reduce noisy early matches.

### Snapshot output

- `SNAPSHOT_DIR` (default: `data/snapshots`)
  - Directory where snapshots are written.
- `LIVE_PREVIEW_FPS` (default: `4.0`)
  - Max publish rate per camera for in-memory web live previews in `main.py`.
  - Set `0` to disable live-preview publishing entirely.
- `SNAPSHOT_TARGET_ASPECT_RATIO` (default: `16:9`)
  - Aspect ratio correction applied before saving/sending.
  - Examples: `16:9`, `4:3`, `1.7778`, `off`.
  - Use `off` (or `none`, `-1`) to disable correction.

## Notes

- Detection pipeline is now split into per-camera workers:
  - Polling thread per camera (frame acquisition).
  - Processing thread per camera (detection, burst face scoring, notification).
- This prevents one camera's burst/processing from blocking other cameras.
- During burst mode, the app runs face detection on each frame and picks the best face-quality frame for alerting.
- Current face embeddings are baseline grayscale embeddings for bootstrapping. Upgrade to InsightFace/ArcFace next for stronger recognition.
- Low-confidence person detections (below `MIN_PERSON_CONFIDENCE_FOR_ALERT`) are logged and also saved to `SNAPSHOT_DIR` (rate-limited by `LOW_CONF_REVIEW_COOLDOWN_SECONDS`).

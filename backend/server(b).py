
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import json
import os

# =========================================================
# NEXA BACKEND
# =========================================================

app = FastAPI(title="NEXA Emergency Backend")

# ---------- PROJECT PATHS ----------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EVENTS_FILE = os.path.join(
    BASE_DIR, "backend", "events.json"
)

VIDEO_FOLDER = os.path.join(
    BASE_DIR, "emergency_clips"
)

DASHBOARD_FOLDER = os.path.join(
    BASE_DIR, "dashboard"
)

# Phase 2 candidate clips and event log
PHASE2_CLIP_FOLDER = os.path.join(
    BASE_DIR, "phase 2", "candidate_clips"
)

PHASE2_EVENTS_FILE = os.path.join(
    PHASE2_CLIP_FOLDER, "events.jsonl"
)

# ---------- CREATE FOLDERS ----------
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)
os.makedirs(PHASE2_CLIP_FOLDER, exist_ok=True)

print("BASE DIR:", BASE_DIR)
print("VIDEO FOLDER:", VIDEO_FOLDER)
print("PHASE 2 CLIP FOLDER:", PHASE2_CLIP_FOLDER)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- STATIC FILE ROUTES ----------

# Phase 1 emergency videos
app.mount(
    "/videos",
    StaticFiles(directory=VIDEO_FOLDER),
    name="videos",
)

# Phase 2 candidate videos
app.mount(
    "/candidate-clips",
    StaticFiles(directory=PHASE2_CLIP_FOLDER),
    name="candidate-clips",
)

# Dashboard
app.mount(
    "/dashboard",
    StaticFiles(directory=DASHBOARD_FOLDER, html=True),
    name="dashboard",
)


# =========================================================
# PHASE 1 - EMERGENCY HAND-SIGN DETECTION
# =========================================================

class EmergencyRequest(BaseModel):
    video: str | None = None


def load_events():
    """Load Phase 1 emergency events."""
    if not os.path.exists(EVENTS_FILE):
        return []

    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        print("Phase 1 event log error:", exc)
        return []


def save_events(events):
    """Save Phase 1 emergency events."""
    with open(EVENTS_FILE, "w", encoding="utf-8") as file:
        json.dump(events, file, indent=4)


@app.get("/")
def home():
    return {
        "system": "NEXA Emergency AI",
        "status": "online",
        "dashboard": "/dashboard/",
        "phase1_events": "/events",
        "phase2_events": "/phase2-events",
    }


@app.post("/emergency")
def emergency_alert(data: EmergencyRequest):
    """Receive an emergency alert from Phase 1."""
    events = load_events()

    alert = {
        "id": len(events) + 1,
        "event": "EMERGENCY_DETECTED",
        "status": "CRITICAL",
        "camera": "CAMERA-01",
        "timestamp": datetime.now().isoformat(),
        "video": data.video,
    }

    events.append(alert)
    save_events(events)

    print("\n" + "=" * 60)
    print("EMERGENCY ALERT RECEIVED")
    print("=" * 60)
    print("Event ID :", alert["id"])
    print("Camera   :", alert["camera"])
    print("Status   :", alert["status"])
    print("Timestamp:", alert["timestamp"])
    print("Video    :", alert["video"])
    print("=" * 60 + "\n")

    return alert


@app.get("/events")
def get_events():
    """Return Phase 1 emergency events."""
    return load_events()


# =========================================================
# PHASE 2 - VD-MIL CANDIDATE INCIDENT DETECTION
# =========================================================

@app.get("/phase2-events")
def get_phase2_events():
    """
    Read candidate events written by live_webcam.py.

    These are experimental model detections requiring
    human review, not confirmed incidents.
    """
    events = []

    if not os.path.exists(PHASE2_EVENTS_FILE):
        return []

    try:
        with open(
            PHASE2_EVENTS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                if not line.strip():
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    print("Skipping invalid Phase 2 event log line.")
                    continue

                # Use only the filename in the public response.
                raw_path = event.get("clip_path", "")
                filename = os.path.basename(raw_path) if raw_path else ""

                # Only expose MP4 files from the candidate folder.
                valid_clip = (
                    bool(filename)
                    and filename.lower().endswith(".mp4")
                    and os.path.isfile(
                        os.path.join(PHASE2_CLIP_FOLDER, filename)
                    )
                )

                event["clip_filename"] = filename if valid_clip else None
                event["clip_url"] = (
                    f"/candidate-clips/{filename}"
                    if valid_clip
                    else None
                )

                # Explicitly identify this as a candidate.
                event["event_type"] = "PHASE2_CANDIDATE"
                event["review_status"] = event.get(
                    "review_status",
                    "pending_human_review",
                )

                events.append(event)

    except OSError as exc:
        print("Phase 2 event log error:", exc)
        return []

    # Newest events first.
    return list(reversed(events))


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "backend": "online",
        "phase1": "available",
        "phase2": "available",
        "phase2_log_exists": os.path.exists(PHASE2_EVENTS_FILE),
    }

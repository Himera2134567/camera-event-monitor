"""Portable paths and external configuration; no user credentials in source."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
DATA_DIR = Path(os.environ.get("CAMERA_MONITOR_DATA", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CameraEventMonitor")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
EVENTS_FILE = DATA_DIR / "events.json"
EVENTS_DIR = DATA_DIR / "events"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(APP_DIR / "models" / "best.pt")))
SMTP_SERVER = os.environ.get("SMTP_SERVER", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_TO = os.environ.get("SMTP_TO", "")

"""Event logging to CSV.

Only meaningful events are written (never one row per frame), and only
numbers: timestamp, event type, risk score, EAR, MAR, head angles, phone
confidence and driver state. No video and no images are ever stored.

The file and its header are created automatically. Each event is appended in
one short write; events are rare, so this stays cheap.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter, deque
from datetime import datetime
from enum import Enum
from pathlib import Path

from app.config import CONFIG, AppConfig

logger = logging.getLogger(__name__)

COLUMNS = (
    "timestamp",
    "event_type",
    "risk_score",
    "EAR",
    "MAR",
    "pitch",
    "yaw",
    "roll",
    "phone_confidence",
    "state",
)


class EventType(str, Enum):
    DROWSINESS_WARNING = "DROWSINESS_WARNING"
    DROWSINESS_CRITICAL = "DROWSINESS_CRITICAL"
    YAWN = "YAWN"
    HEAD_NOD = "HEAD_NOD"
    LOOK_AWAY = "LOOK_AWAY"
    PHONE_DETECTED = "PHONE_DETECTED"
    PHONE_USE_CONFIRMED = "PHONE_USE_CONFIRMED"
    COMBINED_CRITICAL = "COMBINED_CRITICAL"


class EventLogger:
    """Appends events to a CSV file and keeps in-memory counts for the UI."""

    def __init__(self, config: AppConfig = CONFIG, path: Path | None = None) -> None:
        self._path = Path(path if path is not None else config.LOG_FILE_PATH)
        self.counts: Counter[str] = Counter()
        self.recent: deque[dict] = deque(maxlen=50)
        self._ensure_file()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_file(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not self._path.exists() or self._path.stat().st_size == 0:
                with self._path.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(COLUMNS)
        except OSError:
            logger.exception("Could not create log file %s", self._path)

    def log_event(
        self,
        event_type: EventType | str,
        *,
        risk_score: float,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        phone_confidence: float,
        state: str,
    ) -> None:
        name = event_type.value if isinstance(event_type, EventType) else str(event_type)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": name,
            "risk_score": round(risk_score, 1),
            "EAR": round(ear, 3),
            "MAR": round(mar, 3),
            "pitch": round(pitch, 1),
            "yaw": round(yaw, 1),
            "roll": round(roll, 1),
            "phone_confidence": round(phone_confidence, 2),
            "state": state,
        }
        self.counts[name] += 1
        self.recent.append(row)
        try:
            with self._path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=COLUMNS).writerow(row)
        except OSError:
            # A full disk or locked file must not stop the detection loop.
            logger.exception("Could not write event to %s", self._path)

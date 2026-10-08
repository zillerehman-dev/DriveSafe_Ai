"""Central configuration for DriveSafe AI.

Every tunable number lives here so detection code never contains magic
numbers. Modules import the shared ``CONFIG`` instance:

    from app.config import CONFIG
    if ear < CONFIG.EAR_THRESHOLD: ...

The config is a frozen dataclass: values cannot be changed by accident at
runtime. Tests can build their own instance, e.g.
``AppConfig(EAR_THRESHOLD=0.25)``, without touching global state.

All thresholds and weights are engineering starting points, NOT scientifically
validated values. Expect to tune them for your camera, lighting and face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    # ---- Camera -----------------------------------------------------------
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    MIRROR_PREVIEW: bool = True  # flip the preview like a mirror

    # ---- Drowsiness: eyes -------------------------------------------------
    # EAR below this value counts as "eye closed" for a frame.
    EAR_THRESHOLD: float = 0.21
    # A blink is shorter than this; anything longer is closure, not a blink.
    BLINK_MIN_SECONDS: float = 0.05  # shorter than this is landmark noise
    BLINK_MAX_SECONDS: float = 0.40
    # Blinks per minute (rolling window) treated as abnormally frequent.
    BLINK_RATE_HIGH_PER_MIN: float = 35.0
    BLINK_RATE_WINDOW_SECONDS: float = 60.0
    # Continuous closure durations that escalate the state.
    EYE_CLOSURE_WARNING_SECONDS: float = 1.0
    EYE_CLOSURE_CRITICAL_SECONDS: float = 2.0

    # ---- Drowsiness: mouth (yawn) ----------------------------------------
    MAR_THRESHOLD: float = 0.60
    YAWN_MIN_SECONDS: float = 1.0  # mouth must stay open this long

    # ---- Drowsiness: head pose (degrees) ----------------------------------
    HEAD_NOD_PITCH_DEGREES: float = 15.0  # downward tilt magnitude
    HEAD_NOD_MIN_SECONDS: float = 0.8

    # ---- Looking away (distraction), based on head yaw --------------------
    LOOK_AWAY_YAW_DEGREES: float = 25.0  # turned left or right by at least this much
    LOOK_AWAY_SECONDS: float = 5.0  # ...for this long before "Please look straight"
    # A far head turn often makes the face tracker lose the face; count that too.
    LOOK_AWAY_COUNTS_NO_FACE: bool = True

    # ---- Drowsiness: risk score and state thresholds (0-100) --------------
    DROWSINESS_RISK_THRESHOLD: float = 40.0  # enter WARNING
    CRITICAL_RISK_THRESHOLD: float = 85.0  # enter CRITICAL
    DROWSY_RISK_THRESHOLD: float = 70.0  # enter DROWSY
    # Hysteresis: risk must fall this far below a threshold before the state
    # drops, so tiny fluctuations do not make the state flicker.
    STATE_HYSTERESIS_MARGIN: float = 8.0
    # Minimum time a new state must hold before it is accepted.
    STATE_MIN_HOLD_SECONDS: float = 0.25

    # ---- Risk score weights (0-100 points each, capped at 100 in total) ---
    # Eye closure is the strongest signal; the rest are supporting evidence.
    WEIGHT_EYE_CLOSURE: float = 100.0
    WEIGHT_YAWN: float = 15.0
    WEIGHT_HEAD_NOD: float = 15.0
    WEIGHT_BLINK_ABNORMAL: float = 10.0
    # Extra risk added when phone use is confirmed at the same time.
    WEIGHT_PHONE_COMBINED: float = 20.0
    # How fast the risk score decays per second when signals disappear.
    RISK_DECAY_PER_SECOND: float = 60.0

    # ---- Phone detection --------------------------------------------------
    PHONE_CONFIDENCE_THRESHOLD: float = 0.20
    # Consecutive YOLO runs that must see a phone before use is "confirmed".
    PHONE_CONFIRMATION_FRAMES: int = 3
    # Missed YOLO runs tolerated before the confirmation streak resets.
    PHONE_MAX_MISSES: int = 2
    # Seconds a detection is kept alive between (skipped) YOLO runs.
    PHONE_PERSISTENCE_SECONDS: float = 1.0
    PHONE_ALERT_COOLDOWN: float = 8.0
    # Confirmed phone use lasting this long escalates to PHONE_CRITICAL.
    PHONE_CRITICAL_SECONDS: float = 6.0
    # COCO class id for "cell phone" in standard pretrained YOLO models.
    PHONE_CLASS_ID: int = 67

    # ---- Voice alerts -----------------------------------------------------
    ALERT_COOLDOWN_SECONDS: float = 6.0
    VOICE_RATE: int = 170  # words per minute for pyttsx3

    # ---- Performance ------------------------------------------------------
    YOLO_FRAME_SKIP: int = 3  # run YOLO on every Nth frame
    YOLO_IMAGE_SIZE: int = 640  # smaller = faster, less accurate
    TARGET_FPS: int = 30

    # ---- Paths ------------------------------------------------------------
    LOG_FILE_PATH: Path = PROJECT_ROOT / "data" / "logs" / "events.csv"
    YOLO_MODEL_PATH: Path = PROJECT_ROOT / "models" / "yolo_model.pt"
    # MediaPipe Face Landmarker needs its own .task file (separate from YOLO).
    FACE_LANDMARKER_PATH: Path = PROJECT_ROOT / "models" / "face_landmarker.task"


CONFIG = AppConfig()

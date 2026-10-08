"""Facial computer vision (MediaPipe Face Landmarker).

Turns a webcam frame into numbers the drowsiness engine can use:
EAR (eye openness), MAR (mouth openness), blink statistics, continuous
eye-closure time and head pose (pitch / yaw / roll).

Input:  a BGR frame (NumPy array) from OpenCV.
Output: ``VisionResult``. Bad frames or degenerate geometry never raise;
        they produce a result with ``face_detected=False`` or safe zeros.

Layout of this file:
  1. pure geometry helpers (no MediaPipe, easy to unit test)
  2. ``BlinkTracker`` (pure, time based)
  3. ``VisionProcessor`` (MediaPipe + OpenCV wiring)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import CONFIG, AppConfig

logger = logging.getLogger(__name__)

# --- MediaPipe Face Mesh landmark indices (468-point topology) ---------------
# Eye order is [p1, p2, p3, p4, p5, p6]: p1/p4 are the horizontal corners,
# p2/p3 the upper lid and p6/p5 the matching lower lid points.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)  # driver's right eye
LEFT_EYE = (362, 385, 387, 263, 373, 380)  # driver's left eye
# Inner-lip points: three vertical pairs and one horizontal pair.
MOUTH_VERTICAL_PAIRS = ((81, 178), (13, 14), (311, 402))
MOUTH_CORNERS = (78, 308)

# Six landmarks used for head pose, and a generic 3D face model (arbitrary
# units, roughly millimetres). Axes: x right, y down, z away from the camera,
# nose tip at the origin. Same order as HEAD_POSE_LANDMARKS.
HEAD_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)  # nose, chin, eyes, mouth
HEAD_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, 330.0, 65.0),  # chin
        (-225.0, -170.0, 135.0),  # right eye outer corner (image left)
        (225.0, -170.0, 135.0),  # left eye outer corner (image right)
        (-150.0, 150.0, 125.0),  # mouth corner (image left)
        (150.0, 150.0, 125.0),  # mouth corner (image right)
    ],
    dtype=np.float64,
)

_EPS = 1e-6


@dataclass
class VisionResult:
    face_detected: bool = False
    face_count: int = 0
    left_ear: float = 0.0
    right_ear: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    blink_count: int = 0
    blink_rate: float = 0.0  # blinks per minute over a rolling window
    eye_closed: bool = False
    eye_closure_duration: float = 0.0  # seconds of continuous low EAR
    pitch: float = 0.0  # degrees, positive = head tilted down
    yaw: float = 0.0
    roll: float = 0.0
    landmarks: np.ndarray | None = field(default=None, repr=False)  # (N, 2) pixels


# =============================================================================
# 1. Geometry helpers
# =============================================================================
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def calculate_eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|). Returns 0.0 on invalid input."""
    try:
        eye = np.asarray(eye, dtype=float)
        if eye.shape != (6, 2) or not np.all(np.isfinite(eye)):
            return 0.0
        width = _dist(eye[0], eye[3])
        if width < _EPS:
            return 0.0
        return (_dist(eye[1], eye[5]) + _dist(eye[2], eye[4])) / (2.0 * width)
    except (ValueError, TypeError):
        return 0.0


def calculate_mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """Mean inner-lip opening divided by mouth width. 0.0 on invalid input."""
    try:
        pts = np.asarray(landmarks, dtype=float)
        left, right = MOUTH_CORNERS
        width = _dist(pts[left], pts[right])
        if width < _EPS or not np.isfinite(width):
            return 0.0
        openings = [_dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL_PAIRS]
        return float(np.mean(openings)) / width
    except (IndexError, ValueError, TypeError):
        return 0.0


def estimate_head_pose(landmarks: np.ndarray, frame_width: int, frame_height: int) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees using solvePnP, or zeros on failure.

    Pitch is positive when the head tilts down (nodding). The camera matrix is
    approximated from the frame size, which is good enough for relative angles.
    """
    try:
        image_points = np.asarray([landmarks[i] for i in HEAD_POSE_LANDMARKS], dtype=np.float64)
        if image_points.shape != (6, 2) or not np.all(np.isfinite(image_points)):
            return 0.0, 0.0, 0.0
        focal = float(frame_width)
        camera = np.array(
            [[focal, 0, frame_width / 2.0], [0, focal, frame_height / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        ok, rvec, _ = cv2.solvePnP(
            HEAD_MODEL_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0, 0.0
        return _rotation_to_euler(cv2.Rodrigues(rvec)[0])
    except (cv2.error, IndexError, ValueError, TypeError):
        return 0.0, 0.0, 0.0


def _rotation_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Rz(roll) @ Ry(yaw) @ Rx(pitch) into degrees."""
    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = np.degrees(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return float(pitch), float(yaw), float(roll)


# =============================================================================
# 2. Blink tracking
# =============================================================================
class BlinkTracker:
    """Tracks blinks and continuous eye closure from a stream of EAR values.

    A blink is a closure that ends after BLINK_MIN..BLINK_MAX seconds. A longer
    closure is not counted as a blink: it is reported through
    ``closure_duration`` so the drowsiness engine can judge it.
    """

    def __init__(self, config: AppConfig = CONFIG) -> None:
        self._cfg = config
        self.blink_count = 0
        self._closed_since: float | None = None
        self._blink_times: deque[float] = deque()

    def update(self, ear: float, now: float) -> tuple[bool, float]:
        """Feed one EAR sample. Returns (eye_closed, closure_duration_seconds)."""
        closed = ear < self._cfg.EAR_THRESHOLD
        if closed:
            if self._closed_since is None:
                self._closed_since = now
            return True, max(0.0, now - self._closed_since)

        if self._closed_since is not None:  # eyes just reopened
            duration = now - self._closed_since
            if self._cfg.BLINK_MIN_SECONDS <= duration <= self._cfg.BLINK_MAX_SECONDS:
                self.blink_count += 1
                self._blink_times.append(now)
            self._closed_since = None
        return False, 0.0

    def reset_closure(self) -> None:
        """Forget the current closure (used when the face is lost)."""
        self._closed_since = None

    def blink_rate(self, now: float) -> float:
        window = self._cfg.BLINK_RATE_WINDOW_SECONDS
        while self._blink_times and now - self._blink_times[0] > window:
            self._blink_times.popleft()
        return len(self._blink_times) * 60.0 / window


# =============================================================================
# 3. MediaPipe wiring
# =============================================================================
class VisionProcessor:
    """Runs the Face Landmarker on frames and returns ``VisionResult``."""

    def __init__(self, config: AppConfig = CONFIG) -> None:
        self._cfg = config
        self._blinks = BlinkTracker(config)
        self._landmarker = self._create_landmarker(Path(config.FACE_LANDMARKER_PATH))
        self._last_timestamp_ms = -1

    @staticmethod
    def _create_landmarker(model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                "Download face_landmarker.task (see README, 'Model Files') into models/."
            )
        # Imported here so the geometry helpers work without MediaPipe installed.
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,  # 2 so we can warn when more than one face is visible
        )
        return vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, now: float | None = None) -> VisionResult:
        now = time.monotonic() if now is None else now
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            return VisionResult(blink_count=self._blinks.blink_count)

        height, width = frame.shape[:2]
        try:
            faces = self._detect(frame, now)
        except Exception:  # MediaPipe errors must not kill the app
            logger.exception("Face landmarker failed on a frame")
            faces = []

        if not faces:
            self._blinks.reset_closure()
            return VisionResult(blink_count=self._blinks.blink_count, blink_rate=self._blinks.blink_rate(now))

        # Keep the largest face: the driver is the face closest to the camera.
        points = max(
            (np.array([(p.x * width, p.y * height) for p in face], dtype=np.float64) for face in faces),
            key=lambda pts: np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]),
        )

        left_ear = calculate_eye_aspect_ratio(points[list(LEFT_EYE)])
        right_ear = calculate_eye_aspect_ratio(points[list(RIGHT_EYE)])
        ear = (left_ear + right_ear) / 2.0
        eye_closed, closure = self._blinks.update(ear, now)
        pitch, yaw, roll = estimate_head_pose(points, width, height)

        return VisionResult(
            face_detected=True,
            face_count=len(faces),
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            mar=calculate_mouth_aspect_ratio(points),
            blink_count=self._blinks.blink_count,
            blink_rate=self._blinks.blink_rate(now),
            eye_closed=eye_closed,
            eye_closure_duration=closure,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            landmarks=points,
        )

    def _detect(self, frame: np.ndarray, now: float):
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # VIDEO mode needs strictly increasing timestamps in milliseconds.
        timestamp_ms = max(int(now * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return self._landmarker.detect_for_video(image, timestamp_ms).face_landmarks

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            logger.debug("Landmarker close failed", exc_info=True)


def draw_landmarks(frame: np.ndarray, result: VisionResult, color: tuple[int, int, int]) -> None:
    """Draw eye and mouth landmarks in place (no-op when no face)."""
    if not result.face_detected or result.landmarks is None:
        return
    for index in (*LEFT_EYE, *RIGHT_EYE):
        cv2.circle(frame, tuple(int(v) for v in result.landmarks[index]), 2, color, -1)
    for eye in (LEFT_EYE, RIGHT_EYE):
        contour = result.landmarks[list(eye)].astype(np.int32)
        cv2.polylines(frame, [contour], True, color, 1, cv2.LINE_AA)
    mouth = [i for pair in MOUTH_VERTICAL_PAIRS for i in pair] + list(MOUTH_CORNERS)
    for index in mouth:
        cv2.circle(frame, tuple(int(v) for v in result.landmarks[index]), 2, color, -1)
    x0, y0 = result.landmarks.min(axis=0).astype(int)
    x1, y1 = result.landmarks.max(axis=0).astype(int)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)

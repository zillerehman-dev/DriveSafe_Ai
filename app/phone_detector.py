"""Mobile phone detection (Ultralytics YOLO).

Responsibilities:
  * load the YOLO model once,
  * run it only every ``YOLO_FRAME_SKIP`` frames (it is the slow part),
  * keep only the "cell phone" class above the confidence threshold,
  * confirm phone use only after several consecutive detections.

A phone visible for one frame is NOT phone use. ``PhoneConfirmation`` counts
consecutive YOLO runs that saw a phone and only reports ``confirmed`` after
``PHONE_CONFIRMATION_FRAMES`` of them, tolerating a few missed runs.

Input:  a BGR frame.   Output: ``PhoneDetection``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import CONFIG, AppConfig

logger = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class PhoneDetection:
    detected: bool = False
    confidence: float = 0.0
    bbox: BBox | None = None
    confirmed: bool = False


class PhoneConfirmation:
    """Temporal confirmation: N consecutive hits, tolerating a few misses."""

    def __init__(self, required_hits: int, max_misses: int) -> None:
        self._required = required_hits
        self._max_misses = max_misses
        self.hits = 0
        self._misses = 0

    def update(self, hit: bool) -> bool:
        if hit:
            self.hits += 1
            self._misses = 0
        else:
            self._misses += 1
            if self._misses > self._max_misses:
                self.hits = 0
        return self.hits >= self._required


class PhoneDetector:
    def __init__(self, config: AppConfig = CONFIG, model: Any | None = None) -> None:
        """Load the model. Pass ``model`` only to inject a stand-in (tests)."""
        self._cfg = config
        self._model = model if model is not None else self._load_model(Path(config.YOLO_MODEL_PATH))
        self._class_id = self._resolve_phone_class(self._model)
        self._confirmation = PhoneConfirmation(config.PHONE_CONFIRMATION_FRAMES, config.PHONE_MAX_MISSES)
        self._frame_index = 0
        self._last_hit: PhoneDetection | None = None
        self._last_hit_time = float("-inf")

    @staticmethod
    def _load_model(model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Phone detection model not found at {model_path}. "
                "Download a pretrained YOLO model into models/yolo_model.pt (see README, 'Model Files')."
            )
        from ultralytics import YOLO  # imported lazily: heavy import

        return YOLO(str(model_path))  # loaded once, reused for every frame

    def _resolve_phone_class(self, model: Any) -> int:
        """Find the 'cell phone' class by name; fall back to the COCO id."""
        names = getattr(model, "names", None) or {}
        for class_id, name in dict(names).items():
            if "phone" in str(name).lower():
                return int(class_id)
        return self._cfg.PHONE_CLASS_ID

    def detect(self, frame: np.ndarray, now: float | None = None) -> PhoneDetection:
        now = time.monotonic() if now is None else now
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self._current(now)

        run_now = self._frame_index % max(1, self._cfg.YOLO_FRAME_SKIP) == 0
        self._frame_index += 1
        if run_now:
            hit = self._infer(frame)
            self._confirmation.update(hit is not None)
            if hit is not None:
                self._last_hit, self._last_hit_time = hit, now
        return self._current(now)

    def _current(self, now: float) -> PhoneDetection:
        """Detection state for this frame, keeping the last box alive briefly
        so it does not flicker between skipped YOLO runs."""
        fresh = self._last_hit is not None and now - self._last_hit_time <= self._cfg.PHONE_PERSISTENCE_SECONDS
        if not fresh or self._last_hit is None:
            return PhoneDetection()
        confirmed = self._confirmation.hits >= self._cfg.PHONE_CONFIRMATION_FRAMES
        return PhoneDetection(True, self._last_hit.confidence, self._last_hit.bbox, confirmed)

    def _infer(self, frame: np.ndarray) -> PhoneDetection | None:
        """Run YOLO; return the most confident phone or None. Never raises."""
        try:
            results = self._model.predict(
                frame,
                imgsz=self._cfg.YOLO_IMAGE_SIZE,
                conf=self._cfg.PHONE_CONFIDENCE_THRESHOLD,
                classes=[self._class_id],
                verbose=False,
            )
            best: PhoneDetection | None = None
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if best is None or confidence > best.confidence:
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                        best = PhoneDetection(True, confidence, (x1, y1, x2, y2))
            return best
        except Exception:
            logger.exception("YOLO inference failed on a frame")
            return None


def draw_phone(frame: np.ndarray, detection: PhoneDetection) -> None:
    """Draw the phone bounding box in place (no-op when nothing detected)."""
    if not detection.detected or detection.bbox is None:
        return
    color = (0, 0, 255) if detection.confirmed else (0, 165, 255)
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"PHONE {detection.confidence:.0%}"
    cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

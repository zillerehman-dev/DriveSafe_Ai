"""Local real-time application (primary demo).

``DriveSafePipeline`` wires the modules together for one frame:
    vision -> phone detector -> drowsiness engine -> alerts + logger -> overlay
The webcam loop (``run``) and the Streamlit dashboard both use it, so the
behaviour is identical in both places. This file contains no EAR math, no YOLO
code and no text-to-speech code; those live in their own modules.

Run with:  python -m app.main      (press q or Esc to quit)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.alerts import AlertType, VoiceAlertManager, alert_for_status
from app.config import CONFIG, AppConfig
from app.drowsiness import (
    DriverState,
    DrowsinessDetector,
    DrowsinessInput,
    DrowsinessResult,
    OverallStatus,
    PhoneState,
)
from app.logger import EventLogger
from app.phone_detector import PhoneDetection, PhoneDetector, draw_phone
from app.vision import VisionProcessor, VisionResult, draw_landmarks

logger = logging.getLogger(__name__)

# BGR colours per driver state.
STATE_COLORS = {
    DriverState.AWAKE: (0, 200, 0),
    DriverState.WARNING: (0, 165, 255),
    DriverState.DROWSY: (0, 0, 255),
    DriverState.CRITICAL: (0, 0, 170),
}
STATUS_BANNERS = {
    OverallStatus.CRITICAL_DROWSINESS_PHONE: "CRITICAL: DROWSINESS + PHONE USE",
    OverallStatus.CRITICAL_DROWSINESS: "CRITICAL: DROWSINESS",
    OverallStatus.PHONE_USE: "WARNING: PHONE USE",
    OverallStatus.LOOKING_AWAY: "WARNING: PLEASE LOOK STRAIGHT",
}
MSG_NO_FACE = "No driver face detected."
MSG_MULTI_FACE = "Multiple faces detected. Ensure only the driver is visible."
MSG_NO_PHONE_MODEL = "Phone detection model not found."


@dataclass
class FrameAnalysis:
    vision: VisionResult
    phone: PhoneDetection
    drowsiness: DrowsinessResult
    frame: np.ndarray  # annotated frame
    messages: list[str] = field(default_factory=list)


class DriveSafePipeline:
    def __init__(self, config: AppConfig = CONFIG, enable_voice: bool = True) -> None:
        self._cfg = config
        self.vision = VisionProcessor(config)  # raises FileNotFoundError if model missing
        self.phone_detector: PhoneDetector | None = None
        self.startup_warnings: list[str] = []
        try:
            self.phone_detector = PhoneDetector(config)
        except FileNotFoundError as error:
            logger.error("%s", error)
            self.startup_warnings.append(MSG_NO_PHONE_MODEL)
        except Exception:  # e.g. ultralytics not installed
            logger.exception("Phone detector could not start")
            self.startup_warnings.append(MSG_NO_PHONE_MODEL)

        self.detector = DrowsinessDetector(config)
        self.events = EventLogger(config)
        self.alerts = VoiceAlertManager(config) if enable_voice else None

    def process(self, frame: np.ndarray, now: float | None = None) -> FrameAnalysis | None:
        """Analyse one frame. Returns None for an invalid frame (skip it)."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            return None
        now = time.monotonic() if now is None else now

        vision = self.vision.process_frame(frame, now)
        phone = self.phone_detector.detect(frame, now) if self.phone_detector else PhoneDetection()
        phone_state = PhoneState.from_flags(phone.detected, phone.confirmed)

        result = self.detector.update(
            DrowsinessInput(
                face_detected=vision.face_detected,
                ear=vision.ear,
                mar=vision.mar,
                eye_closure_duration=vision.eye_closure_duration,
                blink_rate=vision.blink_rate,
                pitch=vision.pitch,
                phone_state=phone_state,
                yaw=vision.yaw,
            ),
            now,
        )

        for event in result.new_events:
            self.events.log_event(
                event,
                risk_score=result.risk_score,
                ear=vision.ear,
                mar=vision.mar,
                pitch=vision.pitch,
                yaw=vision.yaw,
                roll=vision.roll,
                phone_confidence=phone.confidence,
                state=result.state.name,
            )
        if self.alerts is not None:
            alert = alert_for_status(result.status, result.phone_duration, self._cfg)
            # Eyes are open again: end the drowsiness alert so the next closure is announced.
            if vision.eye_closure_duration < self._cfg.BLINK_MAX_SECONDS:
                if alert in (AlertType.DROWSINESS_WARNING, AlertType.DROWSINESS_CRITICAL):
                    alert = None
                elif alert == AlertType.COMBINED_CRITICAL:
                    alert = AlertType.PHONE_WARNING
            self.alerts.request(alert)

        messages = list(self.startup_warnings)
        if not vision.face_detected:
            messages.append(MSG_NO_FACE)
        elif vision.face_count > 1:
            messages.append(MSG_MULTI_FACE)

        annotated = frame.copy()
        color = STATE_COLORS[result.state]
        draw_landmarks(annotated, vision, color)
        draw_phone(annotated, phone)
        return FrameAnalysis(vision, phone, result, annotated, messages)

    def close(self) -> None:
        self.vision.close()
        if self.alerts is not None:
            self.alerts.close()


# =============================================================================
# Overlay drawing
# =============================================================================
def draw_overlay(analysis: FrameAnalysis, fps: float) -> np.ndarray:
    """Draw the status panel, banners and messages onto the annotated frame."""
    frame = analysis.frame
    vision, phone, result = analysis.vision, analysis.phone, analysis.drowsiness
    color = STATE_COLORS[result.state]
    height, width = frame.shape[:2]

    lines = [
        ("DRIVESAFE AI", (255, 255, 255)),
        (f"Status: {result.state.name}", color),
        (f"Risk: {result.risk_score:.0f}% ({result.risk_level})", color),
        (f"EAR: {vision.ear:.2f}   MAR: {vision.mar:.2f}", (255, 255, 255)),
        (f"Eyes closed: {result.eye_closure_duration:.1f} sec", (255, 255, 255)),
        (f"Blinks: {vision.blink_count}", (255, 255, 255)),
        (f"Head P/Y/R: {vision.pitch:.0f}/{vision.yaw:.0f}/{vision.roll:.0f}", (255, 255, 255)),
        (_phone_line(phone, result.phone_state), (0, 165, 255) if phone.detected else (255, 255, 255)),
        (f"FPS: {fps:.0f}", (200, 200, 200)),
    ]
    panel_h = 12 + 22 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (text, line_color) in enumerate(lines):
        cv2.putText(frame, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 1, cv2.LINE_AA)

    banner = STATUS_BANNERS.get(result.status)
    if banner:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)
        cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 140), -1)
        cv2.putText(frame, banner, (15, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    for i, message in enumerate(analysis.messages):
        cv2.putText(frame, message, (10, panel_h + 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
    return frame


def _phone_line(phone: PhoneDetection, state: PhoneState) -> str:
    if state == PhoneState.PHONE_USE_CONFIRMED:
        return f"Phone: IN USE ({phone.confidence:.0%})"
    if state == PhoneState.PHONE_DETECTED:
        return f"Phone: DETECTED ({phone.confidence:.0%})"
    return "Phone: NOT DETECTED"


# =============================================================================
# Webcam loop
# =============================================================================
def run(config: AppConfig = CONFIG) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    capture = cv2.VideoCapture(config.CAMERA_INDEX)
    if not capture.isOpened():
        print("Unable to access camera.")
        return 1
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    try:
        pipeline = DriveSafePipeline(config)
    except FileNotFoundError as error:
        capture.release()
        print(error)
        return 1

    fps, last = 0.0, time.monotonic()
    bad_reads = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                bad_reads += 1
                if bad_reads > 60:
                    print("Unable to access camera.")
                    return 1
                continue
            bad_reads = 0
            if config.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1.0 / max(now - last, 1e-6)
            last = now

            try:
                analysis = pipeline.process(frame, now)
            except Exception:  # one bad frame must not end the session
                logger.exception("Frame processing failed; skipping frame")
                continue
            if analysis is None:
                continue

            cv2.imshow("DriveSafe AI", draw_overlay(analysis, fps))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        pipeline.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(run())

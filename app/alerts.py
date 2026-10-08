"""Voice alerts (pyttsx3).

``VoiceAlertManager`` decides *whether* to speak (priority, cooldown,
duplicate prevention) and speaks from a background thread so the video loop
never blocks. The main loop may call ``request`` every frame; almost all
calls return False immediately.

Rules:
  * the same alert is not repeated until its cooldown has passed;
  * a lower-priority alert right after a higher-priority one is suppressed;
  * a higher-priority alert (escalation) speaks immediately;
  * when the driver returns to normal (``request(None)``) the state resets so
    the next problem is announced right away;
  * a text-to-speech failure is logged and never stops detection.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from enum import Enum

from app.config import CONFIG, AppConfig
from app.drowsiness import OverallStatus

logger = logging.getLogger(__name__)


class AlertType(Enum):
    # (priority, spoken message); higher priority wins.
    LOOK_STRAIGHT = (0, "Please look straight.")
    DROWSINESS_WARNING = (1, "Hey! Wake up! Please stay alert.")
    PHONE_WARNING = (2, "Please put the phone down.")
    PHONE_CRITICAL = (3, "Stop using your phone while driving.")
    DROWSINESS_CRITICAL = (4, "Wake up! You appear to be falling asleep.")
    COMBINED_CRITICAL = (5, "Warning! You appear drowsy and are using your phone. Please stop the vehicle safely.")

    @property
    def priority(self) -> int:
        return self.value[0]

    @property
    def message(self) -> str:
        return self.value[1]


def alert_for_status(status: OverallStatus, phone_duration: float, config: AppConfig = CONFIG) -> AlertType | None:
    """Map the overall status to the alert that should be active, if any."""
    if status == OverallStatus.CRITICAL_DROWSINESS_PHONE:
        return AlertType.COMBINED_CRITICAL
    if status == OverallStatus.CRITICAL_DROWSINESS:
        return AlertType.DROWSINESS_CRITICAL
    if status == OverallStatus.PHONE_USE:
        if phone_duration >= config.PHONE_CRITICAL_SECONDS:
            return AlertType.PHONE_CRITICAL
        return AlertType.PHONE_WARNING
    if status == OverallStatus.DROWSINESS_WARNING:
        return AlertType.DROWSINESS_WARNING
    if status == OverallStatus.LOOKING_AWAY:
        return AlertType.LOOK_STRAIGHT
    return None


class VoiceAlertManager:
    def __init__(
        self,
        config: AppConfig = CONFIG,
        speak_fn: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        start_worker: bool = True,
    ) -> None:
        self._cfg = config
        self._speak_fn = speak_fn or self._speak_pyttsx3
        self._clock = clock
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=3)
        self._last_alert: AlertType | None = None
        self._last_time = 0.0
        self._speaking = False
        self.spoken_count = 0
        self._thread: threading.Thread | None = None
        if start_worker:
            self._thread = threading.Thread(target=self._worker, name="voice-alerts", daemon=True)
            self._thread.start()

    # ---- public API ---------------------------------------------------------
    def request(self, alert: AlertType | None) -> bool:
        """Ask for ``alert`` to be spoken. Returns True if it was queued."""
        if alert is None:
            self._last_alert = None  # driver is fine again: reset alert state
            return False

        now = self._clock()
        if self._last_alert is not None:
            elapsed = now - self._last_time
            cooldown = self._cooldown(alert)
            if alert == self._last_alert and elapsed < cooldown:
                return False  # duplicate within cooldown
            if alert.priority < self._last_alert.priority and elapsed < cooldown:
                return False  # do not downgrade right after a bigger alert
        if self._speaking and self._last_alert is not None and alert.priority <= self._last_alert.priority:
            return False  # still talking; only escalations may queue

        try:
            self._queue.put_nowait(alert.message)
        except queue.Full:
            return False
        self._last_alert, self._last_time = alert, now
        return True

    def close(self) -> None:
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)

    # ---- internals ----------------------------------------------------------
    def _cooldown(self, alert: AlertType) -> float:
        if alert in (AlertType.PHONE_WARNING, AlertType.PHONE_CRITICAL):
            return self._cfg.PHONE_ALERT_COOLDOWN
        return self._cfg.ALERT_COOLDOWN_SECONDS

    def _worker(self) -> None:
        while True:
            message = self._queue.get()
            if message is None:
                return
            self._speaking = True
            try:
                self._speak_fn(message)
                self.spoken_count += 1
            except Exception:
                logger.exception("Voice alert failed; visual alerts continue")
            finally:
                self._speaking = False

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            import pyttsx3  # imported here so tests and the UI work without audio

            # A fresh engine per message: a reused pyttsx3 engine often speaks
            # only the first time (known pyttsx3 behaviour).
            engine = pyttsx3.init()
            engine.setProperty("rate", self._cfg.VOICE_RATE)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            logger.exception("Text-to-speech failed; continuing without voice")

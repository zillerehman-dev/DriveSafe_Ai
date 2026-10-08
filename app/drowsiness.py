"""Drowsiness and risk decision engine (pure logic, no CV libraries).

Receives already-computed features and decides the driver's state. It never
touches MediaPipe, YOLO or the camera, so it is easy to unit test with
synthetic values and explicit timestamps.

How a decision is made each update:
  1. Each signal (eye closure, yawn, head nod, blink rate) gives a 0..1 level.
  2. Levels are multiplied by configurable weights and summed (max 100).
  3. The score rises immediately but decays gradually, so one good frame does
     not cancel a real warning.
  4. The score maps to a state with hysteresis and a short hold time, so the
     state does not flicker.
  5. Driver state + phone state give one overall status.

IMPORTANT: the risk score is an engineering indicator. It is not a medical
diagnosis, proof of sleep, or a scientifically validated safety measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

from app.config import CONFIG, AppConfig
from app.logger import EventType


class DriverState(IntEnum):
    AWAKE = 0
    WARNING = 1
    DROWSY = 2
    CRITICAL = 3


class PhoneState(str, Enum):
    NO_PHONE = "NO_PHONE"
    PHONE_DETECTED = "PHONE_DETECTED"
    PHONE_USE_CONFIRMED = "PHONE_USE_CONFIRMED"

    @classmethod
    def from_flags(cls, detected: bool, confirmed: bool) -> "PhoneState":
        if confirmed:
            return cls.PHONE_USE_CONFIRMED
        return cls.PHONE_DETECTED if detected else cls.NO_PHONE


class OverallStatus(str, Enum):
    """Combined result, listed from lowest to highest priority."""

    AWAKE = "AWAKE"
    LOOKING_AWAY = "LOOKING_AWAY"
    DROWSINESS_WARNING = "DROWSINESS_WARNING"
    PHONE_USE = "PHONE_USE"
    CRITICAL_DROWSINESS = "CRITICAL_DROWSINESS"
    CRITICAL_DROWSINESS_PHONE = "CRITICAL_DROWSINESS_PHONE"


@dataclass(frozen=True)
class DrowsinessInput:
    face_detected: bool
    ear: float
    mar: float
    eye_closure_duration: float
    blink_rate: float
    pitch: float
    phone_state: PhoneState = PhoneState.NO_PHONE
    yaw: float = 0.0


@dataclass(frozen=True)
class DrowsinessResult:
    state: DriverState
    risk_score: float
    reason: str
    eye_closure_duration: float
    is_yawning: bool
    is_head_nodding: bool
    phone_state: PhoneState
    status: OverallStatus
    risk_level: str  # LOW / MODERATE / HIGH / CRITICAL
    phone_duration: float  # seconds of continuously confirmed phone use
    new_events: tuple[EventType, ...] = ()  # events that started on this update
    is_looking_away: bool = False


def combine_status(state: DriverState, phone_state: PhoneState, looking_away: bool = False) -> OverallStatus:
    """Priority: drowsy+phone > critical drowsiness > phone use > warning > awake."""
    phone_confirmed = phone_state == PhoneState.PHONE_USE_CONFIRMED
    if phone_confirmed and state >= DriverState.DROWSY:
        return OverallStatus.CRITICAL_DROWSINESS_PHONE
    if state == DriverState.CRITICAL:
        return OverallStatus.CRITICAL_DROWSINESS
    if phone_confirmed:
        return OverallStatus.PHONE_USE
    if state >= DriverState.WARNING:
        return OverallStatus.DROWSINESS_WARNING
    if looking_away:
        return OverallStatus.LOOKING_AWAY
    return OverallStatus.AWAKE


class DrowsinessDetector:
    def __init__(self, config: AppConfig = CONFIG) -> None:
        self._cfg = config
        self._risk = 0.0
        self._state = DriverState.AWAKE
        self._pending: DriverState | None = None
        self._pending_since = 0.0
        self._last_time: float | None = None
        self._yawn_since: float | None = None
        self._nod_since: float | None = None
        self._away_since: float | None = None
        self._was_away = False
        self._phone_since: float | None = None
        self._was_yawning = False
        self._was_nodding = False
        self._last_status = OverallStatus.AWAKE
        self._last_phone_state = PhoneState.NO_PHONE

    # ---- public API ---------------------------------------------------------
    def update(self, features: DrowsinessInput, now: float) -> DrowsinessResult:
        dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)
        self._last_time = now
        cfg = self._cfg

        is_yawning = self._held(features.face_detected and features.mar >= cfg.MAR_THRESHOLD,
                                "_yawn_since", now, cfg.YAWN_MIN_SECONDS)
        is_nodding = self._held(features.face_detected and features.pitch >= cfg.HEAD_NOD_PITCH_DEGREES,
                                "_nod_since", now, cfg.HEAD_NOD_MIN_SECONDS)

        away_now = (features.face_detected and abs(features.yaw) >= cfg.LOOK_AWAY_YAW_DEGREES) or (
            not features.face_detected and cfg.LOOK_AWAY_COUNTS_NO_FACE
        )
        is_away = self._held(away_now, "_away_since", now, cfg.LOOK_AWAY_SECONDS)

        closure = features.eye_closure_duration if features.face_detected else 0.0
        eye_level = self._eye_level(closure)
        blink_level = 1.0 if features.face_detected and features.blink_rate >= cfg.BLINK_RATE_HIGH_PER_MIN else 0.0

        target = (
            eye_level * cfg.WEIGHT_EYE_CLOSURE
            + float(is_yawning) * cfg.WEIGHT_YAWN
            + float(is_nodding) * cfg.WEIGHT_HEAD_NOD
            + blink_level * cfg.WEIGHT_BLINK_ABNORMAL
        )
        phone_confirmed = features.phone_state == PhoneState.PHONE_USE_CONFIRMED
        if phone_confirmed and target > 0:
            target += cfg.WEIGHT_PHONE_COMBINED  # distraction compounds fatigue

        target = min(100.0, target)
        self._risk = max(target, self._risk - cfg.RISK_DECAY_PER_SECOND * dt)
        self._risk = min(100.0, max(0.0, self._risk))

        self._state = self._next_state(self._risk, now)
        self._phone_since = self._track_phone(phone_confirmed, now)
        phone_duration = 0.0 if self._phone_since is None else now - self._phone_since
        status = combine_status(self._state, features.phone_state, is_away)

        events = self._collect_events(is_yawning, is_nodding, status, features.phone_state, is_away)
        return DrowsinessResult(
            state=self._state,
            risk_score=self._risk,
            reason=self._reason(features, closure, is_yawning, is_nodding, blink_level, is_away),
            eye_closure_duration=closure,
            is_yawning=is_yawning,
            is_head_nodding=is_nodding,
            phone_state=features.phone_state,
            status=status,
            risk_level=self._risk_level(self._risk),
            phone_duration=phone_duration,
            new_events=events,
            is_looking_away=is_away,
        )

    # ---- internals ----------------------------------------------------------
    def _eye_level(self, closure: float) -> float:
        """0 for blinks, 0.5 at the warning duration, 1.0 at the critical one."""
        cfg = self._cfg
        if closure <= cfg.BLINK_MAX_SECONDS:
            return 0.0
        if closure < cfg.EYE_CLOSURE_WARNING_SECONDS:
            span = cfg.EYE_CLOSURE_WARNING_SECONDS - cfg.BLINK_MAX_SECONDS
            return 0.5 * (closure - cfg.BLINK_MAX_SECONDS) / span
        if closure < cfg.EYE_CLOSURE_CRITICAL_SECONDS:
            span = cfg.EYE_CLOSURE_CRITICAL_SECONDS - cfg.EYE_CLOSURE_WARNING_SECONDS
            return 0.5 + 0.5 * (closure - cfg.EYE_CLOSURE_WARNING_SECONDS) / span
        return 1.0

    def _held(self, active: bool, attr: str, now: float, minimum: float) -> bool:
        """True once ``active`` has been continuously true for ``minimum`` seconds."""
        if not active:
            setattr(self, attr, None)
            return False
        since = getattr(self, attr)
        if since is None:
            setattr(self, attr, now)
            since = now
        return now - since >= minimum

    def _thresholds(self) -> dict[DriverState, float]:
        cfg = self._cfg
        return {
            DriverState.WARNING: cfg.DROWSINESS_RISK_THRESHOLD,
            DriverState.DROWSY: cfg.DROWSY_RISK_THRESHOLD,
            DriverState.CRITICAL: cfg.CRITICAL_RISK_THRESHOLD,
        }

    def _next_state(self, risk: float, now: float) -> DriverState:
        thresholds = self._thresholds()
        # Raw candidate with no memory of the current state.
        candidate = DriverState.AWAKE
        for state, threshold in thresholds.items():
            if risk >= threshold:
                candidate = state

        current = self._state
        if candidate > current:  # going up: must persist for a short hold time
            if self._pending is None:
                self._pending_since = now  # timer keeps running while the candidate climbs
            self._pending = candidate
            if now - self._pending_since >= self._cfg.STATE_MIN_HOLD_SECONDS:
                self._pending = None
                return DriverState(current + 1)  # one step at a time, so no state is skipped
            return current

        self._pending = None
        # Going down: score must fall a margin below the current state's entry
        # threshold, otherwise stay. Prevents flicker around a threshold.
        while current > DriverState.AWAKE and risk < thresholds[current] - self._cfg.STATE_HYSTERESIS_MARGIN:
            current = DriverState(current - 1)
        return current

    def _track_phone(self, confirmed: bool, now: float) -> float | None:
        if not confirmed:
            return None
        return now if self._phone_since is None else self._phone_since

    def _risk_level(self, risk: float) -> str:
        if risk >= self._cfg.CRITICAL_RISK_THRESHOLD:
            return "CRITICAL"
        if risk >= self._cfg.DROWSY_RISK_THRESHOLD:
            return "HIGH"
        if risk >= self._cfg.DROWSINESS_RISK_THRESHOLD:
            return "MODERATE"
        return "LOW"

    def _collect_events(
        self, yawning: bool, nodding: bool, status: OverallStatus, phone_state: PhoneState, looking_away: bool = False
    ) -> tuple[EventType, ...]:
        events: list[EventType] = []
        if looking_away and not self._was_away:
            events.append(EventType.LOOK_AWAY)
        if yawning and not self._was_yawning:
            events.append(EventType.YAWN)
        if nodding and not self._was_nodding:
            events.append(EventType.HEAD_NOD)
        if status == OverallStatus.DROWSINESS_WARNING and self._last_status in (
            OverallStatus.AWAKE,
            OverallStatus.PHONE_USE,
        ):
            events.append(EventType.DROWSINESS_WARNING)
        if status == OverallStatus.CRITICAL_DROWSINESS and self._last_status != status:
            events.append(EventType.DROWSINESS_CRITICAL)
        if status == OverallStatus.CRITICAL_DROWSINESS_PHONE and self._last_status != status:
            events.append(EventType.COMBINED_CRITICAL)
        if phone_state == PhoneState.PHONE_DETECTED and self._last_phone_state == PhoneState.NO_PHONE:
            events.append(EventType.PHONE_DETECTED)
        if phone_state == PhoneState.PHONE_USE_CONFIRMED and self._last_phone_state != phone_state:
            events.append(EventType.PHONE_USE_CONFIRMED)

        self._was_yawning, self._was_nodding, self._was_away = yawning, nodding, looking_away
        self._last_status, self._last_phone_state = status, phone_state
        return tuple(events)

    @staticmethod
    def _reason(
        features: DrowsinessInput, closure: float, yawning: bool, nodding: bool, blink_level: float, away: bool = False
    ) -> str:
        if not features.face_detected:
            return "No driver face detected"
        parts = []
        if closure > 0.4:
            parts.append(f"eyes closed {closure:.1f}s")
        if yawning:
            parts.append("yawning")
        if nodding:
            parts.append("head nodding")
        if blink_level:
            parts.append("frequent blinking")
        if away:
            parts.append("looking away")
        if features.phone_state == PhoneState.PHONE_USE_CONFIRMED:
            parts.append("phone use")
        return ", ".join(parts).capitalize() if parts else "Normal"

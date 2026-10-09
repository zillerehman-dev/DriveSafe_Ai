from __future__ import annotations

import csv

import numpy as np
import pytest

from app.alerts import AlertType, VoiceAlertManager, alert_for_status
from app.config import CONFIG, AppConfig
from app.drowsiness import (
    DriverState,
    DrowsinessDetector,
    DrowsinessInput,
    OverallStatus,
    PhoneState,
    combine_status,
)
from app.logger import COLUMNS, EventLogger, EventType
from app.phone_detector import PhoneConfirmation, PhoneDetector
from app.vision import (
    HEAD_POSE_LANDMARKS,
    HEAD_MODEL_POINTS,
    BlinkTracker,
    calculate_eye_aspect_ratio,
    calculate_mouth_aspect_ratio,
    estimate_head_pose,
)

STEP = 0.1  # simulated seconds between frames


def features(closure=0.0, phone=PhoneState.NO_PHONE, ear=0.30, mar=0.2, pitch=0.0, blink_rate=15.0, face=True):
    return DrowsinessInput(face, ear, mar, closure, blink_rate, pitch, phone)


def simulate(detector, seconds, make, start=0.0):
    """Feed ``make(t_since_start)`` features every STEP seconds; return results."""
    results, t = [], 0.0
    while t <= seconds + 1e-9:
        results.append(detector.update(make(t), start + t))
        t += STEP
    return results


# --- 1-2. Normal behaviour and blinks must not alarm ---------------------------
def test_normal_ear_does_not_trigger_drowsiness():
    results = simulate(DrowsinessDetector(), 5, lambda t: features(ear=0.31))
    assert all(r.state == DriverState.AWAKE for r in results)
    assert all(r.risk_score == 0 for r in results)


def test_short_eye_closure_is_not_critical():
    # A 0.3 s closure (a normal blink) then eyes open again.
    results = simulate(DrowsinessDetector(), 3, lambda t: features(closure=t if t < 0.35 else 0.0, ear=0.1 if t < 0.35 else 0.3))
    assert all(r.state != DriverState.CRITICAL for r in results)
    assert all(r.state == DriverState.AWAKE for r in results)


# --- 3-4. Prolonged closure ----------------------------------------------------
def test_prolonged_eye_closure_triggers_warning_or_drowsy():
    results = simulate(DrowsinessDetector(), 1.8, lambda t: features(closure=t, ear=0.1))
    assert results[-1].state in (DriverState.WARNING, DriverState.DROWSY)
    assert results[-1].status == OverallStatus.DROWSINESS_WARNING
    assert EventType.DROWSINESS_WARNING in [e for r in results for e in r.new_events]


def test_critical_prolonged_closure_triggers_critical():
    results = simulate(DrowsinessDetector(), 3.0, lambda t: features(closure=t, ear=0.1))
    # The driver passes through every state on the way up (no skipped DROWSY).
    assert {DriverState.WARNING, DriverState.DROWSY, DriverState.CRITICAL} <= {r.state for r in results}
    assert results[-1].state == DriverState.CRITICAL
    assert results[-1].status == OverallStatus.CRITICAL_DROWSINESS
    assert EventType.DROWSINESS_CRITICAL in [e for r in results for e in r.new_events]


def test_single_noisy_frame_does_not_cause_critical():
    detector = DrowsinessDetector()
    detector.update(features(), 0.0)
    noisy = detector.update(features(closure=5.0, ear=0.0), 0.1)  # one absurd frame
    assert noisy.state != DriverState.CRITICAL  # needs to persist for the hold time


# --- 5-6. Phone and combined states -------------------------------------------
def test_phone_alone_produces_phone_state():
    results = simulate(DrowsinessDetector(), 2, lambda t: features(phone=PhoneState.PHONE_USE_CONFIRMED))
    assert results[-1].status == OverallStatus.PHONE_USE
    assert results[-1].state == DriverState.AWAKE
    assert results[-1].phone_duration > 1.5
    assert EventType.PHONE_USE_CONFIRMED in [e for r in results for e in r.new_events]


def test_phone_detected_but_not_confirmed_is_not_an_alert():
    results = simulate(DrowsinessDetector(), 2, lambda t: features(phone=PhoneState.PHONE_DETECTED))
    assert results[-1].status == OverallStatus.AWAKE
    assert EventType.PHONE_DETECTED in [e for r in results for e in r.new_events]


def test_drowsiness_plus_phone_produces_combined_critical():
    results = simulate(
        DrowsinessDetector(), 2.5, lambda t: features(closure=t, ear=0.1, phone=PhoneState.PHONE_USE_CONFIRMED)
    )
    assert results[-1].status == OverallStatus.CRITICAL_DROWSINESS_PHONE
    assert EventType.COMBINED_CRITICAL in [e for r in results for e in r.new_events]


def test_status_priority_order():
    confirmed = PhoneState.PHONE_USE_CONFIRMED
    assert combine_status(DriverState.DROWSY, confirmed) == OverallStatus.CRITICAL_DROWSINESS_PHONE
    assert combine_status(DriverState.CRITICAL, PhoneState.NO_PHONE) == OverallStatus.CRITICAL_DROWSINESS
    assert combine_status(DriverState.WARNING, confirmed) == OverallStatus.PHONE_USE
    assert combine_status(DriverState.WARNING, PhoneState.NO_PHONE) == OverallStatus.DROWSINESS_WARNING
    assert combine_status(DriverState.AWAKE, PhoneState.PHONE_DETECTED) == OverallStatus.AWAKE


# --- 7. Risk bounds ---------------
def test_risk_score_stays_between_0_and_100():
    detector = DrowsinessDetector()
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(500):
        t += float(rng.uniform(0, 0.5))
        result = detector.update(
            features(
                closure=float(rng.uniform(0, 50)),
                mar=float(rng.uniform(0, 2)),
                pitch=float(rng.uniform(-90, 90)),
                blink_rate=float(rng.uniform(0, 100)),
                phone=PhoneState.PHONE_USE_CONFIRMED,
                face=bool(rng.integers(0, 2)),
            ),
            t,
        )
        assert 0.0 <= result.risk_score <= 100.0


# --- 8. Alert cooldown ------------------------
class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make_manager(clock):
    return VoiceAlertManager(CONFIG, speak_fn=lambda text: None, clock=clock, start_worker=False)


def test_alert_cooldown_prevents_repeated_alerts():
    clock = FakeClock()
    manager = make_manager(clock)
    assert manager.request(AlertType.DROWSINESS_WARNING) is True
    clock.now += 1.0
    assert manager.request(AlertType.DROWSINESS_WARNING) is False  # inside cooldown
    clock.now += CONFIG.ALERT_COOLDOWN_SECONDS
    assert manager.request(AlertType.DROWSINESS_WARNING) is True  # cooldown over


def test_escalation_speaks_immediately_but_downgrade_is_suppressed():
    clock = FakeClock()
    manager = make_manager(clock)
    assert manager.request(AlertType.DROWSINESS_WARNING)
    clock.now += 0.5
    assert manager.request(AlertType.COMBINED_CRITICAL)  # escalation
    clock.now += 0.5
    assert not manager.request(AlertType.DROWSINESS_WARNING)  # lower priority, too soon


def test_alert_state_resets_when_driver_recovers():
    clock = FakeClock()
    manager = make_manager(clock)
    assert manager.request(AlertType.DROWSINESS_CRITICAL)
    manager.request(None)  # back to AWAKE
    clock.now += 0.5
    assert manager.request(AlertType.DROWSINESS_CRITICAL)  # announced again at once


def test_worker_speaks_without_blocking_and_survives_tts_errors():
    spoken = []

    def flaky(text):
        spoken.append(text)
        raise RuntimeError("audio device missing")

    manager = VoiceAlertManager(CONFIG, speak_fn=flaky, clock=FakeClock())
    assert manager.request(AlertType.PHONE_WARNING)
    manager.close()  # waits for the worker; must not raise
    assert spoken == [AlertType.PHONE_WARNING.message]


def test_alert_for_status_mapping():
    assert alert_for_status(OverallStatus.AWAKE, 0) is None
    assert alert_for_status(OverallStatus.PHONE_USE, 1.0) == AlertType.PHONE_WARNING
    assert alert_for_status(OverallStatus.PHONE_USE, CONFIG.PHONE_CRITICAL_SECONDS + 1) == AlertType.PHONE_CRITICAL
    assert alert_for_status(OverallStatus.CRITICAL_DROWSINESS_PHONE, 0) == AlertType.COMBINED_CRITICAL


# --- 9. State transitions ------------------------------------------------------
def test_state_recovers_gradually_after_eyes_reopen():
    detector = DrowsinessDetector()
    simulate(detector, 3.0, lambda t: features(closure=t, ear=0.1))
    assert detector.update(features(), 3.1).state == DriverState.CRITICAL  # decay is gradual
    recovery = simulate(detector, 6.0, lambda t: features(), start=3.2)
    states = [r.state for r in recovery]
    assert states[-1] == DriverState.AWAKE
    assert states == sorted(states, reverse=True)  # only ever steps down, no flicker


def test_hysteresis_prevents_flicker_near_threshold():
    cfg = AppConfig(RISK_DECAY_PER_SECOND=170.0)  # score follows the signal quickly
    detector = DrowsinessDetector(cfg)
    simulate(detector, 1.0, lambda t: features(closure=1.0))  # steady 50 points
    assert detector.update(features(closure=1.0), 1.2).state == DriverState.WARNING
    # ~33 points: below the WARNING threshold (40) but inside the hysteresis
    # margin (40 - 8 = 32), so the state must hold.
    dip = detector.update(features(closure=0.8), 1.3)
    assert 32 < dip.risk_score < 40
    assert dip.state == DriverState.WARNING
    # A clearly lower score finally releases the state.
    assert detector.update(features(closure=0.6), 1.4).state == DriverState.AWAKE


def test_no_face_is_not_treated_as_drowsy():
    results = simulate(DrowsinessDetector(), 4, lambda t: features(face=False))
    assert all(r.state == DriverState.AWAKE for r in results)
    assert results[-1].reason == "No driver face detected"


def test_yawn_and_head_nod_need_persistence():
    detector = DrowsinessDetector()
    brief = detector.update(features(mar=0.9, pitch=30), 0.0)
    assert not brief.is_yawning and not brief.is_head_nodding
    results = simulate(detector, 2.0, lambda t: features(mar=0.9, pitch=30), start=0.1)
    assert results[-1].is_yawning and results[-1].is_head_nodding
    events = [e for r in results for e in r.new_events]
    assert events.count(EventType.YAWN) == 1 and events.count(EventType.HEAD_NOD) == 1


# --- Vision maths (pure functions) ---------------------------------------------
OPEN_EYE = np.array([[0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]], dtype=float)


def test_ear_open_closed_and_invalid():
    assert calculate_eye_aspect_ratio(OPEN_EYE) == pytest.approx(2 / 3)
    closed = OPEN_EYE.copy()
    closed[:, 1] = 0
    assert calculate_eye_aspect_ratio(closed) == pytest.approx(0.0)
    degenerate = np.zeros((6, 2))
    assert calculate_eye_aspect_ratio(degenerate) == 0.0  # zero width: no crash
    assert calculate_eye_aspect_ratio(np.array([[np.nan] * 2] * 6)) == 0.0
    assert calculate_eye_aspect_ratio(np.zeros((3, 2))) == 0.0


def test_mar_open_vs_closed_mouth():
    points = np.zeros((468, 2))
    points[78], points[308] = (0, 0), (10, 0)
    for (a, b), x in zip(((81, 178), (13, 14), (311, 402)), (2.5, 5, 7.5)):
        points[a], points[b] = (x, -3), (x, 3)
    assert calculate_mouth_aspect_ratio(points) == pytest.approx(0.6)
    for a, b in ((81, 178), (13, 14), (311, 402)):
        points[a], points[b] = points[a] * [1, 0], points[b] * [1, 0]
    assert calculate_mouth_aspect_ratio(points) == pytest.approx(0.0)
    assert calculate_mouth_aspect_ratio(np.zeros((468, 2))) == 0.0
    assert calculate_mouth_aspect_ratio(np.zeros((5, 2))) == 0.0  # too few points


@pytest.mark.parametrize("angles", [(0, 0, 0), (20, 0, 0), (-15, 0, 0), (0, 25, 0), (10, -15, 8)])
def test_head_pose_recovers_known_rotation(angles):
    import cv2

    pitch, yaw, roll = np.radians(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    width, height = 640, 480
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1.0]])
    rvec, _ = cv2.Rodrigues(rz @ ry @ rx)
    projected, _ = cv2.projectPoints(HEAD_MODEL_POINTS, rvec, np.array([0, 0, 1500.0]), camera, np.zeros(4))
    landmarks = np.zeros((468, 2))
    for i, index in enumerate(HEAD_POSE_LANDMARKS):
        landmarks[index] = projected[i, 0]
    assert estimate_head_pose(landmarks, width, height) == pytest.approx(angles, abs=0.5)


def test_head_pose_invalid_landmarks_return_zeros():
    assert estimate_head_pose(np.full((468, 2), np.nan), 640, 480) == (0.0, 0.0, 0.0)
    assert estimate_head_pose(np.zeros((3, 2)), 640, 480) == (0.0, 0.0, 0.0)


def test_blink_tracker_counts_blinks_but_not_long_closures():
    tracker = BlinkTracker()
    closed, open_ = 0.1, 0.3
    # A 0.2 s blink.
    assert tracker.update(open_, 0.0) == (False, 0.0)
    assert tracker.update(closed, 1.0)[0] is True
    assert tracker.update(closed, 1.15)[1] == pytest.approx(0.15)
    tracker.update(open_, 1.2)
    assert tracker.blink_count == 1
    # A 1.5 s closure is a closure, not a blink.
    tracker.update(closed, 2.0)
    assert tracker.update(closed, 3.5)[1] == pytest.approx(1.5)
    tracker.update(open_, 3.6)
    assert tracker.blink_count == 1
    assert tracker.blink_rate(3.6) == pytest.approx(1.0)  # 1 blink in a 60 s window


# --- Phone detection -----------------------------------------------------------
def test_phone_confirmation_needs_consecutive_hits_and_tolerates_a_miss():
    conf = PhoneConfirmation(required_hits=3, max_misses=1)
    assert [conf.update(True), conf.update(True)] == [False, False]
    assert conf.update(False) is False  # one miss tolerated
    assert conf.update(True) is True  # third hit
    conf2 = PhoneConfirmation(3, 1)
    conf2.update(True)
    conf2.update(False)
    conf2.update(False)  # second miss resets the streak
    assert conf2.hits == 0


class FakeBox:
    def __init__(self, confidence, xyxy):
        self.conf = [np.float32(confidence)]
        self.xyxy = [np.array(xyxy)]


class FakeModel:
    names = {0: "person", 67: "cell phone"}

    def __init__(self, boxes):
        self.boxes, self.calls = boxes, 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return [type("R", (), {"boxes": self.boxes})()]


def test_phone_detector_confirms_after_frames_and_skips_yolo_runs():
    cfg = AppConfig(YOLO_FRAME_SKIP=2, PHONE_CONFIRMATION_FRAMES=3)
    model = FakeModel([FakeBox(0.9, [10, 10, 50, 80]), FakeBox(0.5, [0, 0, 5, 5])])
    detector = PhoneDetector(cfg, model=model)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    seen = [detector.detect(frame, now=i * 0.03) for i in range(6)]
    assert model.calls == 3  # every 2nd frame only
    assert model.kwargs["classes"] == [67]
    assert seen[0].detected and not seen[0].confirmed
    assert seen[0].bbox == (10, 10, 50, 80) and seen[0].confidence == pytest.approx(0.9)  # best box
    assert seen[-1].confirmed


def test_phone_detector_no_phone_and_invalid_frames():
    detector = PhoneDetector(AppConfig(), model=FakeModel([]))
    assert detector.detect(np.zeros((120, 160, 3), dtype=np.uint8), now=0.0).detected is False
    assert detector.detect(None, now=0.1).detected is False  # invalid frame: skipped safely


def test_phone_box_expires_after_persistence_window():
    cfg = AppConfig(YOLO_FRAME_SKIP=1000)  # one YOLO run, then reuse
    detector = PhoneDetector(cfg, model=FakeModel([FakeBox(0.8, [1, 1, 9, 9])]))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert detector.detect(frame, now=0.0).detected
    assert detector.detect(frame, now=cfg.PHONE_PERSISTENCE_SECONDS / 2).detected
    assert not detector.detect(frame, now=cfg.PHONE_PERSISTENCE_SECONDS + 1).detected


def test_missing_yolo_model_gives_clear_error(tmp_path):
    cfg = AppConfig(YOLO_MODEL_PATH=tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="Phone detection model not found"):
        PhoneDetector(cfg)


# --- Logger --------------------------------------------------------------------
def test_event_logger_creates_file_and_appends_rows(tmp_path):
    path = tmp_path / "logs" / "events.csv"
    log = EventLogger(path=path)
    assert path.exists()
    kwargs = dict(risk_score=91.234, ear=0.1, mar=0.3, pitch=5, yaw=1, roll=2, phone_confidence=0.0, state="DROWSY")
    log.log_event(EventType.DROWSINESS_CRITICAL, **kwargs)
    log.log_event("YAWN", **kwargs)
    rows = list(csv.DictReader(path.open()))
    assert tuple(rows[0].keys()) == COLUMNS
    assert [r["event_type"] for r in rows] == ["DROWSINESS_CRITICAL", "YAWN"]
    assert rows[0]["risk_score"] == "91.2"
    assert log.counts["YAWN"] == 1
    EventLogger(path=path)  # reopening must not wipe or duplicate the header
    assert len(list(csv.DictReader(path.open()))) == 2


# --- Config sanity --------------------------------------------------------------
def test_config_thresholds_are_consistent():
    assert CONFIG.EYE_CLOSURE_WARNING_SECONDS < CONFIG.EYE_CLOSURE_CRITICAL_SECONDS
    assert CONFIG.DROWSINESS_RISK_THRESHOLD < CONFIG.DROWSY_RISK_THRESHOLD < CONFIG.CRITICAL_RISK_THRESHOLD <= 100
    assert CONFIG.BLINK_MAX_SECONDS < CONFIG.EYE_CLOSURE_WARNING_SECONDS
    assert CONFIG.WEIGHT_EYE_CLOSURE >= CONFIG.CRITICAL_RISK_THRESHOLD  # closure alone can reach CRITICAL


# --- Pipeline integration (fakes replace MediaPipe and YOLO) --------------------
class FakeVision:
    """Stands in for VisionProcessor: eyes closed from t=1 onwards."""

    def __init__(self, config):
        self.faces = 1

    def process_frame(self, frame, now):
        from app.vision import VisionResult

        closed = now >= 1.0
        points = np.random.default_rng(0).uniform(100, 500, (468, 2))
        return VisionResult(
            face_detected=True,
            face_count=self.faces,
            ear=0.1 if closed else 0.3,
            eye_closed=closed,
            eye_closure_duration=max(0.0, now - 1.0) if closed else 0.0,
            landmarks=points,
        )

    def close(self):
        pass


def test_pipeline_end_to_end_with_fakes(tmp_path, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "VisionProcessor", FakeVision)
    cfg = AppConfig(
        YOLO_MODEL_PATH=tmp_path / "missing.pt",
        LOG_FILE_PATH=tmp_path / "events.csv",
        YOLO_FRAME_SKIP=1,
        PHONE_CONFIRMATION_FRAMES=2,
    )
    pipeline = main.DriveSafePipeline(cfg, enable_voice=False)
    assert main.MSG_NO_PHONE_MODEL in pipeline.startup_warnings  # app still starts

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert pipeline.process(None) is None  # invalid frame is skipped
    analysis = None
    for i in range(40):
        analysis = pipeline.process(frame, now=i * 0.1)
        main.draw_overlay(analysis, fps=30.0)  # must not raise
    assert analysis.drowsiness.state == DriverState.CRITICAL
    assert analysis.frame.shape == frame.shape
    logged = {row["event_type"] for row in csv.DictReader(cfg.LOG_FILE_PATH.open())}
    assert {"DROWSINESS_WARNING", "DROWSINESS_CRITICAL"} <= logged

    pipeline.vision.faces = 2
    assert main.MSG_MULTI_FACE in pipeline.process(frame, now=4.1).messages


# --- Looking away -> "Please look straight" -------------------------------------
def test_looking_away_needs_five_seconds():
    detector = DrowsinessDetector()
    results = simulate(detector, 4.5, lambda t: DrowsinessInput(True, 0.3, 0.2, 0.0, 15.0, 0.0, yaw=40.0))
    assert not any(r.is_looking_away for r in results)  # 4.5 s: not yet
    later = simulate(detector, 1.5, lambda t: DrowsinessInput(True, 0.3, 0.2, 0.0, 15.0, 0.0, yaw=40.0), start=4.6)
    assert later[-1].is_looking_away and later[-1].status == OverallStatus.LOOKING_AWAY
    assert alert_for_status(later[-1].status, 0) == AlertType.LOOK_STRAIGHT
    assert AlertType.LOOK_STRAIGHT.message == "Please look straight."
    assert [e for r in later for e in r.new_events].count(EventType.LOOK_AWAY) == 1


def test_looking_away_works_for_left_and_resets_when_straight():
    detector = DrowsinessDetector()
    left = lambda t: DrowsinessInput(True, 0.3, 0.2, 0.0, 15.0, 0.0, yaw=-35.0)  # noqa: E731
    assert simulate(detector, 6.0, left)[-1].is_looking_away
    straight = detector.update(features(), 6.2)
    assert not straight.is_looking_away and straight.status == OverallStatus.AWAKE
    # The 5 s timer restarts after looking straight.
    assert not simulate(detector, 3.0, left, start=6.3)[-1].is_looking_away


def test_small_head_turn_is_not_looking_away():
    results = simulate(DrowsinessDetector(), 8.0, lambda t: DrowsinessInput(True, 0.3, 0.2, 0.0, 15.0, 0.0, yaw=15.0))
    assert not any(r.is_looking_away for r in results)


def test_face_lost_counts_as_looking_away_and_phone_has_priority():
    results = simulate(DrowsinessDetector(), 6.0, lambda t: features(face=False))
    assert results[-1].status == OverallStatus.LOOKING_AWAY
    assert combine_status(DriverState.AWAKE, PhoneState.PHONE_USE_CONFIRMED, True) == OverallStatus.PHONE_USE
    assert combine_status(DriverState.WARNING, PhoneState.NO_PHONE, True) == OverallStatus.DROWSINESS_WARNING

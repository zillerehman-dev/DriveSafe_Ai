"""Streamlit dashboard (demo / deployment interface).

Two modes:
  * Live webcam (local): reads the camera of the machine running Streamlit and
    updates the dashboard continuously. Use this when running locally.
  * Camera snapshot: analyses single photos taken in the browser. This is the
    cloud-friendly mode, but one photo cannot show eye closure over time, so
    temporal alerts do not build up. The local OpenCV app (python -m app.main)
    remains the primary real-time demo.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
import streamlit as st

from app.config import CONFIG
from app.drowsiness import DriverState, OverallStatus, PhoneState
from app.logger import EventType
from app.main import DriveSafePipeline, FrameAnalysis, draw_overlay

LIVE_MODE = "Live webcam (local)"
SNAPSHOT_MODE = "Camera snapshot (cloud-friendly)"
STATE_ICONS = {
    DriverState.AWAKE: "🟢 AWAKE",
    DriverState.WARNING: "🟠 WARNING",
    DriverState.DROWSY: "🔴 DROWSY",
    DriverState.CRITICAL: "🔴 CRITICAL",
}
DROWSINESS_EVENTS = (
    EventType.DROWSINESS_WARNING,
    EventType.DROWSINESS_CRITICAL,
    EventType.YAWN,
    EventType.HEAD_NOD,
    EventType.COMBINED_CRITICAL,
)
PHONE_EVENTS = (EventType.PHONE_DETECTED, EventType.PHONE_USE_CONFIRMED)
HISTORY_LENGTH = 300
METRIC_LABELS = {
    "status": "Status",
    "risk": "Drowsiness risk",
    "phone": "Phone",
    "phone_conf": "Phone confidence",
    "ear": "EAR",
    "mar": "MAR",
    "blinks": "Blinks",
    "closure": "Eyes closed",
    "session": "Session",
    "drowsy_events": "Drowsiness events",
    "phone_events": "Phone events",
    "fps": "FPS",
}

st.set_page_config(page_title="DriveSafe AI", page_icon="🚗", layout="wide")


def get_pipeline(enable_voice: bool) -> DriveSafePipeline | None:
    """One pipeline per browser session; rebuilt if the voice option changes."""
    state = st.session_state
    if state.get("pipeline") is not None and state.get("voice") == enable_voice:
        return state["pipeline"]
    if state.get("pipeline") is not None:
        state["pipeline"].close()
        state["pipeline"] = None
    try:
        state["pipeline"] = DriveSafePipeline(CONFIG, enable_voice=enable_voice)
        state["voice"] = enable_voice
        state["started"] = time.monotonic()
        state["history"] = deque(maxlen=HISTORY_LENGTH)
    except FileNotFoundError as error:
        st.error(str(error))
        return None
    return state["pipeline"]


def render(analysis: FrameAnalysis, pipeline: DriveSafePipeline, slots: dict, fps: float, chart: bool) -> None:
    result, vision, phone = analysis.drowsiness, analysis.vision, analysis.phone
    st.session_state["history"].append(result.risk_score)

    banner = slots["banner"]
    if result.status == OverallStatus.CRITICAL_DROWSINESS_PHONE:
        banner.error("🔴 CRITICAL: DROWSINESS + PHONE USE")
    elif result.status == OverallStatus.CRITICAL_DROWSINESS:
        banner.error("🔴 CRITICAL: DROWSINESS")
    elif result.status == OverallStatus.PHONE_USE:
        banner.warning("📱 PHONE USE DETECTED")
    elif result.status == OverallStatus.DROWSINESS_WARNING:
        banner.warning("🟠 Signs of fatigue: " + result.reason)
    elif result.status == OverallStatus.LOOKING_AWAY:
        banner.warning("👀 Please look straight")
    elif analysis.messages:
        banner.info(" ".join(analysis.messages))
    else:
        banner.success("🟢 Driver alert")

    phone_text = {
        PhoneState.NO_PHONE: "NOT DETECTED",
        PhoneState.PHONE_DETECTED: "DETECTED",
        PhoneState.PHONE_USE_CONFIRMED: "IN USE",
    }[result.phone_state]
    counts = pipeline.events.counts
    elapsed = int(time.monotonic() - st.session_state["started"])

    slots["status"].metric("Status", STATE_ICONS[result.state])
    slots["risk"].metric("Drowsiness risk", f"{result.risk_score:.0f}%", result.risk_level, delta_color="off")
    slots["phone"].metric("Phone", phone_text)
    slots["phone_conf"].metric("Phone confidence", f"{phone.confidence:.0%}")
    slots["ear"].metric("EAR", f"{vision.ear:.2f}")
    slots["mar"].metric("MAR", f"{vision.mar:.2f}")
    slots["blinks"].metric("Blinks", vision.blink_count)
    slots["closure"].metric("Eyes closed", f"{result.eye_closure_duration:.1f} s")
    slots["session"].metric("Session", f"{elapsed // 60:02d}:{elapsed % 60:02d}")
    slots["drowsy_events"].metric("Drowsiness events", sum(counts[e.value] for e in DROWSINESS_EVENTS))
    slots["phone_events"].metric("Phone events", sum(counts[e.value] for e in PHONE_EVENTS))
    slots["fps"].metric("FPS", f"{fps:.0f}")
    if chart:
        slots["chart"].line_chart(list(st.session_state["history"]), height=180)

    rgb = cv2.cvtColor(draw_overlay(analysis, fps), cv2.COLOR_BGR2RGB)
    slots["video"].image(rgb, channels="RGB", use_container_width=True)


def build_layout() -> dict:
    slots: dict = {"banner": st.empty()}
    video_col, metrics_col = st.columns([3, 2])
    slots["video"] = video_col.empty()
    with metrics_col:
        names = [
            ("status", "risk"),
            ("phone", "phone_conf"),
            ("ear", "mar"),
            ("blinks", "closure"),
            ("session", "fps"),
            ("drowsy_events", "phone_events"),
        ]
        for left, right in names:
            a, b = st.columns(2)
            slots[left], slots[right] = a.empty(), b.empty()
    st.subheader("Risk history")
    slots["chart"] = st.empty()
    for key, label in METRIC_LABELS.items():  # idle placeholders until frames arrive
        slots[key].metric(label, "—")
    return slots


def run_live(pipeline: DriveSafePipeline, slots: dict) -> None:
    capture = cv2.VideoCapture(CONFIG.CAMERA_INDEX)
    if not capture.isOpened():
        st.error("Unable to access camera.")
        return
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.FRAME_HEIGHT)
    fps, last, frame_no = 0.0, time.monotonic(), 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if CONFIG.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1.0 / max(now - last, 1e-6)
            last = now
            analysis = pipeline.process(frame, now)
            if analysis is not None:
                frame_no += 1
                render(analysis, pipeline, slots, fps, chart=frame_no % 15 == 0)
    finally:  # runs when Streamlit stops the script (toggle off / rerun)
        capture.release()


def main() -> None:
    st.title("DriveSafe AI")
    st.caption("Real-Time Driver Drowsiness & Distraction Detection")
    st.caption(
        "Driver monitoring prototype. The risk score is an engineering indicator, "
        "not a medical or safety-validated measurement."
    )

    with st.sidebar:
        mode = st.radio("Input", [LIVE_MODE, SNAPSHOT_MODE])
        voice = st.checkbox("Voice alerts (plays on the machine running Streamlit)", value=False)
        running = st.toggle("Start camera", value=False) if mode == LIVE_MODE else False
        st.caption("Cloud hosts usually cannot reach a local webcam or audio. Use the snapshot mode there.")

    pipeline = get_pipeline(voice and mode == LIVE_MODE)
    if pipeline is None:
        st.stop()
    for warning in pipeline.startup_warnings:
        st.warning(warning)

    slots = build_layout()
    if mode == LIVE_MODE:
        if running:
            run_live(pipeline, slots)
        else:
            slots["banner"].info("Toggle 'Start camera' in the sidebar to begin.")
        return

    photo = st.camera_input("Take a photo")
    if photo is not None:
        frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        analysis = pipeline.process(frame) if frame is not None else None
        if analysis is None:
            st.error("Could not read the photo.")
        else:
            st.caption("Single-frame analysis: eye closure over time and alerts need the live mode.")
            render(analysis, pipeline, slots, fps=0.0, chart=True)


main()

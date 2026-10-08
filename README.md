# DriveSafe AI

**Real-Time Driver Drowsiness & Distraction Detection**

## Overview

DriveSafe AI is an AI-assisted driver monitoring **prototype**. It watches a webcam for signs of drowsiness (long eye closure, yawning, head nodding) and distraction (mobile phone use), and responds with on-screen warnings, a risk score, voice alerts and an event log.

It is a learning and portfolio project. It is **not** an automotive-certified safety system, and it does not prevent accidents.

## Features

- Real-time drowsiness detection with time-based rules (not single-frame decisions)
- Eye closure detection: normal blinks are counted but never trigger an alarm
- Yawning detection (mouth aspect ratio)
- Head pose monitoring (pitch, yaw, roll)
- Phone detection with temporal confirmation
- Combined risk detection: drowsiness + phone use gives a CRITICAL state
- Voice alerts with priority, cooldown and duplicate prevention
- Event logging to CSV (numbers only)
- Risk score from 0 to 100 (engineering indicator)
- Streamlit dashboard

## Computer Vision

- **MediaPipe Face Landmarker** finds 468 facial landmarks per frame. If several faces are visible, the largest one (closest to the camera) is treated as the driver.
- **EAR (Eye Aspect Ratio)**: `(|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`. Around 0.3 for an open eye, near 0 for a closed one. Averaged over both eyes.
- **MAR (Mouth Aspect Ratio)**: mean inner-lip opening divided by mouth width. A large value held for about a second counts as a yawn.
- **Head pose**: OpenCV `solvePnP` fits six facial landmarks to a generic 3D face model. Pitch is positive when the head tilts down. Head pose is supporting evidence only.
- **Blink tracking**: a closure of 0.05 to 0.4 s counts as a blink. Longer closures are reported as continuous eye-closure time, which the drowsiness engine judges.

## Phone Detection

A pretrained Ultralytics YOLO model detects the COCO "cell phone" class. The model is loaded once and runs every `YOLO_FRAME_SKIP` frames (default 5) on a reduced image size to protect the frame rate. A detection stays visible for a short time between runs.

A visible phone is not treated as phone *use*. Phone use is **confirmed** only after `PHONE_CONFIRMATION_FRAMES` consecutive YOLO detections (a couple of missed runs are tolerated). The phone state is `NO_PHONE`, `PHONE_DETECTED` or `PHONE_USE_CONFIRMED`.

## Alert System

Overall status, highest priority first:

1. `CRITICAL_DROWSINESS_PHONE` ("CRITICAL: DROWSINESS + PHONE USE")
2. `CRITICAL_DROWSINESS`
3. `PHONE_USE`
4. `DROWSINESS_WARNING`
5. `AWAKE`

Voice alerts (pyttsx3) are spoken from a background thread, so video never freezes. The same alert is not repeated until its cooldown (`ALERT_COOLDOWN_SECONDS`, `PHONE_ALERT_COOLDOWN`) has passed, a lower-priority alert right after a higher one is suppressed, an escalation speaks immediately, and the state resets when the driver recovers. If text-to-speech is unavailable, the error is logged and visual alerts continue.

## Risk Score

The risk score (0 to 100) adds weighted signals: eye closure (strongest), yawning, head nodding, abnormal blink rate, and an extra amount when confirmed phone use happens together with drowsiness signs. It rises immediately and decays gradually. States use hysteresis and a short hold time so they do not flicker: `AWAKE`, `WARNING` (from 40), `DROWSY` (from 70), `CRITICAL` (from 85).

**The score is an engineering indicator.** It is not a medical diagnosis, not proof that someone is asleep, and not a scientifically validated safety measurement. All weights and thresholds are starting points in `app/config.py` and can be tuned.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Model Files

Model files are not committed. Download them into `models/`:

```bash
# Phone detection (Ultralytics YOLOv8 nano, COCO classes)
curl -L -o models/yolo_model.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# Face landmarks (MediaPipe Face Landmarker)
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

On Windows PowerShell use `curl.exe -L -o ...`, or download the files in a browser and save them under those names.

Check each model's license before redistributing it (Ultralytics models are AGPL-3.0 or commercial; MediaPipe models are Apache-2.0). If a file is missing, the app prints a clear error instead of crashing.

## Run Local Application

```bash
python -m app.main
```

Press `q` or `Esc` to quit. This is the primary real-time demo.

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

- **Live webcam (local)**: reads the camera of the machine running Streamlit. Use this when running locally.
- **Camera snapshot**: analyses a single browser photo. It is the cloud-friendly mode, but one photo cannot show eye closure over time, so temporal alerts do not build up.

Webcam behaviour on cloud hosting differs from local use: a hosted server cannot reach your webcam or speakers. Treat the local OpenCV app as the real demo and Streamlit as the demonstration/deployment interface.

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub.
2. The model files must be available to the deployed app. They are small (about 6 MB and 4 MB), so the simplest option is to remove the `models/*.pt` and `models/*.task` lines from `.gitignore` and commit them, after checking the licenses above.
3. Create the app on Streamlit Community Cloud with `streamlit_app.py` as the main file.
4. If the build reports a missing system library for OpenCV (such as `libGL`), add a `packages.txt` file containing `libgl1`.
5. Use the snapshot mode in the deployed app.

## Run Tests

```bash
python -m pytest
```

The tests use synthetic values and fake models: no webcam, YOLO weights, MediaPipe model or audio device is needed.

## Tuning

Everything adjustable is in `app/config.py`. Common changes:

| Problem | Setting |
|---------|---------|
| Eyes flagged closed too easily (glasses, squinting) | lower `EAR_THRESHOLD` |
| Eyes not flagged when closed | raise `EAR_THRESHOLD` |
| Yawn never detected | lower `MAR_THRESHOLD` |
| Phone box flickers or is missed | adjust `PHONE_CONFIDENCE_THRESHOLD`, `YOLO_IMAGE_SIZE` |
| Low FPS | raise `YOLO_FRAME_SKIP`, lower `YOLO_IMAGE_SIZE` |
| Alerts too chatty | raise `ALERT_COOLDOWN_SECONDS` |

## Project Structure

| Path | Responsibility |
|------|----------------|
| `app/config.py` | All thresholds, weights, paths and performance settings |
| `app/vision.py` | MediaPipe landmarks, EAR, MAR, blink tracking, head pose, landmark drawing |
| `app/phone_detector.py` | YOLO phone detection, frame skipping, temporal confirmation |
| `app/drowsiness.py` | Risk score, driver states with hysteresis, phone state, combined status |
| `app/alerts.py` | Non-blocking voice alerts with priority and cooldown |
| `app/logger.py` | CSV event logging and event types |
| `app/main.py` | `DriveSafePipeline`, overlay drawing and the local webcam loop |
| `app/__init__.py` | Empty file that makes `app` a Python package |
| `streamlit_app.py` | Streamlit dashboard |
| `tests/test_drowsiness.py` | Unit and integration tests |
| `models/` | Model files (downloaded, not committed) |
| `data/logs/` | Runtime event logs (not committed) |

## Privacy

No video or images are stored. Only numeric detection events are written to `data/logs/events.csv`: timestamp, event type, risk score, EAR, MAR, head angles, phone confidence and state. There is no face recognition and no driver identification. Frames are processed in memory and discarded.

## Demo

<!-- Add a screenshot, GIF or demo video here, for example: -->
<!-- ![DriveSafe AI demo](docs/demo.gif) -->

Suggested 15 to 30 second recording (screen-record the local app):

1. **Normal**: look at the camera. Overlay shows `AWAKE`, low risk.
2. **Eyes closed**: close your eyes for 2 to 3 seconds. EAR drops, risk climbs, state moves to `DROWSY`, and you hear "Hey! Wake up! Please stay alert."
3. **Phone**: hold a phone in view. `PHONE: DETECTED`, then in use after about a second, and you hear "Please put the phone down."
4. **Combined**: hold the phone and close your eyes. The banner shows `CRITICAL: DROWSINESS + PHONE USE` with the combined voice warning.

Describe it as a "driver monitoring prototype" or "real-time drowsiness and distraction detection". Do not claim it prevents accidents or give accuracy figures you have not measured.

## Limitations

- Webcam lighting affects detection
- Glasses may affect eye landmarks
- Camera angle affects accuracy, including head pitch, which is not calibrated per driver
- If the face is lost (for example the head drops out of frame), eye-closure timing restarts
- Phone detection depends on the YOLO model and can miss or confuse objects
- Thresholds and weights are untuned starting points, not validated values
- Not automotive-certified
- The risk score is not medically validated

## Future Improvements

- Custom-trained drowsiness model (for example NTHU-DDD, after checking its license)
- Feature vector into Random Forest or XGBoost, later LSTM/GRU on temporal sequences
- Better phone-use interaction detection (hand near ear or face)
- Infrared camera support
- Edge deployment
- Driver-specific calibration

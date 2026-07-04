# Edge Surveillance

Multi-camera edge surveillance system with drone detection and gesture control.

**Status:** Phase I/II — Foundational scaffolding, motion-triggered detection, and gesture interaction framework.

## Overview

A fully local, edge-executable surveillance pipeline that:
- Detects and tracks objects from an aerial (drone) perspective
- Preserves object identity when handing off tracking from drone to ground camera
- Allows hands-free operator control via hand gestures (swipe, fist)

## Tech Stack

- **Detection:** YOLOv8-nano (Ultralytics)
- **Tracking:** ByteTrack
- **Gesture Recognition:** MediaPipe Hands
- **Model Runtime:** ONNX Runtime
- **Video I/O:** OpenCV

## Project Structure

```
edge-surveillance/
  src/
    detection/
    tracking/
    gesture/
    state/
    main.py
  models/
  configs/
  requirements.txt
  README.md
```

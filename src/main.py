"""Combined demo: detection + tracking + gesture control on a single webcam feed.

Detection/tracking runs on the main thread.
Gesture classification runs in a parallel worker thread on the same frames.
"""

import time
import threading
import queue
import cv2
import numpy as np
from ultralytics import YOLO

from gesture.landmark_extractor import LandmarkExtractor
from gesture.gesture_classifier import GestureClassifier

MOTION_THRESHOLD = 5000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
LOCK_DURATION = 2.0


def gesture_worker(frame_queue, event_queue, stop_event):
    extractor = LandmarkExtractor()
    classifier = GestureClassifier()

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        landmarks, _ = extractor.extract(frame)
        _, event = classifier.update(landmarks)

        if event:
            event_queue.put(event)

    extractor.close()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    model = YOLO("yolov8n.pt")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=36, detectShadows=False
    )

    frame_queue = queue.Queue(maxsize=2)
    event_queue = queue.Queue()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=gesture_worker,
        args=(frame_queue, event_queue, stop_event),
        daemon=True,
    )
    thread.start()

    fps_start = time.time()
    frame_count = 0
    target_locked_until = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        fg_mask = bg_subtractor.apply(frame)
        motion_pixels = np.count_nonzero(fg_mask)

        if motion_pixels > MOTION_THRESHOLD:
            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False
            )
        else:
            results = []

        largest_bbox = None
        largest_area = 0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                box_xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for x1, y1, x2, y2, tid, cls, conf in zip(
                    box_xyxy[:, 0], box_xyxy[:, 1], box_xyxy[:, 2],
                    box_xyxy[:, 3], track_ids, clss, confs
                ):
                    ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
                    area = (ix2 - ix1) * (iy2 - iy1)
                    color = (0, 255, 0)

                    now = time.time()
                    if now < target_locked_until and area > largest_area:
                        largest_area = area
                        largest_bbox = (ix1, iy1, ix2, iy2)
                        color = (0, 0, 255)

                    cv2.rectangle(display, (ix1, iy1), (ix2, iy2), color, 2)
                    label = f"{model.names[cls]} {tid} {conf:.2f}"
                    cv2.putText(
                        display, label, (ix1, iy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    )

        while not event_queue.empty():
            gesture = event_queue.get_nowait()
            if gesture == "fist":
                target_locked_until = time.time() + LOCK_DURATION
            elif gesture == "swipe":
                print(
                    "VIEW SWAP TRIGGERED "
                    "(single-camera demo — second stream not yet connected)"
                )

        if largest_bbox is not None and time.time() < target_locked_until:
            ix1, iy1, ix2, iy2 = largest_bbox
            cv2.putText(
                display, "TARGET LOCKED", (ix1, iy2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        remaining = target_locked_until - time.time()
        if remaining > 0:
            cv2.putText(
                display, f"LOCKED {remaining:.1f}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        status = "HEAVY" if motion_pixels > MOTION_THRESHOLD else "idle"
        cv2.putText(
            display, f"Mode: {status} | Motion: {motion_pixels}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        cv2.imshow("Edge Surveillance — Combined Demo", display)

        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
            fps_start = time.time()
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()
    thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

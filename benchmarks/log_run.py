"""Benchmark log runner — wraps the combined pipeline and logs per-frame metrics.

Writes a CSV to benchmarks/results/ with: timestamp, fps, detection_ms,
gesture_inference_ms.
"""

import sys
import os
import csv
import time
import threading
import queue
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gesture.landmark_extractor import LandmarkExtractor
from gesture.gesture_classifier import GestureClassifier

MOTION_THRESHOLD = 5000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
DURATION_SECONDS = 30


def gesture_worker(frame_queue, event_queue, timing_queue, stop_event):
    extractor = LandmarkExtractor()
    classifier = GestureClassifier()

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame, frame_id = item
        t0 = time.perf_counter()
        landmarks, _ = extractor.extract(frame)
        _, event = classifier.update(landmarks)
        t1 = time.perf_counter()

        timing_queue.put((frame_id, (t1 - t0) * 1000))

        if event:
            event_queue.put(event)

    extractor.close()


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"benchmark_{timestamp_str}.csv")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    model = YOLO("yolov8n.pt")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=36, detectShadows=False
    )

    frame_queue = queue.Queue(maxsize=2)
    event_queue = queue.Queue()
    timing_queue = queue.Queue()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=gesture_worker,
        args=(frame_queue, event_queue, timing_queue, stop_event),
        daemon=True,
    )
    thread.start()

    fps_start = time.time()
    frame_count = 0
    frame_id = 0
    target_locked_until = 0.0
    pending_gesture_ms = {}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "fps", "detection_ms", "gesture_inference_ms"])

        start_time = time.time()

        while time.time() - start_time < DURATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_queue.put_nowait((frame.copy(), frame_id))
            except queue.Full:
                pass

            fg_mask = bg_subtractor.apply(frame)
            motion_pixels = np.count_nonzero(fg_mask)

            detection_ms = 0
            if motion_pixels > MOTION_THRESHOLD:
                t0 = time.perf_counter()
                results = model.track(
                    frame, persist=True, tracker="bytetrack.yaml", verbose=False
                )
                t1 = time.perf_counter()
                detection_ms = (t1 - t0) * 1000
            else:
                results = []

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
                        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                        label = f"{model.names[cls]} {tid} {conf:.2f}"
                        cv2.putText(
                            frame, label, (ix1, iy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                        )

            while not event_queue.empty():
                event_queue.get_nowait()

            gesture_ms = 0
            while not timing_queue.empty():
                done_id, ms = timing_queue.get_nowait()
                if done_id == frame_id:
                    gesture_ms = ms

            fps = 0
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                fps_start = time.time()
                frame_count = 0

            writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                f"{fps:.1f}" if fps else "",
                f"{detection_ms:.2f}",
                f"{gesture_ms:.2f}",
            ])

            status = "HEAVY" if motion_pixels > MOTION_THRESHOLD else "idle"
            cv2.putText(
                frame, f"BENCHMARK | Mode: {status} | {time.time() - start_time:.0f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
            cv2.imshow("Benchmark Run", frame)

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    stop_event.set()
    thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nBenchmark complete. Results saved to: {csv_path}")
    print(f"Duration: {DURATION_SECONDS}s target")


if __name__ == "__main__":
    main()

"""Motion-triggered YOLOv8n detection with ByteTrack on webcam feed.

Energy-efficient wakeup: only runs full inference when MOG2 background
subtraction detects motion above a configurable threshold.
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

MOTION_THRESHOLD = 5000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    model = YOLO("yolov8n.pt")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=36, detectShadows=False
    )

    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        motion_pixels = np.count_nonzero(fg_mask)

        if motion_pixels > MOTION_THRESHOLD:
            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False
            )
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
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[cls]} {tid} {conf:.2f}"
                    cv2.putText(
                        frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )

        status = "HEAVY" if motion_pixels > MOTION_THRESHOLD else "idle"
        cv2.putText(
            frame, f"Mode: {status} | Motion: {motion_pixels}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        cv2.imshow("Motion-Triggered Detection", frame)

        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
            fps_start = time.time()
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

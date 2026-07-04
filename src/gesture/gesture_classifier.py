"""Rule-based gesture classification from hand landmark sequences.

Gesture vocabulary: swipe, fist, none
Debounce: same gesture must be classified for 5 consecutive frames
before being confirmed as an event.
"""

import math
import time


FINGERTIP_IDS = [4, 8, 12, 16, 20]
FIST_DISTANCE_THRESHOLD = 0.35
SWIPE_DISTANCE_THRESHOLD = 0.15
DEBOUNCE_FRAMES = 5
COOLDOWN_SECONDS = 1.0


def wrist_center(landmarks):
    wrist = landmarks[0]
    palm = (
        (wrist[0] + landmarks[9][0]) / 2,
        (wrist[1] + landmarks[9][1]) / 2,
    )
    return palm


def classify(landmarks):
    """Classify a single frame's landmarks into swipe/fist/none using history."""
    if not landmarks:
        return "none"

    hand = landmarks[0]
    palm_x, palm_y = wrist_center(hand)
    max_fingertip_dist = 0.0
    for fid in FINGERTIP_IDS:
        tx, ty, _ = hand[fid]
        dist = math.sqrt((tx - palm_x) ** 2 + (ty - palm_y) ** 2)
        if dist > max_fingertip_dist:
            max_fingertip_dist = dist

    return "fist" if max_fingertip_dist < FIST_DISTANCE_THRESHOLD else "none"


class GestureClassifier:
    def __init__(self):
        self.history = []
        self.current_gesture = "none"
        self.streak_count = 0
        self.last_event_time = 0.0
        self.last_swipe_x = None
        self.swipe_frames = []

    def update(self, landmarks):
        raw = classify(landmarks)

        swipe_detected = False
        swipe_dir = None
        if landmarks:
            hand = landmarks[0]
            palm_x, _ = wrist_center(hand)
            if self.last_swipe_x is not None:
                self.swipe_frames.append(palm_x)
                if len(self.swipe_frames) > 10:
                    self.swipe_frames.pop(0)

                if len(self.swipe_frames) >= 10:
                    delta = abs(self.swipe_frames[-1] - self.swipe_frames[0])
                    if delta > SWIPE_DISTANCE_THRESHOLD:
                        swipe_detected = True
                        swipe_dir = "right" if self.swipe_frames[-1] > self.swipe_frames[0] else "left"

                    if raw == "none" and swipe_detected:
                        raw = "swipe"

            self.last_swipe_x = palm_x
        else:
            self.last_swipe_x = None
            self.swipe_frames = []

        if raw == self.current_gesture:
            self.streak_count += 1
        else:
            self.current_gesture = raw
            self.streak_count = 1

        event = None
        now = time.time()
        if self.streak_count >= DEBOUNCE_FRAMES and self.current_gesture != "none":
            if now - self.last_event_time >= COOLDOWN_SECONDS:
                event = self.current_gesture
                self.last_event_time = now

        return self.current_gesture, event


def main():
    import cv2
    from landmark_extractor import LandmarkExtractor

    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()
    classifier = GestureClassifier()

    print("Gesture control active. Show 'fist' or 'swipe'. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, annotated = extractor.extract(frame)
        _, event = classifier.update(landmarks)

        if event:
            print(f"GESTURE DETECTED: {event}")

        cv2.putText(
            annotated,
            f"Gesture: {classifier.current_gesture} ({classifier.streak_count})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Gesture Control", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

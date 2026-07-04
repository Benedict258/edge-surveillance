"""Extract 21 hand landmark coordinates from webcam using MediaPipe Hands."""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class LandmarkExtractor:
    def __init__(self, static_image_mode=False, max_num_hands=1,
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append((lm.x, lm.y, lm.z))
                landmarks.append(coords)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return landmarks, frame

    def close(self):
        self.hands.close()

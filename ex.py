import cv2
import mediapipe as mp
import numpy as np
import time
import math
import sys

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)
capture.set(3, 1200)
capture.set(4, 900)
capture.set(10, 100)

with mp_hands.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    max_num_hands=1
) as hands:
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img2 = frame.copy()
        imgHSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        image_height, image_width, _ = frame.shape

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Index finger tip position
                x = int(hand_landmarks.landmark[8].x * image_width)
                y = int(hand_landmarks.landmark[8].y * image_height)

                # Draw a circle at the index finger tip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break
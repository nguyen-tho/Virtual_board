import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)
capture.set(3, 1200)
capture.set(4, 900)
capture.set(10, 100)

points = []
drawing = False

# Vùng nút xoá: góc trên bên trái 100x100px
clear_zone_x, clear_zone_y = 100, 100  

def wipe(imgg):
    contours, _ = cv2.findContours(imgg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 45000:
            points.clear()

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

                x = int(hand_landmarks.landmark[8].x * image_width)
                y = int(hand_landmarks.landmark[8].y * image_height)

                # kiểm tra khoảng cách giữa ngón trỏ và ngón cái
                x_thumb = int(hand_landmarks.landmark[4].x * image_width)
                y_thumb = int(hand_landmarks.landmark[4].y * image_height)
                distance = np.hypot(x - x_thumb, y - y_thumb)

                drawing = distance < 30

                # Kiểm tra nếu ngón trỏ chạm vào vùng clear zone
                if x < clear_zone_x and y < clear_zone_y:
                    points.clear()

                # nếu đang vẽ thì thêm điểm mới
                if drawing:
                    points.append([x, y])

        # Vẽ tất cả các đường nối giữa các điểm đã lưu
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(img2, (points[i-1][0], points[i-1][1]), (points[i][0], points[i][1]), (0, 0, 255), 5)

        # Vẽ nút xoá (vùng màu xanh ở góc trái trên)
        cv2.rectangle(img2, (0, 0), (clear_zone_x, clear_zone_y), (0, 255, 0), cv2.FILLED)
        cv2.putText(img2, 'CLEAR', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        lower = np.array([5, 31, 207])
        upper = np.array([27, 135, 255])
        mask = cv2.inRange(imgHSV, lower, upper)
        wipe(mask)

        cv2.imshow('Virtual Pen', img2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()

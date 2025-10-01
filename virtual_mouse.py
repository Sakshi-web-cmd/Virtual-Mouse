import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ------------------ Initialize ------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
dragging = False  # Track drag state

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)          # Mirror image
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_detector.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks)
            
            # ------------------ Landmarks ------------------
            landmarks = hand_landmarks.landmark

            # Draw rectangle around hand
            x_list = [int(lm.x * w) for lm in landmarks]
            y_list = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0,255,0), 2)

            # Fingertips positions
            index = (int(landmarks[8].x * w), int(landmarks[8].y * h))
            middle = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            ring = (int(landmarks[16].x * w), int(landmarks[16].y * h))
            thumb = (int(landmarks[4].x * w), int(landmarks[4].y * h))

            # Map index finger to screen
            screen_x = int(landmarks[8].x * screen_w)
            screen_y = int(landmarks[8].y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # ------------------ Gestures ------------------
            # Left Click: index + middle
            # ------------------ Gestures ------------------
            # Left Click: thumb + index pinch
            last_click=0
            # Left Click: thumb + index pinch with cooldown
            if distance(index, thumb) < 30:
                if time.time() - last_click > 0.5:  # 0.5 second cooldown
                    pyautogui.click()
                    last_click = time.time()
            # Right Click: index + middle pinch
            elif distance(index, middle) < 30:
                pyautogui.rightClick()
                pyautogui.sleep(0.2)

            # Drag & Drop: middle + ring pinch
            elif distance(middle, ring) < 30:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
            # Scroll: vertical movement of index + middle
            if distance(index, middle) < 40:
                if index[1] < middle[1] - 40:
                    pyautogui.scroll(50)   # scroll up
                elif index[1] > middle[1] + 40:
                    pyautogui.scroll(-50)  # scroll down

    # ------------------ Show Webcam ------------------
    cv2.imshow("Virtual Mouse Advanced", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
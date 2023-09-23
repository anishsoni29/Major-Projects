import cv2
import pyautogui
import mediapipe as mp

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# Initialize variables
hand_gesture = 'other'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            if index_finger.y < thumb_y:
                hand_gesture = 'volumeup'
            elif index_finger.y > thumb_y:
                hand_gesture = 'volumedown'
            else:
                hand_gesture = 'other'

            # Perform volume control actions based on hand gestures
            if hand_gesture == 'volumeup':
                pyautogui.press('volumeup')
            elif hand_gesture == 'volumedown':
                pyautogui.press('volumedown')

    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

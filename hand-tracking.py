import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def is_finger_up(landmarks, tip, pip):
    return (landmarks[tip].y < landmarks[pip].y)

def is_thumb_up(landmarks, hand_label):
    if hand_label == "right":
        return (landmarks[4].x > landmarks[3].x)
    else:
        return (landmarks[4].x < landmarks[3].x)

def classify_hand_gesture(landmarks, hand_label):
    index_up = is_finger_up(landmarks, 8, 6) # index
    middle_up = is_finger_up(landmarks, 12, 10) # middle
    ring_up = is_finger_up(landmarks, 16, 14) # ring
    pinky_up = is_finger_up(landmarks, 20, 18)  # pinky
    thumb_up = is_thumb_up(landmarks, hand_label)

    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return "open palm"
    elif not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "fist"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        image.flags.writeable = True

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_handedness.classification[0].label
                    
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks,mp_hands.HAND_CONNECTIONS, 
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    gesture = classify_hand_gesture(hand_landmarks.landmark, hand_label)
                    
                    cv2.putText(image, f"{hand_label} - {gesture}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow('Hand Gesture Detection', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
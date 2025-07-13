import cv2
import mediapipe as mp

def is_finger_up(landmarks, tip, pip):
    return (landmarks[tip].y < landmarks[pip].y)

def is_thumb_up(landmarks, hand_label):
    tip = landmarks[4]
    joint = landmarks[3]
    if hand_label == "Left":
        return tip.x > joint.x
    else:
        return tip.x < joint.x  

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
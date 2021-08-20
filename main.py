import math
import keyboard
import pyautogui
import cv2
import mediapipe as mp
import pygetwindow as gw
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

pyautogui.PAUSE = 0
e_time = 0
right_click_time = 0


def is_hand_thumb_up(landmarks):
    for landmark in landmarks:
        if landmark.y < landmarks[4].y:  # y 0 is at top
            return False
    return True


def get_thumbs_up(data):
    landmarks = []
    for data_landmarks in data.multi_hand_landmarks:
        for idx, landmark in enumerate(data_landmarks.landmark):
            landmarks.append(landmark)

    if len(landmarks) == 0:
        return False, False
    elif len(landmarks) == 21:
        return (is_hand_thumb_up(landmarks), False) if landmarks[4].x <= .5 else (False, is_hand_thumb_up(landmarks))
    else:
        return (is_hand_thumb_up(landmarks[:21]), is_hand_thumb_up(landmarks[21:])) if landmarks[4].x <= 0.5 else (is_hand_thumb_up(landmarks[21:]), is_hand_thumb_up(landmarks[:21]))


def is_fist_clenched(landmarks):
    distance_sum = 0.0
    for i in range(0, len(landmarks)):
        for j in range(0, len(landmarks)):
            distance_sum += math.sqrt(math.pow(landmarks[i].x - landmarks[j].x, 2) + math.pow(landmarks[i].y - landmarks[j].y, 2) + math.pow(landmarks[i].z - landmarks[j].z, 2))
    avg_distance = distance_sum / len(landmarks) / len(landmarks)
    return avg_distance < .1


def get_hands_shown(data):  # 0 is base of hand, 4 is end of thumb
    landmarks = []
    for data_landmarks in data.multi_hand_landmarks:
        for idx, landmark in enumerate(data_landmarks.landmark):
            landmarks.append(landmark)

    if len(landmarks) == 0:
        return 'gone', 'gone'
    if len(landmarks) == 42:
        return 'clenched' if is_fist_clenched(landmarks[:21]) else 'shown', 'clenched' if is_fist_clenched(landmarks[21:]) else 'shown'  # TODO may be wrong
    elif landmarks[4].x > landmarks[0].x:  # thumb is to the right of hand base (assumes palm is facing camera)
        return 'clenched' if is_fist_clenched(landmarks) else 'shown', 'gone'
    else:
        return 'gone', 'clenched' if is_fist_clenched(landmarks) else 'shown'


def is_mouth_open(data):  # 13 is top lip, 14 is bottom lip
    landmarks = []
    for data_landmarks in data.multi_face_landmarks:
        for idx, landmark in enumerate(data_landmarks.landmark):
            landmarks.append(landmark)

    distance = math.sqrt(math.pow(landmarks[14].x - landmarks[13].x, 2) + math.pow(landmarks[14].y - landmarks[13].y, 2) + math.pow(landmarks[14].z - landmarks[13].z, 2))
    if distance > .01:
        if distance > .05:
            return 'run'
        else:
            return 'walk'
    else:
        return 'stay'


def are_eyebrows_raised(data):  # 65 is bottom of left eyebrow, 133 is the rightmost point of the left eye
    landmarks = []
    for data_landmarks in data.multi_face_landmarks:
        for idx, landmark in enumerate(data_landmarks.landmark):
            landmarks.append(landmark)

    eye_eyebrow_distance = math.sqrt(math.pow(landmarks[65].x - landmarks[133].x, 2) + math.pow(landmarks[65].y - landmarks[133].y, 2) + math.pow(landmarks[65].z - landmarks[133].z, 2))
    return eye_eyebrow_distance > .06


def get_face_rotation(data):  # 93 is left, 323 is right, 109 is top, 148 is bottom
    landmarks = []
    for data_landmarks in data.multi_face_landmarks:
        for idx, landmark in enumerate(data_landmarks.landmark):
            landmarks.append(landmark)

    left_right_delta_z = landmarks[93].z - landmarks[323].z
    top_bottom_delta_z = landmarks[109].z - landmarks[148].z
    final_delta_y = landmarks[109].y - landmarks[148].y

    left_right_axis = 'none'
    if left_right_delta_z > 0.1:
        left_right_axis = 'left'
    elif left_right_delta_z < -0.1:
        left_right_axis = 'right'
    top_bottom_axis = 'none'
    if top_bottom_delta_z > 0.1:
        top_bottom_axis = 'up'
    elif top_bottom_delta_z < -0.05:
        top_bottom_axis = 'down'
    final_axis = 'none'
    if abs(final_delta_y) < 0.25:
        if landmarks[109].x < landmarks[148].x:
            final_axis = 'left'
        else:
            final_axis = 'right'

    return left_right_axis, top_bottom_axis, final_axis, left_right_delta_z, top_bottom_delta_z, final_delta_y


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results_face_mesh = face_mesh.process(image)
            results_hands = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the face mesh annotations on the image.
            if results_face_mesh.multi_face_landmarks:
                for face_landmarks in results_face_mesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACE_CONNECTIONS,
                                              landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

            # Draw the hand annotations on the image.
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('MediaPipe Face Mesh and Hands', image)

            # Post-processing and input
            if results_hands.multi_hand_landmarks is not None:
                print(get_thumbs_up(results_hands))
            active_window = gw.getActiveWindow()
            if active_window is not None and 'Minecraft' in active_window.title:
                if results_face_mesh.multi_face_landmarks:
                    if is_mouth_open(results_face_mesh) == 'walk' or is_mouth_open(results_face_mesh) == 'run':
                        keyboard.press('w')
                    else:
                        keyboard.release('w')
                    if is_mouth_open(results_face_mesh) == 'run':
                        keyboard.press('z')
                    else:
                        keyboard.release('z')
                    if are_eyebrows_raised(results_face_mesh):
                        keyboard.press('space')
                    else:
                        keyboard.release('space')

                    rotation = get_face_rotation(results_face_mesh)
                    delta_x = 0
                    delta_y = 0
                    if rotation[0] == 'left':
                        delta_x = -25
                    elif rotation[0] == 'right':
                        delta_x = 25
                    if rotation[1] == 'up':
                        delta_y = -10
                    elif rotation[1] == 'down':
                        delta_y = 10
                    if rotation[2] == 'left' and time.time() - e_time >= 1:
                        keyboard.press_and_release('e')
                        e_time = time.time()
                    elif rotation[2] == 'right':
                        pyautogui.scroll(-10)

                    delta_x = rotation[3] * -200
                    delta_y = rotation[4] * -200

                    pyautogui.moveRel(delta_x, delta_y)

                if results_hands.multi_hand_landmarks is not None:
                    hands_shown = get_hands_shown(results_hands)
                else:
                    hands_shown = 'gone', 'gone'

                if hands_shown[0] == 'shown' or hands_shown[0] == 'clenched':
                    pyautogui.mouseDown(button='left')
                else:
                    pyautogui.mouseUp(button='left')
                if (hands_shown[1] == 'shown' or hands_shown[1] == 'clenched') and time.time() - right_click_time >= 1:
                    pyautogui.click(button='right')
                    right_click_time = time.time()

                if hands_shown[0] == 'clenched' or hands_shown[1] == 'clenched':
                    keyboard.press('left_shift')
                else:
                    keyboard.release('left_shift')

                if results_hands.multi_hand_landmarks is not None:
                    thumbs = get_thumbs_up(results_hands)
                else:
                    thumbs = False, False

                if thumbs[0]:
                    keyboard.press('s')
                else:
                    keyboard.release('s')
                if thumbs[1]:
                    pyautogui.mouseDown(button='right')
                else:
                    pyautogui.mouseUp(button='right')

            # Exit
            if cv2.waitKey(5) & 0xFF == 27:
                break
cap.release()

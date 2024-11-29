import cv2
import numpy as np
import mediapipe as mp
from gesture_module import GestureRecognizer
import serial
import time
import threading
from google.protobuf.json_format import MessageToDict

arduino_serial = serial.Serial('COM3', 9600)
depth = 1

def open_palm():
    global all_servo_lengths
    all_servo_lengths = [0, 180, 0, 0, 180]
    print("OPEN PALM")

def close_palm():
    global all_servo_lengths
    all_servo_lengths = [180, 0, 180, 180, 0]
    print("CLOSE PALM")

def love_you():
    global all_servo_lengths
    all_servo_lengths = [0, 180, 180, 180, 180]
    print("LOVEYOU")

def victory():
    global all_servo_lengths
    all_servo_lengths = [180, 180, 0, 180, 0]
    print("VICTORY")

def point_up():
    global all_servo_lengths
    all_servo_lengths = [0, 180, 180, 0, 0]
    print("POINT UP")

def send_servo_angles():
    while True:
        angles_str = ' '.join(map(str, all_servo_lengths)) + '\n'
        arduino_serial.write(angles_str.encode())
        time.sleep(0.1)

def capture_and_process():
    global taken_off, depth
    while True:
        _, src = cap.read()
        flipped = cv2.flip(src, 1)

        frame, hand_gesture, hand_process_results = gesture.loop_run(flipped, False)

        left_hand = hand_gesture["Left"]
        right_hand = hand_gesture["Right"]

        hands.updateResults(hand_process_results)

        depth = 1

        if hands.results.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(hands.results.multi_hand_landmarks):
                depth = abs(MessageToDict(hand_landmarks)["landmark"][0]["z"])
                depth *= 10000000

        for i in range(max_servos):
            servo_length, is_active = hands.length_between_landmarks(frame, 0, (i * 4) + 4)
            servo_angles = [180, 0] if flip_servo_lengths[i] else [0, 180]
            scaled_length = round(np.interp(servo_length, origin_servo_scale[i], servo_angles))

            all_servo_lengths[i] = scaled_length

            active_text = "ON" if is_active else "OFF"
            #cv2.putText(frame, f"Servo {i}: {scaled_length}, {active_text}", (10, (i * 30) + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2, cv2.LINE_AA)

        for func in gesture_funcs.keys():
            if func == left_hand or func == right_hand:
                #gesture_funcs[func]()
                pass

        cv2.imshow('Hands', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            arduino_serial.close()
            cap.release()
            cv2.destroyAllWindows()
            break

gesture = GestureRecognizer()
hands = HandDetector()

hand_opened = False

gesture_funcs = {
    "Open_Palm": open_palm,
    "Closed_Fist": close_palm,
    "ILoveYou": love_you,
    "Victory": victory,
    "Pointing_Up": point_up
}

window_width = 1280
window_height = 720

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

max_servos = 5
all_servo_lengths = [0, 0, 0, 0, 0]
flip_servo_lengths = [True, False, True, True, False]
origin_servo_scale = [
    [100, 250],
    [150, 325],
    [150, 325],
    [150, 325],
    [150, 325]
]

servo_thread = threading.Thread(target=send_servo_angles)
servo_thread.daemon = True
servo_thread.start()

capture_thread = threading.Thread(target=capture_and_process)
capture_thread.daemon = True
capture_thread.start()

capture_thread.join()
servo_thread.join()

arduino_serial.close()
cap.release()
cv2.destroyAllWindows()

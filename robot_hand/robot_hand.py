import cv2
import threading
import serial
import time
from gestures import GestureRecognizer

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
    print("LOVE YOU")

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
    global depth
    while True:
        ret, frame = gesture_recognizer.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame, hand_data = gesture_recognizer.run(frame)

        left_gesture = hand_data["Left"]["gesture"]
        right_gesture = hand_data["Right"]["gesture"]

        for func_name, func in gesture_funcs.items():
            if func_name == left_gesture or func_name == right_gesture:
                func()

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

gesture_recognizer = GestureRecognizer(use_gestures=True)

gesture_funcs = {
    "Open_Palm": open_palm,
    "Closed_Fist": close_palm,
    "ILoveYou": love_you,
    "Victory": victory,
    "Pointing_Up": point_up
}

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

gesture_recognizer.cleanup()
arduino_serial.close()

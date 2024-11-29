import cv2
import math
from gestures import GestureRecognizer
from enum import Enum

cv2.setUseOptimized(True)

gesture = GestureRecognizer()

window_name = "Gesture Control"
cap = cv2.VideoCapture(0)

window_width = 1280
window_height = 720

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

ControlMode = Enum("ControlMode", ["INACTIVE", "SERVO", "GIMBAL"])

ControlNames = {
    ControlMode.INACTIVE: "Deactivated",
    ControlMode.SERVO: "Individual Servo Control",
    ControlMode.GIMBAL: "Gimbal Joystick",
}

gimbal_pos_set = False

deadzone = 48
mode = ControlMode.INACTIVE
laser_on = False
has_activated = False

max_servos = 4
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    frame, hand_gesture = gesture.loop_run(frame, False)

    left_hand = hand_gesture["Right"]
    right_hand = hand_gesture["Left"]

    if left_hand == "ILoveYou":
        mode = ControlMode.SERVO
    elif left_hand == "Victory":
        mode = ControlMode.GIMBAL
    elif left_hand == "Pointing_Up":
        mode = ControlMode.INACTIVE

    if mode == ControlMode.INACTIVE:
        gimbal_pos_set = False

    elif mode == ControlMode.SERVO:
        gimbal_pos_set = False
        
        for i in range(max_servos):
            servo_length, is_active = gesture.length_between_landmarks(frame, 0, (i * 4) + 8)
            active_text = "ON" if is_active else "OFF"

            cv2.putText(frame, f"Servo {i}: {servo_length}, {active_text}", (10, (i * 30) + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        gesture.length_between_landmarks(frame, 4, 5, "Right", False, 30)
        
    elif mode == ControlMode.GIMBAL:
        landmark_1, landmark_2, landmark_3 = 8, 5, 17
        
        if gesture.landmark_list and gesture.check_landmark_handedness([landmark_1], "Right"):
            x1, y1 = gesture.landmark_list[landmark_1][1], gesture.landmark_list[landmark_1][2]

            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)

            if not gimbal_pos_set and x1 > window_width // 2:
                gimbal_pos_set = True

                center_x = x1
                center_y = y1

            cv2.circle(frame, (center_x, center_y), deadzone, (255, 0, 255), 4)

            cv2.line(frame, (center_x, center_y), (x1, y1), (0, 0, 255), 4)
            cv2.line(frame, (center_x, center_y), (x1, center_y), (0, 255, 255), 4)
            cv2.line(frame, (x1, center_y), (x1, y1), (0, 255, 0), 4)

            _, laser_active = gesture.length_between_landmarks(frame, 4, 5, "Right", True, 30)

            if laser_active:
                if not has_activated:
                    has_activated = True
                    laser_on = not laser_on
            else:
                has_activated = False

        cv2.putText(frame, f"Laser: {laser_on}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Mode: {ControlNames[mode]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

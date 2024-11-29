import cv2
import mediapipe as mp
import threading
import math
from mediapipe.tasks import python
from google.protobuf.json_format import MessageToDict


class GestureRecognizer:
	def __init__(self, use_gestures=False):
		try:
			self.use_gesture_recognition = use_gestures
			self.num_hands = 2
			self.tracking_confidence = 0.55
			self.detection_confidence = 0.55
			self.smoothing_factor = 0.5

			self.hand_data = {
				"Left": {"landmarks": [], "gesture": "None"},
				"Right": {"landmarks": [], "gesture": "None"}
			}

			self.previous_landmarks = {
				"Left": [],
				"Right": []
			}

			if use_gestures:
				model_path = "../models/gesture_recognizer.task"
				GestureRecognizer = mp.tasks.vision.GestureRecognizer
				GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
				VisionRunningMode = mp.tasks.vision.RunningMode

				self.lock = threading.Lock()
				options = GestureRecognizerOptions(
					base_options=python.BaseOptions(model_asset_path=model_path),
					running_mode=VisionRunningMode.LIVE_STREAM,
					num_hands=self.num_hands,
					result_callback=self.results_callback
				)
				self.recognizer = GestureRecognizer.create_from_options(options)

			self.mp_drawing = mp.solutions.drawing_utils
			self.mp_hands = mp.solutions.hands
			self.hands = self.mp_hands.Hands(
				static_image_mode=False,
				max_num_hands=self.num_hands,
				min_detection_confidence=self.detection_confidence,
				min_tracking_confidence=self.tracking_confidence,
				model_complexity=1
			)

			self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

			if not self.cap.isOpened():
				print("Error: Could not open video capture.")
		except Exception as e:
			print(f"Error initializing GestureRecognizer: {e}")

	def smooth_landmarks(self, new_landmarks, hand_label):
		smoothed_landmarks = []

		if len(self.previous_landmarks[hand_label]) == 0:
			self.previous_landmarks[hand_label] = new_landmarks
			return new_landmarks

		for i, (prev, new) in enumerate(zip(self.previous_landmarks[hand_label], new_landmarks)):
			smooth_x = prev[1] * self.smoothing_factor + new[1] * (1 - self.smoothing_factor)
			smooth_y = prev[2] * self.smoothing_factor + new[2] * (1 - self.smoothing_factor)
			smoothed_landmarks.append([new[0], int(smooth_x), int(smooth_y), new[3]])

		self.previous_landmarks[hand_label] = smoothed_landmarks
		return smoothed_landmarks

	def process_hands(self, results, frame):
		if not results.multi_hand_landmarks or not results.multi_handedness:
			self.hand_data["Left"]["landmarks"] = []
			self.hand_data["Left"]["gesture"] = "None"
			self.hand_data["Right"]["landmarks"] = []
			self.hand_data["Right"]["gesture"] = "None"
			return

		for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
			label = MessageToDict(handedness)["classification"][0]["label"]
			landmarks = []
			h, w, c = frame.shape

			for id, lm in enumerate(hand_landmarks.landmark):
				cx, cy = int(lm.x * w), int(lm.y * h)
				landmarks.append([id, cx, cy, label])

			smoothed_landmarks = self.smooth_landmarks(landmarks, label)

			self.hand_data[label]["landmarks"] = smoothed_landmarks
			self.hand_data[label]["gesture"] = self.hand_data[label].get("gesture", "None")

	def run(self, frame: cv2.typing.MatLike, draw_gestures: bool = True):
		try:
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = self.hands.process(frame_rgb)

			self.process_hands(results, frame)

			# for hand_label in ["Left", "Right"]:
			# 	landmarks = self.hand_data[hand_label]["landmarks"]
			# 	if landmarks:
			# 		for lm in landmarks:
			# 			cv2.circle(frame, (lm[1], lm[2]), 6, (255, 0, 0), cv2.FILLED)

			# 		x_min = min([lm[1] for lm in landmarks])
			# 		y_min = min([lm[2] for lm in landmarks])
			# 		x_max = max([lm[1] for lm in landmarks])
			# 		y_max = max([lm[2] for lm in landmarks])

			# 		cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

			if draw_gestures and self.use_gesture_recognition:
				left_gesture = self.hand_data["Left"]["gesture"]
				right_gesture = self.hand_data["Right"]["gesture"]

				cv2.putText(frame, f"Left Hand: {left_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
				cv2.putText(frame, f"Right Hand: {right_gesture}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

			return frame, self.hand_data
		except Exception as e:
			print(f"Error in run: {e}")

	def results_callback(self, result, output_image, timestamp_ms):
		try:
			with self.lock:
				if result and any(result.gestures):
					for index, hand in enumerate(result.handedness):
						hand_name = hand[0].category_name
						current_gesture = result.gestures[index][0].category_name
						self.hand_data[hand_name]["gesture"] = current_gesture
		except Exception as e:
			print(f"Error in results_callback: {e}")

	def cleanup(self):
		try:
			self.cap.release()
			cv2.destroyAllWindows()
		except Exception as e:
			print(f"Error in cleanup: {e}")

if __name__ == "__main__":
	try:
		gesture_recognizer = GestureRecognizer(use_gestures=True)

		while True:
			ret, frame = gesture_recognizer.cap.read()
			if not ret:
				print("Error: Could not read frame.")
				break

			frame, hand_data = gesture_recognizer.run(frame)

			cv2.imshow("Gesture Recognizer", frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		gesture_recognizer.cleanup()
	except Exception as e:
		print(f"Error in main loop: {e}")

import cv2
import math
import numpy as np
import pyautogui
from gestures import GestureRecognizer

class ObjectController(GestureRecognizer):
	def __init__(self, use_gestures=True):
		super().__init__(use_gestures)
		self.object_scale = 1.0
		self.object_rotation = 0.0
		self.object_position = (0, 0)
		self.previous_distance = None
		self.previous_angle = None
		self.key_pressed = False

	def calculate_distance(self, point1, point2):
		x1, y1 = point1
		x2, y2 = point2
		return math.hypot(x2 - x1, y2 - y1)

	def calculate_angle(self, point1, point2):
		x1, y1 = point1
		x2, y2 = point2
		return math.atan2(y2 - y1, x2 - x1)

	def update_object_controls(self):
		left_hand_landmarks = self.hand_data["Left"]["landmarks"]
		right_hand_landmarks = self.hand_data["Right"]["landmarks"]

		min_scale, max_scale = 0.5, 5
		smoothing_factor = 0.2
		rotation_speed = 5
		default_scale = 0.75

		if left_hand_landmarks and right_hand_landmarks:
			landmarks = [0, 5, 17]

			left_coords = [(left_hand_landmarks[l][1], left_hand_landmarks[l][2]) for l in landmarks]
			right_coords = [(right_hand_landmarks[l][1], right_hand_landmarks[l][2]) for l in landmarks]

			left_hand_point = [sum(x) // len(x) for x in zip(*left_coords)]
			right_hand_point = [sum(x) // len(x) for x in zip(*right_coords)]

			x_mid = (left_hand_point[0] + right_hand_point[0]) // 2
			y_mid = (left_hand_point[1] + right_hand_point[1]) // 2
			self.object_position = (x_mid, y_mid)

			current_distance = self.calculate_distance(left_hand_point, right_hand_point)
			if self.previous_distance is not None:
				scale_change = (current_distance - self.previous_distance) / 100.0
				self.object_scale += scale_change
				self.object_scale = max(min_scale, min(self.object_scale, max_scale))
			self.previous_distance = current_distance

			current_angle = self.calculate_angle(left_hand_point, right_hand_point)
			if self.previous_angle is not None:
				rotation_change = math.degrees(current_angle - self.previous_angle)
				self.object_rotation += rotation_change
				self.object_rotation %= 360 
			self.previous_angle = current_angle
		else:
			target_position = (640, 480)
			current_position_x = int(self.object_position[0] * (1 - smoothing_factor) + target_position[0] * smoothing_factor)
			current_position_y = int(self.object_position[1] * (1 - smoothing_factor) + target_position[1] * smoothing_factor)
			self.object_position = (current_position_x, current_position_y)

			self.object_rotation = (self.object_rotation + rotation_speed) % 360

			self.object_scale = self.object_scale * (1 - smoothing_factor) + default_scale * smoothing_factor
		
		return left_hand_landmarks and right_hand_landmarks

	def draw_object(self, frame, hands_present):
		x, y = self.object_position
		size = int(50 * self.object_scale)
		rad_angle = math.radians(self.object_rotation)
		current_mode = "?"

		current_color = (255, 0, 30)

		if hands_present:
			if self.object_rotation >= 0 and self.object_rotation <= 180:
				current_color = (0, 0, 255)
				current_mode = "Latch"
			elif self.object_rotation > 180 and self.object_rotation <= 360: 
				current_color = (0, 255, 0)
				current_mode = "Conex"
			
			if self.object_scale >= 3:
				current_mode += " ACTIVE"
				current_color = (255, 255, 255)

				if not self.key_pressed:
					self.key_pressed = True
					pyautogui.press("right")
			else:
				self.key_pressed = False

		corners = [
			(-size, -size), 
			(size, -size), 
			(size, size), 
			(-size, size)
		]

		rotated_corners = [
			(
				int(x + math.cos(rad_angle) * cx - math.sin(rad_angle) * cy),
				int(y + math.sin(rad_angle) * cx + math.cos(rad_angle) * cy),
			)
			for cx, cy in corners
		]

		text_size = cv2.getTextSize(current_mode, cv2.FONT_HERSHEY_COMPLEX, 1, 3)[0]
		text_width = text_size[0]
		text_x = x - text_width // 2 

		cv2.putText(frame, current_mode, (text_x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

		cv2.polylines(frame, [np.array(rotated_corners)], isClosed=True, color=current_color, thickness=4)

	def run(self, frame: cv2.typing.MatLike, draw_gestures: bool = True):
		frame, data = super().run(frame, draw_gestures)
		
		hands_present = self.update_object_controls()
		self.draw_object(frame, hands_present)

		return frame

if __name__ == "__main__":
	try:
		controller = ObjectController(use_gestures=False)

		while True:
			ret, frame = controller.cap.read()
			frame = cv2.flip(frame, 1)

			if not ret:
				print("Error: Could not read frame.")
				break

			frame = controller.run(frame)

			cv2.imshow("Object Controller", frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		controller.cleanup()
	except Exception as e:
		print(f"Error in main loop: {e}")

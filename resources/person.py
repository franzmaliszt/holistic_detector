from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.holistic import HandLandmark
from resources import KeyPoint


class Person:
    def __init__(self):
        self.face = {}
        self.left_hand = {}
        self.right_hand = {}
        self.pose = {}
        self.angle_list = {}

    def read_landmark(self, result):
        if result.face_landmarks:
            for idx, lm in enumerate(result.face_landmarks.landmark):
                point = KeyPoint.create_for_face(idx, lm)
                self.face[point.name] = point

        if result.left_hand_landmarks:
            for lm, name in zip(result.left_hand_landmarks.landmark, HandLandmark):
                point = KeyPoint.create_from_landmark(name, lm)
                self.left_hand[point.name] = point

        if result.right_hand_landmarks:
            for lm, name in zip(result.right_hand_landmarks.landmark, HandLandmark):
                point = KeyPoint.create_from_landmark(name, lm)
                self.right_hand[point.name] = point

        if result.pose_landmarks:
            for lm, name in zip(result.pose_landmarks.landmark, PoseLandmark):
                point = KeyPoint.create_from_landmark(name, lm)
                self.pose[point.name] = point

import cv2
import csv
import os
from time import time
import numpy as np
import pandas as pd
import pickle

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import holistic as mp_holistic

from resources import Person, KeyPoint
from config import *


class Detector:
    def __init__(self):
        self.model = None
        self.image = None
        self.cap = None
        self.result = None
        self.person = Person()
        self.body_language_prob = None
        self.stage = None
        self.stage_counter = 0
        self._time = 0
        self._frame_count = 0
        self.capture_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._quit()

    def init_model(self, path=None, *args):
        self._time = time()
        self.model = mp_holistic.Holistic(args)
        if path:
            self.cap = cv2.VideoCapture(path)
        else:
            self.cap = cv2.VideoCapture(0)

        self.cap.set(3, WIDTH)
        self.cap.set(4, HEIGHT)

    def _gather(self):
        ret, frame = self.cap.read()
        if ret:
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            self._quit()

    def _update(self):
        self._process_image()
        self._get_landmarks()
        self._check_patterns()
        # self.predict()

    def _render(self):
        def get_fps():
            inference = time() - self._time
            fps = 0
            if inference > 0:
                fps = self._frame_count / inference
                self._frame_count = 0
                self._time = time()
            return str(int(fps))

        self._frame_count += 1
        mp_drawing.draw_landmarks(self.image, self.result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(self.image, self.result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        mp_drawing.draw_landmarks(self.image, self.result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(self.image, self.result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        font = cv2.FONT_HERSHEY_SIMPLEX
        blue = (0, 255, 0)
        black = (255, 255, 255)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        # feedbox
        cv2.rectangle(self.image, (0, 0), (90, 50), (245, 117, 16), -1)
        cv2.putText(self.image, 'count: {}'.format(str(self.stage_counter)), (10, 40), font, .5, black, 1, cv2.LINE_AA)
        # fps counter
        cv2.putText(self.image, 'fps: {}'.format(get_fps()), (10, 20), font, .5, black, 1, cv2.LINE_AA)
        # landmark information
        [cv2.putText(self.image, '{}: {}'.format(point.name, int(self.person.angle_list[point.name])),
                     (point.x, point.y), font, .4, blue, 1, cv2.LINE_AA)
         for _, point in self.person.pose.items() if point.name in self.person.angle_list.keys()]
        # Prediction information
        # cv2.putText(self.image, 'PROB'
        #             , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(self.image, str(round(self.body_language_prob[np.argmax(self.body_language_prob)], 2))
        #             , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Pose", self.image)

    def detect(self):
        def polling():
            key = cv2.waitKey(30)
            if key == 27:
                self._quit()

        while self.cap.isOpened():
            self._gather()
            self._update()
            self._render()
            polling()
        self._quit()

    def _process_image(self):
        self.image.flags.writeable = False
        self.result = self.model.process(self.image)
        self.image.flags.writeable = True

    def _get_landmarks(self):
        if self.result.pose_landmarks:
            # self.capture()
            # self.predict()
            self.person.read_landmark(self.result)

            left_wrist = self.person.pose['LEFT_WRIST']
            right_wrist = self.person.pose['RIGHT_WRIST']
            left_elbow = self.person.pose['LEFT_ELBOW']
            right_elbow = self.person.pose['RIGHT_ELBOW']
            left_shoulder = self.person.pose['LEFT_SHOULDER']
            right_shoulder = self.person.pose['RIGHT_SHOULDER']
            left_hip = self.person.pose['LEFT_HIP']
            right_hip = self.person.pose['RIGHT_HIP']
            self.person.angle_list = {
                'LEFT_ELBOW': KeyPoint.get_angle(left_wrist, left_elbow, left_shoulder),
                'RIGHT_ELBOW': KeyPoint.get_angle(right_wrist, right_elbow, right_shoulder),
                'LEFT_SHOULDER': KeyPoint.get_angle(left_hip, left_shoulder, left_elbow),
                'RIGHT_SHOULDER': KeyPoint.get_angle(right_hip, right_shoulder, right_elbow)
            }

    def capture(self):
        new_label = 'break'
        # model_size = len(self.result.pose_landmarks.landmark) + len(self.result.face_landmarks.landmark)

        labels = ['class']
        for i in range(1, 501 + 1):
            labels += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i), 'v{}'.format(i)]

        if not os.path.isfile('coordinates.csv'):
            with open('coordinates.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(labels)

        row = self.flatten_pose(self.result)
        row.insert(0, new_label)
        if row:
            with open('coordinates.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
                self.capture_count += 1

        if self.capture_count == 288:
            self._quit()

    def predict(self):
        with open('body_language.pkl', 'rb') as f:
            model = pickle.load(f)
            row = self.flatten_pose(self.result)
            if row:
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                self.body_language_prob = body_language_prob
                print(body_language_class, body_language_prob)

    @staticmethod
    def flatten_pose(result):
        try:
            pose = result.pose_landmarks.landmark
            pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())

            face = result.face_landmarks.landmark
            face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face]).flatten())

            return pose_row + face_row
        except:
            pass

    def _check_patterns(self):
        # Define your patterns here
        try:
            left_elbow = self.person.angle_list['LEFT_ELBOW']
            left_shoulder = self.person.angle_list['LEFT_SHOULDER']
            if left_elbow > 160 and left_shoulder > 130:
                self.stage = 'open'
            elif left_elbow < 110 and left_shoulder < 160 and self.stage == 'open':
                self.stage = 'center'
                self.stage_counter += 1
        except:
            pass

    def _quit(self):
        self.cap.release()
        cv2.destroyAllWindows()

# import dependencies


from cv2 import imshow
from cv2 import putText
from cv2 import waitKey
from cv2 import rectangle
from cv2 import VideoCapture
from cv2 import destroyAllWindows
from cv2 import FONT_HERSHEY_SIMPLEX

from numpy import array
from numpy import argmax
from numpy import concatenate
from numpy import expand_dims

from tensorflow.keras.models import load_model

import mediapipe
Holistic = mediapipe.solutions.holistic.Holistic
POSE_CONNECTIONS = mediapipe.solutions.holistic.POSE_CONNECTIONS
HAND_CONNECTIONS = mediapipe.solutions.holistic.HAND_CONNECTIONS
draw_landmarks = mediapipe.solutions.drawing_utils.draw_landmarks


# define variables


SIGNS = {
    0 : "-",
    1 : "hola",
    2 : "como estas",
    3 : "bien",
    4 : "mal",
    5 : "con permiso",
    6 : "gracias",
    7 : "de nada",
    8 : "por favor",
    9 : "perdon",
    10: "adios",
    11: "cuidate",
    12: "nos vemos",
    13: "te quiero"
}


# define functions


def show_image(image, text=None, landmarks=None):

    if text:
        rectangle(image, (image.shape[1], image.shape[0]), (0, image.shape[0] - 40), (0, 0, 0), -1)
        putText(image, text.upper(), (10, image.shape[0] - 10), FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    if landmarks:
        draw_landmarks(image, landmarks.pose_landmarks, POSE_CONNECTIONS)
        draw_landmarks(image, landmarks.left_hand_landmarks, HAND_CONNECTIONS)
        draw_landmarks(image, landmarks.right_hand_landmarks, HAND_CONNECTIONS)

    imshow("Video Capture", image)


def get_keypoints(landmarks):

    pose = array([[lm.x, lm.y, lm.visibility] for lm in landmarks.pose_landmarks.landmark]).flatten() if landmarks.pose_landmarks else array(3 * 33 * [-1.0])
    left = array([[lm.x, lm.y, lm.z] for lm in landmarks.left_hand_landmarks.landmark]).flatten() if landmarks.left_hand_landmarks else array(3 * 21 * [-1.0])
    right = array([[lm.x, lm.y, lm.z] for lm in landmarks.right_hand_landmarks.landmark]).flatten() if landmarks.right_hand_landmarks else array(3 * 21 * [-1.0])

    return concatenate([pose, left, right])


# import translator


translator = load_model("models/python/model.h5")


# test translator


capture = VideoCapture(0)

with Holistic() as tracker:
    while capture.isOpened():

        _, image = capture.read()

        landmarks = tracker.process(image)

        detection = translator.predict(expand_dims(a=get_keypoints(landmarks), axis=0)).flatten()
        prediction = f"{'{:.2f}'.format(detection[argmax(detection)])} <=> {SIGNS[argmax(detection)]}"

        show_image(image, prediction, landmarks)

        if waitKey(1) & 0xFF == ord("q"):
            break

capture.release()
destroyAllWindows()

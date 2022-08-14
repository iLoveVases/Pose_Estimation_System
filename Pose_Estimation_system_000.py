# IMPORT #########################################################################
import tensorflow as tf
from numpy import size, shape
import xgboost as xgb
from definitions import LIGHTNING_PATH, THUNDER_PATH
tf.get_logger().setLevel('INFO')
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import pandas as pd

# Variables ########################################
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
print('essa')

model = xgb.XGBClassifier()
model.load_model("xgb_100.json")  # add variable path later


# Functions #####################################################################################

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # conversion to piixels

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), 1)  # bylo -1


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)


# MAIN ####################################################################################


interpreter = tf.lite.Interpreter(model_path=THUNDER_PATH)
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # mozna dac video zamiast 0
# capig = tf.io.read_file("zjecie.jpg")
# capig = tf.image.decode_jpeg(capig)
# cap = tf.expand_dims(capig, axis=0)
# cap = tf.image.resize_with_pad(cap, 192, 192)

while cap.isOpened():  # while webcam is still connected
    ret, frame = cap.read()  # read the frame from the webcam

    # Reshape for MovNet model 192 x 192 (lightning) or 256x256 (thunder)
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()  # make pediction
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores)

    # Rendering
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    # Normalisation

    X = keypoints_with_scores
    X = np.squeeze(X)

    conf = X[:, 2:3]  # sclicing confidence
    conf = conf.flatten()

    X = X[:, :2]
    X = X.flatten()
    X = X.reshape((1, -1))  # (1,34)
    # print(X)
    # print(type(X))
    # print(shape(X))
    output = (model.predict(X))
    print(all(x < 0.4 for x in conf))
    if all(x < 0.4 for x in conf):
        text = " "
    elif output == 1:
        text = "STANDING"
    elif output == 0:
        text = "SITTING"
    elif output == 2:
        text = "LYING"

    cv2.putText(img=frame, text=text, org=(200, 70), fontFace=1, fontScale=5, color=(0, 255, 0), thickness=2)
    #cv2.putText(img=frame, text=str(output), org=(100, 100), fontFace=1, fontScale=5, color=(0, 255, 0), thickness=2)

    cv2.imshow('MoveNet Lightning', frame)  # render to the screen ('name of window), frame - rozmiar

    if cv2.waitKey(10) & 0xFF == ord('q'):  # wyłączenie
        break

cap.release()
cv2.destroyAllWindows()

# right_eye = keypoints_with_scores[0][0][2]
# left_elbow = keypoints_with_scores[0][0][7]
shaped = np.squeeze(np.multiply(interpreter.get_tensor(interpreter.get_output_details()[0]['index']), [480, 640, 1]))

for kp in shaped:
    ky, kx, kp_conf = kp
    # print(int(ky), int(kx), kp_conf)

shaped[0], shaped[1]

for edge, color in EDGES.items():
    p1, p2 = edge
    y1, x1, c1 = shaped[p1]
    y2, x2, c2 = shaped[p2]
    # print((int(x2), int(y2)))

# IMPORT #########################################################################
import time
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from definitions import THUNDER_PATH, ROOT_DIR

# Variables ########################################
label = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow',
         'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle',
         'right ankle']
KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
                  (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
                  (13, 15), (12, 14), (14, 16)]
width = 640
height = 640
x_points = []
y_points = []
image_name = 1
number_of_images = 70

pose = 'sitting'
# pose = 'standing'
# pose = 'lying'

image_path = f"learning_data/{pose}/{image_name}.jpg"
screen_path = f"screens\{pose}"
# Plotting #####################################################################################
plt.title(image_path, loc='left', fontdict={'family': 'serif', 'color': 'green', 'size': 15})
plt.xlabel("X pixels")
plt.ylabel("Y pixels")
plt.gca().invert_yaxis()


# Functions #####################################################################################
def draw_keypoints(keypoints, confidence_threshold):
    for keypoint in keypoints[0][0]:
        x = int(keypoint[1] * width)
        y = int(keypoint[0] * height)
        c = (keypoint[2])
        if c > confidence_threshold:
            cv2.circle(image_np, (x, y), 4, (0, 0, 255), 1)
            x_points.append(int(keypoint[1] * width))
            y_points.append(int(keypoint[0] * height))
            plt.plot(int(keypoint[1] * width), int(keypoint[0] * height), linestyle='none', marker='*',
                     label='label[i]')


def draw_edges(keypoints):
    for edge in KEYPOINT_EDGES:
        x1 = int(keypoints[0][0][edge[0]][1] * width)
        y1 = int(keypoints[0][0][edge[0]][0] * height)

        x2 = int(keypoints[0][0][edge[1]][1] * width)
        y2 = int(keypoints[0][0][edge[1]][0] * height)
        cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 1)


def counting():
    path, dirs, files = next(os.walk(f"learning_data/{pose}/"))
    file_count = len(files)
    print(f"Collecting Data from {file_count} images")
    return file_count


# MAIN ####################################################################################

interpreter = tf.lite.Interpreter(model_path=THUNDER_PATH)
interpreter.allocate_tensors()

number_of_images = counting()
for sample in range(1, number_of_images + 1):
    image_path = f"learning_data/{pose}/{sample}.jpg"
    image_read = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_read)
    # normalisation
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)

    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # invoking
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()

    # extract keypoints
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    print(f'photo {str(sample)}: ')
    print(keypoints)

    # saving
    np.save(f'learning_data/{pose}_npy/{sample}', keypoints)

    # going back to print it
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, width, height)
    input_image = tf.cast(input_image, dtype=tf.uint8)

    image_np = np.squeeze(input_image.numpy(), axis=0)
    image_np = cv2.resize(image_np, (width, height))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    draw_keypoints(keypoints, 0.0)
    draw_edges(keypoints)

    cv2.putText(img=image_np, text=str(sample), org=(320, 70), fontFace=1, fontScale=5, color=(0, 255, 0), thickness=2)
    cv2.imshow("COLLECTING DATA", image_np)
    os.chdir(screen_path)
    cv2.imwrite(f'{sample}.jpg', image_np)
    os.chdir(ROOT_DIR)
    cv2.waitKey(delay=10)
    # plt.show()

print(f"{str(sample)} samples has been collected!")

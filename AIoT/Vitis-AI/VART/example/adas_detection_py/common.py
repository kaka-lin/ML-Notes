import os
import math
import argparse
import colorsys
import random

import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_file', default="video/adas.webm", type=str)
    parser.add_argument(
        '--model_file', default="/usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel", type=str)
    parser.add_argument('--yolo_runner', default=4, type=int)

    return parser

####################################################################################


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''

    h, w, _ = image.shape
    #print(f"Origin shape: ({w}, {h})")
    desired_w, desired_h = size
    scale = min(desired_w/w, desired_h/h)
    #print(f"Scale rarion: {scale}")

    new_w, new_h = int(w * scale), int(h * scale)
    #print(f"The shape after scaled: ({new_w}, {new_h})")

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((desired_h, desired_w, 3), np.uint8) * 128

    # Put the image that after resized into the center of new image
    # 將縮放後的圖片放入新圖片的正中央
    h_start = (desired_h - new_h) // 2
    w_start = (desired_w - new_w) // 2
    new_image[h_start:h_start+new_h, w_start:w_start+new_w, :] = image

    return new_image


def preprocess_one_image_fn(image, width=416, height=416):
    """pre-process for yolo"""

    image = letterbox_image(image, (width, height))
    image = image / 255.
    return image


def load_classes(filename):
    with open(filename) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


# Scale boxes back to original image shape
def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height, width = image_shape
    image_dims = np.stack([width, height, width, height]).astype(np.float32)
    image_dims = np.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def draw_outputs(image, outputs, class_names, colors):
    h, w, _ = image.shape
    scores, boxes, classes = outputs
    boxes = scale_boxes(boxes, (h, w))

    for i in range(scores.shape[0]):
        left, top, right, bottom = boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        class_id = int(classes[i])
        predicted_class = class_names[class_id]
        score = scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        # colors: RGB
        cv2.rectangle(image, (left, top), (right, bottom),
                      tuple(colors[class_id]), 1)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        label_size = cv2.getTextSize(
            label, font_face, font_scale, font_thickness)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(
            left + 3 + label_size[0]), int(top - 5 - label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),
                      tuple(colors[class_id]), -1)

        cv2.putText(image, label, (left, int(top - 4)),
                    font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return image

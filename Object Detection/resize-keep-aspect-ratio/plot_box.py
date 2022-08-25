import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import parse_annotation


def plot_box(image, bbox, save=True, show=False):
    left = int(bbox[0])
    top = int(bbox[1])
    right = int(bbox[2])
    bottom = int(bbox[3])

    # plot box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)

    # show image
    if show:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save:
        cv2.imwrite('data/street_bbox.jpg', image)


if __name__ == "__main__":
    image_path = "data/street.jpg"
    annotations_path = "data/street.xml"

    # read image and get width, height
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # get bounding box
    bbox = parse_annotation(annotations_path)

    # plot boinding box and save image
    plot_box(image, bbox)

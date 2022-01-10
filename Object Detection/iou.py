import numpy as np


def calculate_iou(bbox1, bbox2):
    """
    calculate iou
    args:
    - bbox1 [array]: 1x4 single bbox
    - bbox2 [array]: 1x4 single bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    xmin = max(bbox1[0], bbox2[0]) # x_left
    ymin = max(bbox1[1], bbox2[1]) # y_top
    xmax = min(bbox1[2], bbox2[2]) # x_right
    ymax = min(bbox1[3], bbox2[3]) # y_bottom

    intersection = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = bbox1_area + bbox2_area - intersection
    return intersection / union


if __name__ == "__main__":
    bbox1 = np.array([661, 27, 679, 47])
    bbox2 = np.array([662, 27, 682, 47])
    iou = calculate_iou(bbox1, bbox2)
    print(iou)

    bbox1 = np.array([0, 0, 100, 100])
    bbox2 = np.array([101, 101, 200, 200])
    iou = calculate_iou(bbox1, bbox2)
    print(iou)

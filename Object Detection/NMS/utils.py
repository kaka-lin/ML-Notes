import numpy as np


def calculate_iou(bbox1, bbox2):
    """
    Calculate IOU
    Args:
        gt_bbox [ndarray]: 1x4 single gt bbox
        pred_bbox [ndarray]: Nx4 single pred bbox
    Returns:
        iou [float]: iou between 2 bboxes
    """
    xmin = max(bbox1[0], bbox2[0]) # x_left
    ymin = max(bbox1[1], bbox2[1]) # y_top
    xmax = min(bbox1[2], bbox2[2]) # x_right
    ymax = min(bbox1[3], bbox2[3]) # y_bottom

    intersection = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union = bbox1_area + bbox2_area - intersection
    return intersection / union

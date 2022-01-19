import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def calculate_iou(bbox1, bbox2):
    """
    Calculate IOU
    Args:
        gt_bbox [ndarray]: 1x4 single gt bbox (bbox1)
        pred_bbox [ndarray]: 1x4 single pred bbox (bbox2)
    Returns:
        iou [float]: iou between 2 bboxes
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


def calc_map(preds, truth, plot=True):
    """Calculate mAP with `Area under curve (AUC)` """
    # 1. create precision - recall curve
    total_object = len(truth['boxes'])
    TP = 0
    curve = []

    for idx, pred in enumerate(preds):
        for gt_bbox in truth['boxes']:
            iou = calculate_iou(gt_bbox, pred[1])
            if iou > 0.5 and pred[0] == 1: # class 1
                TP += 1
        precision = TP / (idx+1)
        recall = TP / total_object
        curve.append([precision, recall])

    # 2. smooth PR curve
    curve = np.array(curve)
    ct = Counter(curve[:, 1])
    # find the boundaries of rectangular blocks.
    # that whenever it drops
    boundaries = sorted([k for k, v in ct.items() if v > 1])
    # get max precision values
    maxes = []
    for i in range(len(boundaries)):
        # not the last boundary of dropping (不是最後一條轉折邊界)
        if i != len(boundaries) - 1:
            loc = [p[0] for p in curve if boundaries[i+1] >= p[1] > boundaries[i]]
            maxes.append(np.max(loc))
        else:
            loc = [p[0] for p in curve if p[1] > boundaries[i]]
            maxes.append(np.max(loc))

    smoothed = curve.copy() # deep copy
    replace = -1
    for i in range(smoothed.shape[0]-1):
        if replace != -1:
            smoothed[i, 0] = maxes[replace]
        if smoothed[i, 1] == smoothed[i+1, 1]:
            replace += 1

    if plot:
        plt.plot(curve[:, 1], curve[:, 0], linewidth=4)
        plt.plot(smoothed[:, 1], smoothed[:, 0], linewidth=4)
        plt.xlabel('recall', fontsize=18)
        plt.ylabel('precision', fontsize=18)
        plt.show()

    # 3. calculate mAP
    cmin = 0
    mAP = 0
    for i in range(smoothed.shape[0] - 1):
        if smoothed[i, 1] == smoothed[i+1, 1]:
            mAP += (smoothed[i, 1] - cmin) * smoothed[i, 0]
            cmin = smoothed[i, 1]
    mAP += (smoothed[-1, 1] - cmin) * smoothed[-1, 0]
    return mAP


if __name__ == "__main__":
    # load data
    with open('data/predictions.json', 'r') as f:
        preds = json.load(f)

    with open('data/ground_truths.json', 'r') as f:
        ground_truth = json.load(f)

    # processing predictions
    boxes = preds[0]['boxes']
    scores = preds[0]['scores']
    classes = preds[0]['classes']
    predictions = [(clas, box, score) for clas, box, score in zip(classes, boxes, scores)]
    predictions = sorted(predictions, key=lambda k:k[-1])[::-1]

    mAP = calc_map(predictions, ground_truth[0])
    # in sample, correct mAP is 0.7268
    round_output = np.round(mAP * 1e4) / 1e4
    assert round_output == 0.7286, 'Something is wrong with the mAP calculation'
    print('mAP calculation is correct!')

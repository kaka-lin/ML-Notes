import json

import numpy as np

from utils import calculate_iou


def py_cpu_nms(boxes, scores, max_boxes=100, iou_thresh=0.5):
    """Pure Python NMS baseline.

    Arguments:
        boxes:      shape of [-1, 4]
        scores:     shape of [-1,]
        max_boxes:  representing the maximum of boxes
                    to be selected by non_max_suppression
        iou_thresh: representing iou_threshold
                    for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        inds = []
        for j in range(1, order.size):
            cmp_j = order[j]
            iou = calculate_iou(boxes[i], boxes[cmp_j])
            if iou <= iou_thresh:
                inds.append(j)
        order = order[inds]

    return keep[:max_boxes]


if __name__ == "__main__":
    with open('test_data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)

    boxes = np.array(predictions['boxes'], dtype=np.float32)
    scores = np.array(predictions['scores'], dtype=np.float32)

    selected_indices = py_cpu_nms(boxes, scores)
    new_boxes = np.take(boxes, selected_indices, axis=0)
    new_scores = np.take(scores, selected_indices, axis=0)
    #output = np.column_stack((new_boxes, new_scores))

    truth = np.load('test_data/nms.npy', allow_pickle=True)
    truth_boxes = np.array([box for box in truth[:, 0]], dtype=np.float32)
    truth_scores= np.array([score for score in truth[:, 1]], dtype=np.float32)

    assert np.array_equal(truth_boxes, new_boxes) and \
           np.array_equal(truth_scores, new_scores), 'The NMS implementation is wrong'
    print('The NMS implementation is correct!')

import numpy as np


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

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # The area of every box
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        # keep the highest score of this class
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(overlap <= iou_thresh)[0]
        # because overlap is start from index 1 of order
        order = order[inds + 1]

    return keep[:max_boxes]

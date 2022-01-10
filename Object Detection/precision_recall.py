import numpy as np


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i, j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


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


def precision_recall(ious, gt_classes, pred_classes):
    """
    calculate precision and recall
    args:
    - ious [array]: NxM array of ious
    - gt_classes [array]: 1xN array of ground truth classes
    - pred_classes [array]: 1xM array of pred classes
    returns:
    - precision [float]
    - recall [float]
    """
    # IMPLEMENT THIS FUNCTION
    TP, FP, FN = 0, 0, 0

    # Using a threshold of 0.5 IoU to determine if a prediction is a true positive or not.
    # x, y = np.where(ious>0.5)
    pt_idxs = [] # prediction positive
    for n in range(len(gt_classes)):
        for m in range(len(pred_classes)):
            if ious[n][m] > 0.5:
                pt_idxs.append((n, m))

    # calculate true positive and false positive
    all_pos_list = set()
    for i, j in pt_idxs:
        all_pos_list.add(i)
        if (gt_classes[i] == pred_classes[j]):
            TP += 1
        else:
            FP += 1

    # calculate false negative
    FN = len(gt_classes) - len(all_pos_list)

    # calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall


if __name__ == "__main__":
    gt_bboxes = np.array([[793, 1134, 1001, 1718],
                          [737, 0, 898, 260],
                          [763, 484, 878, 619],
                          [734, 0, 1114, 277],
                          [853, 0, 1280, 250],
                          [820, 1566, 974, 1914],
                          [762, 951, 844, 1175],
                          [748, 197, 803, 363]])

    gt_classes = np.array([1, 1, 1, 1, 2, 1, 1, 1])

    pred_boxes = np.array([[783, 1104, 1011, 1700],
                           [853, 0, 1220, 200],
                           [734, 0, 1100, 240],
                           [753, 474, 868, 609],
                           [830, 1500, 1004, 1914]])
    pred_classes = np.array([1, 2, 1, 2, 1])

    ious = calculate_ious(gt_bboxes, pred_boxes)
    precision, recall = precision_recall(ious, gt_classes, pred_classes)
    print("IOU: \n{}".format(ious))
    print("precision: {}, recall: {}".format(precision, recall))

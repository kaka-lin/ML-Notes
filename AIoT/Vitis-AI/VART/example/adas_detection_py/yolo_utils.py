import numpy as np

from common import sigmoid

# anchor boxes
# (width, height)
yolo_anchors = np.array([
    (347, 98), (167, 83), (165, 158), (123, 100), (98, 174),
    (105, 63), (76, 37), (74, 64), (66, 131), (40, 97),
    (52, 42), (47, 23), (33, 29), (28, 68), (18, 46),
    (24, 17), (14, 11), (13, 29), (8, 17), (5.5, 7)], np.float32) / [512, 256]

yolo_anchor_masks = np.array([[0, 1, 2, 3, 4],
                              [5, 6, 7, 8, 9],
                              [10, 11, 12, 13, 14],
                              [15, 16, 17, 18, 19]])


def yolo_boxes(pred, anchors, classes):
    """YOLO bounding box formula
    bx = sigmoid(tx) + cx
    by = sigmoid(ty) + cy
    bw = pw * exp^(tw)
    bh = ph * exp^(th)
    Pr(obj) * IOU(b, object) = sigmoid(to) # confidence
    (tx, ty, tw, th, to) are the output of the model.
    """
    # pred: (batch_size, grid, grid, anchors * (tx, ty, tw, th, conf, ...classes))
    batch_size, grid_height, grid_width, _ = pred.shape
    pred = np.reshape(pred, (batch_size, grid_height,
                      grid_width, len(anchors), -1))

    box_xy = sigmoid(pred[..., 0:2])
    box_wh = pred[..., 2:4]
    box_confidence = sigmoid(pred[..., 4:5])
    box_class_probs = sigmoid(pred[..., 5:])

    # box_xy: (grid_size, grid_size, num_anchors, 2)
    # grid: (grdid_siez, grid_size, 1, 2)
    #       -> [0,0],[0,1],...,[0,12],[1,0],[1,1],...,[12,12]
    # !!! grid[x][y] == (y, x)
    grid = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    grid = np.expand_dims(np.stack(grid, axis=-1),
                          axis=2).astype(int)  # [gx, gy, 1, 2]

    box_xy = (box_xy + grid) / [grid_width, grid_height]
    box_wh = np.exp(box_wh) * anchors
    pred_box = np.concatenate((box_xy, box_wh), axis=-1)

    return pred_box, box_confidence, box_class_probs


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

###########################################################################################
# Post-processing


def yolo_eval(yolo_outputs,
              image_shape=(416, 416),
              anchors=yolo_anchors,
              classes=80,
              max_boxes=100,
              score_threshold=0.5,
              iou_threshold=0.5):
    # Retrieve outputs of the YOLO model.
    for i in range(0, 4):
        _boxes, _box_scores = yolo_boxes_and_scores(
            yolo_outputs[i], anchors[i*5:i*5+5], classes)

        if i == 0:
            boxes, box_scores = _boxes, _box_scores
        else:
            boxes = np.concatenate([boxes, _boxes], axis=0)
            box_scores = np.concatenate([box_scores, _box_scores], axis=0)

    # # Perform Score-filtering and Non-max suppression
    scores, boxes, classes = yolo_non_max_suppression(boxes, box_scores,
                                                      classes,
                                                      max_boxes,
                                                      score_threshold,
                                                      iou_threshold)

    return scores, boxes, classes


def yolo_boxes_and_scores(yolo_output, anchors=yolo_anchors, classes=80):
    """Process output layer"""
    # yolo_boxes: pred_box, box_confidence, box_class_probs
    pred_box, box_confidence, box_class_probs = yolo_boxes(
        yolo_output, anchors, classes)

    # Convert boxes to be ready for filtering functions.
    # Convert YOLO box predicitions to bounding box corners.
    # (x, y, w, h) -> (x1, y1, x2, y2)
    box_xy = pred_box[..., 0:2]
    box_wh = pred_box[..., 2:4]
    box_x1y1 = box_xy - (box_wh / 2.)
    box_x2y2 = box_xy + (box_wh / 2.)
    boxes = np.concatenate([box_x1y1, box_x2y2], axis=-1)
    boxes = np.reshape(boxes, [-1, 4])

    # Compute box scores
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes])
    return boxes, box_scores


def yolo_non_max_suppression(boxes, box_scores,
                             classes=80,
                             max_boxes=100,
                             score_threshold=0.5,
                             iou_threshold=0.5):
    """Perform Score-filtering and Non-max suppression
    boxes: (54400, 4)
    box_scores: (54400, 3)
    anchor box: 5
    # 54400 = (8*16 + 16*32 + 32*64 + 64*128) * 5
    """

    # Create a mask, same dimension as box_scores.
    mask = box_scores >= score_threshold  # (54400, 3)

    output_boxes = []
    output_scores = []
    output_classes = []

    # Perform NMS for all classes
    for c in range(classes):
        # class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_boxes = boxes[mask[:, c]]
        #  class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        class_box_scores = box_scores[:, c][mask[:, c]]

        selected_indices = py_cpu_nms(
            class_boxes, class_box_scores, max_boxes, iou_threshold)

        class_boxes = np.take(class_boxes, selected_indices, axis=0)
        class_box_scores = np.take(class_box_scores, selected_indices, axis=0)

        classes = np.ones_like(class_box_scores, 'int32') * c

        output_boxes.append(class_boxes)
        output_scores.append(class_box_scores)
        output_classes.append(classes)

    output_boxes = np.concatenate(output_boxes, axis=0)
    output_scores = np.concatenate(output_scores, axis=0)
    output_classes = np.concatenate(output_classes, axis=0)

    return output_scores, output_boxes, output_classes

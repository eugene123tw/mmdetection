__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import base64
import typing as t
import warnings
import zlib

import cv2
import numpy as np
from pycocotools import _mask as coco_mask


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            'encode_binary_mask expects a binary mask, received dtype == %s' %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            'encode_binary_mask expects a 2d mask, received shape == %s' %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]['counts']

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def prefilter_predictions(boxes, masks, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_predictions = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print(
                'Error. Length of boxes arrays not equal to length of scores array: {} != {}'
                .format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print(
                'Error. Length of boxes arrays not equal to length of labels array: {} != {}'
                .format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            mask_part = masks[t][j]
            x1 = box_part[0]
            y1 = box_part[1]
            x2 = box_part[2]
            y2 = box_part[3]

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1

            x1 = np.maximum(x1, 0.0)
            y1 = np.maximum(y1, 0.0)
            x2 = np.minimum(x2, 1.0)
            y2 = np.minimum(y2, 1.0)

            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn('Zero area box skipped: {}.'.format(box_part))
                continue

            # [label, score, weight, model_index, x1, y1, x2, y2, mask array]
            b = [
                int(label),
                float(score) * weights[t], weights[t], t, x1, y1, x2, y2,
                mask_part
            ]
            if label not in new_predictions:
                new_predictions[label] = []
            new_predictions[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_predictions:
        current_boxes = np.array(new_predictions[k])
        new_predictions[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_predictions


def resize_mask(mask, w, h):
    mask = np.pad(mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    mask = cv2.resize(mask.astype(np.float32), (w, h))
    return mask


def get_weighted_box(boxes, conf_type='avg'):
    """Create weighted box for set of boxes.

    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2)
    """

    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        b = b.astype(np.float32)
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """Reimplementation of find_matching_box with numpy instead of loops.

    Gives significant speed up for larger arrays (~100x). This was previously
    the bottleneck since the function is called for every entry in the array.
    """

    def bb_iou_array(boxes, new_box):
        # bb intersection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:], new_box[4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_mask_fusion(
    weighted_boxes,
    box_cluster,
    mask_cluster,
    img_h,
    img_w,
):
    assert len(weighted_boxes) == len(box_cluster) == len(mask_cluster)
    weighted_masks = []
    keep_idx = []
    for i, (weighted_box, boxes, masks) in enumerate(
            zip(weighted_boxes, box_cluster, mask_cluster)):
        fused_mask = np.zeros((img_h, img_w), dtype=np.float32)
        num_samples = len(boxes)
        for box, mask in zip(boxes, masks):
            score = box[1]
            x1, y1, x2, y2 = box[4:]
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)
            mask = resize_mask(mask, x2 - x1, y2 - y1)
            # fused_mask[y1:y2, x1:x2] += mask * score
            fused_mask[y1:y2, x1:x2] += mask
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        fused_mask = cv2.dilate(fused_mask, kernel, 3)
        # NOTE: we can use weighted box score as threshold or we can use 0.5 as threshold
        # NOTE: what is the best threshold?
        bit_mask = (fused_mask / num_samples) >= 0.01
        if bit_mask.sum() == 0:
            warnings.warn('Zero area mask skipped.')
        else:
            keep_idx.append(i)
        encoded = encode_binary_mask(bit_mask.astype(bool))
        weighted_masks.append(encoded)
    return weighted_masks, keep_idx


def weighted_boxes_masks_fusion(boxes_list,
                                mask_list,
                                scores_list,
                                labels_list,
                                img_h,
                                img_w,
                                weights=None,
                                iou_thr=0.55,
                                skip_box_thr=0.0,
                                conf_type='avg',
                                allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: masks: encoded masks for each box
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print(
            'Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'
            .format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in [
            'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'
    ]:
        print(
            'Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'
            .format(conf_type))
        exit()

    filtered_predictions = prefilter_predictions(boxes_list, mask_list,
                                                 scores_list, labels_list,
                                                 weights, skip_box_thr)
    if len(filtered_predictions) == 0:
        return np.zeros((0, 4)), np.zeros((0, )), np.zeros((0, )), []

    overall_boxes = []
    for label in filtered_predictions:
        predictions = filtered_predictions[label]
        new_boxes = []
        new_masks = []
        weighted_boxes = np.empty((0, 8))

        # Clusterize boxes
        for j in range(0, len(predictions)):
            index, best_iou = find_matching_box_fast(weighted_boxes,
                                                     predictions[j], iou_thr)

            if index != -1:
                new_boxes[index].append(predictions[j][:-1])
                new_masks[index].append(predictions[j][-1])
                weighted_boxes[index] = get_weighted_box(
                    new_boxes[index], conf_type)
            else:
                new_masks.append([predictions[j][-1]])
                new_boxes.append([predictions[j][:-1]])
                weighted_boxes = np.vstack(
                    (weighted_boxes, predictions[j][:-1]))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(
                    clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes[
                    idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[
                    i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (
                        weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(
                    len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(
                    clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    encoded_masks, keep_idx = weighted_mask_fusion(overall_boxes, new_boxes,
                                                   new_masks, img_h, img_w)
    scores = overall_boxes[:, 1]

    encoded_masks = [encoded_masks[i] for i in keep_idx]
    scores = scores[keep_idx]

    sorted_idx = np.argsort(scores)[::-1]

    encoded_masks = [encoded_masks[i] for i in sorted_idx]
    scores = scores[sorted_idx]
    # overall_boxes = overall_boxes[sorted_idx]
    # predictions = overall_boxes[:, 4:]
    # labels = overall_boxes[:, 0]
    return encoded_masks, scores

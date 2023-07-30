import base64
import typing as t
import warnings
import zlib
from pathlib import Path

import cv2
import mmcv
import numpy as np
import pandas as pd
from pycocotools import _mask as coco_mask
from skimage import measure

from mmdet.apis import inference_detector, init_detector

BBOX_INDEX = 0
MASK_INDEX = 1
LABLE_INDEX = 0  # ONLY PICK BLOOD VESSEL


def get_wsf_mask(wbf_box, wbf_org, pmasks, pmasks_lkup, thres=0.5):
    # Fuse masks that belong to fused boxes
    w, h = 512, 512
    mask = np.zeros((w, h), dtype=np.uint8)
    for i in range(len(wbf_org)):
        key = bbox_to_key(wbf_org[i][4:])
        model = int(wbf_org[i][3])
        ind = pmasks_lkup[model][key]
        mask = mask + pmasks[model][ind]

    # convert thres to integer based on number of boxes
    threshold = max(1, int(thres * len(wbf_org)))
    # remove pixels outside WBF box
    m2 = np.zeros((w, h), dtype=np.uint8)
    x1 = max(0, int(h * wbf_box[0]))
    y1 = max(0, int(w * wbf_box[1]))
    x2 = min(h, int(h * wbf_box[2]))
    y2 = min(w, int(w * wbf_box[3]))
    m2[y1:y2, x1:x2] = 1
    mask = (mask >= threshold) * m2
    return mask.astype(np.uint8)


def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_weighted_box(boxes, conf_type='avg'):
    """Create weighted box for set of boxes.

    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, x1, y1, x2, y2)
    """

    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[1] = conf / len(boxes)
    box[2] = w
    # model index field is retained for consistency but is not used.
    box[3] = -1
    box[4:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[4:], new_box[4:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for n in range(len(boxes)):
        assert len(boxes[n]) == len(scores[n])
        assert len(boxes[n]) == len(labels[n])

        for j in range(len(boxes[n])):
            score = scores[n][j]
            if score < thr:
                continue
            label = int(labels[n][j])
            box_part = boxes[n][j]
            x1 = max(float(box_part[0]), 0.)
            y1 = max(float(box_part[1]), 0.)
            x2 = max(float(box_part[2]), 0.)
            y2 = max(float(box_part[3]), 0.)

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 > 1:
                warnings.warn(
                    'X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.'
                )
                x1 = 1
            if x2 > 1:
                warnings.warn(
                    'X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.'
                )
                x2 = 1
            if y1 > 1:
                warnings.warn(
                    'Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.'
                )
                y1 = 1
            if y2 > 1:
                warnings.warn(
                    'Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.'
                )
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn('Zero area box skipped: {}.'.format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [
                int(label),
                float(score) * weights[n], weights[n], n, x1, y1, x2, y2
            ]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def bbox_to_key(bbox):
    # create dict key from bbox
    return str(np.round(bbox, 6))


def weighted_boxes_fusion_tracking(boxes_list,
                                   scores_list,
                                   labels_list,
                                   weights=None,
                                   iou_thr=0.55,
                                   skip_box_thr=0.0,
                                   conf_type='avg',
                                   allows_overflow=False):

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    assert conf_type in [
        'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'
    ]

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list,
                                     weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0, )), np.zeros((0, )), np.zeros(
            (0, 4))

    overall_boxes = []
    original_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []
        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j],
                                                iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(
                    new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())
        # Rescale confidence based on number of models and boxes
        original_boxes.append(new_boxes)
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == 'box_and_model_avg':
                # weighted average for boxes
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(
                    clustered_boxes) / weighted_boxes[i][2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i][1] = weighted_boxes[i][1] * clustered_boxes[
                    idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(
                    clustered_boxes) / (
                        weighted_boxes[i][2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i][1] = weighted_boxes[i][1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(
                    len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(
                    clustered_boxes) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    sidx = overall_boxes[:, 1].argsort()
    overall_boxes = overall_boxes[sidx[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    # sort originals according to wbf
    original_boxes = original_boxes[0]
    wbfo = [original_boxes[i] for i in sidx[::-1]]
    return boxes, scores, labels, wbfo


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


def init_model(model_dict):
    config_file = model_dict['config']
    ckpt = model_dict['ckpt']

    config = mmcv.Config.fromfile(config_file)
    if config.data.test.pipeline[1].type == 'MultiScaleFlipAug':
        tta_pipeline = config.data.test.pipeline[1]
        tta_pipeline.flip = True
        tta_pipeline.flip_direction = ['horizontal', 'vertical']

    detector = init_detector(config, ckpt, device='cuda:0')
    return detector


def format_results(masks, scores, score_thr=0.001, kernel_size=3):
    num_predictions = masks.shape[0]
    n = 0
    pred_string = ''
    for i in range(num_predictions):
        mask = masks[i]
        score = scores[i]
        if score >= score_thr and mask.sum() > 32:
            # NOTE: add dilation to make the mask larger
            mask = mask.astype(np.uint8)
            kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)
            bitmask = cv2.dilate(mask, kernel, 3)
            bitmask = bitmask.astype(bool)

            encoded = encode_binary_mask(bitmask)
            if n == 0:
                pred_string += f"0 {score} {encoded.decode('utf-8')}"
            else:
                pred_string += f" 0 {score} {encoded.decode('utf-8')}"
            n += 1
    return pred_string


def hubmap_ensemble(model_list,
                    image_root,
                    iou_thr=0.5,
                    score_thr=0.001,
                    max_num=100):
    fnames = list(Path(image_root).glob('*.tif'))

    ids = []
    heights = []
    widths = []
    prediction_strings = []

    detectors = [init_model(model_dict) for model_dict in model_list]

    for fname in fnames:
        pred_string = ''
        h, w, c = cv2.imread(str(fname)).shape
        aug_boxes = []
        aug_scores = []
        aug_labels = []
        aug_masks = []
        mask_lookup = []
        for detector in detectors:
            results = inference_detector(detector, str(fname))
            bboxes, masks = results
            masks = np.stack(masks[LABLE_INDEX])
            boxes = bboxes[LABLE_INDEX][:, :4] / 512.0
            scores = bboxes[LABLE_INDEX][:, 4]
            num_pred = len(bboxes[LABLE_INDEX])

            aug_boxes.append(boxes)
            aug_scores.append(scores)
            aug_labels.append(len(masks) * [0])
            aug_masks.append(masks)
            pred_dict = {}
            for k in range(num_pred):
                pred_dict[bbox_to_key(boxes[k])] = k
            mask_lookup.append(pred_dict)

        # weighted boxes fusion
        wbf_boxes, wbf_scores, _, wbf_originals = weighted_boxes_fusion_tracking(
            aug_boxes,
            aug_scores,
            labels_list=aug_labels,
            iou_thr=0.55,
            skip_box_thr=0.01)

        # Finally, process masks, making sure there is no overlap
        used = np.zeros((512, 512), dtype=int)

        # process
        n = 0
        for j in range(len(wbf_boxes)):
            mask_score = wbf_scores[j]
            mask = get_wsf_mask(
                wbf_boxes[j],
                wbf_originals[j],
                aug_masks,
                mask_lookup,
                thres=0.5)
            # get shape properties
            props = measure.regionprops(mask)

            # if there are multiple separated masks, pick the larger one
            areas = []
            for a in range(len(props)):
                areas.append(props[a].area)

            target = np.argmax(areas)
            mask2 = np.zeros((512, 512), dtype=int)

            y1 = props[target].bbox[0]
            x1 = props[target].bbox[1]
            y2 = props[target].bbox[2]
            x2 = props[target].bbox[3]

            mask2[y1:y2, x1:x2] = props[target].filled_image
            # extract properties of interest
            major_axis_len = props[target].major_axis_length
            # check against limits
            if major_axis_len >= 32:
                mask2 = mask2 * (1 - used)
                # check if mask is chopped up by previous detections
                if len(
                        measure.find_contours(
                            mask2, 0.5, positive_orientation='low')) == 1:
                    used += mask2
                    mask2 = mask2.astype(np.uint8)
                    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
                    bitmask = cv2.dilate(mask, kernel, 3)
                    bitmask = bitmask.astype(bool)
                    encoded = encode_binary_mask(bitmask)
                    if n == 0:
                        pred_string += f"0 {mask_score} {encoded.decode('utf-8')}"
                    else:
                        pred_string += f" 0 {mask_score} {encoded.decode('utf-8')}"
                    n += 1
        ids.append(fname.stem)
        heights.append(h)
        widths.append(w)
        prediction_strings.append(pred_string)

    return ids, prediction_strings, heights, widths


if __name__ == '__main__':
    image_root = '/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature/test'

    model_list = [
        {
            'config':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s6_cls1/solov2_x101_dcn_fpn_hubmap_s6_cls1.py',
            'ckpt':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s6_cls1/best_segm_mAP_epoch_9.pth'
        },
        {
            'config':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s7_cls1/solov2_x101_dcn_fpn_hubmap_s7_cls1.py',
            'ckpt':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s7_cls1/best_segm_mAP_epoch_7.pth'
        },
    ]

    ids, prediction_strings, heights, widths = hubmap_ensemble(
        model_list, image_root, iou_thr=0.6, score_thr=0.01, max_num=100)
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    submission.to_csv('submission.csv')

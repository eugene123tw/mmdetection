import base64
import os
import typing as t
import zlib
from pathlib import Path

import cv2
import mmcv
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from ensemble_boxes_wbf import weighted_boxes_fusion
from pycocotools import _mask as coco_mask

from mmdet.apis import inference_detector, init_detector
from mmdet.core import encode_mask_results

BBOX_INDEX = 0
MASK_INDEX = 1
LABLE_INDEX = 1  # ONLY PICK BLOOD VESSEL


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


def init_model(model_dict, iou_thr=0.5, score_thr=0.001, max_num=100):
    config_file = model_dict['config']
    ckpt = model_dict['ckpt']

    config = mmcv.Config.fromfile(config_file)
    if 'rcnn' in config.model.test_cfg:
        # Mask RCNN
        config.model.test_cfg.rcnn.nms.iou_threshold = iou_thr
        config.model.test_cfg.rcnn.score_thr = score_thr
        config.model.test_cfg.rcnn.max_per_img = max_num
    else:
        # SOLOv2
        config.model.test_cfg.score_thr = score_thr
        config.model.test_cfg.max_per_img = max_num

    detector = init_detector(config, ckpt, device='cuda:0')
    return detector


def ensemble_masks_from_boxes(fused_boxes,
                              masks,
                              img_h,
                              img_w,
                              mask_score_thres: float = 0.5):
    """Ensemble masks from boxes.

    Args:
        fused_boxes (_type_):
        masks (_type_): _description_
        img_h (_type_): _description_
        img_w (_type_): _description_
        mask_score_thres (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    encoded_string = []
    scores = []
    for i, fused_box in enumerate(fused_boxes):
        fused_mask = np.zeros((img_h, img_w), dtype=bool)
        x1, y1, x2, y2, score = fused_box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        vote_mask = np.zeros((y2 - y1, x2 - x1))
        for j, mask in enumerate(masks):
            # crop mask from Model j
            vote_mask += mask[y1:y2, x1:x2]

        vote_mask = vote_mask / np.max(vote_mask)
        vote_mask = vote_mask > mask_score_thres
        kernel = np.ones(shape=(3, 3), dtype=np.uint8)
        vote_mask = cv2.dilate(vote_mask.astype(np.uint8), kernel, 3)
        fused_mask[y1:y2, x1:x2] = vote_mask
        if np.sum(fused_mask) > 0:
            scores.append(score)
            encoded_string.append(encode_binary_mask(fused_mask))
    return encoded_string, scores


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

    detectors = [init_model(model_dict, iou_thr, score_thr, max_num) for model_dict in model_list]

    for i, fname in enumerate(fnames):
        h, w, c = cv2.imread(str(fname)).shape
        pred_string = ''
        img_dim = np.array([w, h, w, h])
        aggregate_boxes = []
        aggregate_scores = []
        aggregate_labels = []
        aggregate_masks = []
        for dectector in detectors:
            result = inference_detector(dectector, str(fname))
            blood_vessel_bboxes = result[BBOX_INDEX][LABLE_INDEX][:, :4]
            blood_vessel_scores = result[BBOX_INDEX][LABLE_INDEX][:, -1]
            blood_vessel_masks = result[MASK_INDEX][LABLE_INDEX]

            # NOTE: normalize bbox to [0, 1]
            aggregate_boxes.append(blood_vessel_bboxes / img_dim)
            aggregate_scores.append(blood_vessel_scores)
            aggregate_labels.append(len(blood_vessel_bboxes) * [0])
            aggregate_masks.extend(blood_vessel_masks)
        fused_box, fused_score, fused_label = weighted_boxes_fusion(
            aggregate_boxes,
            aggregate_scores,
            aggregate_labels,
            weights=None,
            iou_thr=iou_thr,
            skip_box_thr=0.0001)
        fused = np.concatenate(
            [fused_box * img_dim,
             fused_score.reshape(-1, 1)], axis=1)
        # decode masks and stack masks and take mean
        # crop masks from bounding boxes
        encoded_strings, scores = ensemble_masks_from_boxes(fused, aggregate_masks, h, w)
        assert len(fused) == len(encoded_strings)
        n = 0
        for score, encoded in zip(scores, encoded_strings):
            if score > score_thr:
                if n == 0:
                    pred_string += f"0 {score} {encoded.decode('utf-8')}"
                else:
                    pred_string += f" 0 {score} {encoded.decode('utf-8')}"
                n += 1
        ids.append(str(os.path.basename(fname)).split('.')[0].split('/')[-1])
        heights.append(h)
        widths.append(w)
        prediction_strings.append(pred_string)

    return ids, prediction_strings, heights, widths


if __name__ == '__main__':
    image_root = '/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature/train'

    model_list = [
        {
            'name': 'solov2_x101_dcn_fpn_hubmap_s5_best',
            'config': '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy5/mask_rcnn_r50_fpn_giou_loss_strategy5.py',
            'ckpt':'/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy5/epoch_5.pth'
        },
        {
            'name': 'solov2_x101_dcn_fpn_hubmap_s5_best',
            'config': '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy5/mask_rcnn_r50_fpn_giou_loss_strategy5.py',
            'ckpt':'/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy5/epoch_1.pth'
        },
    ]

    ids, prediction_strings, heights, widths = hubmap_ensemble(
        model_list, image_root, iou_thr=0.6, score_thr=0.001, max_num=200)
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    # submission.to_csv("/kaggle/working/mmdetection/submission.csv")
    # submission.to_csv("submission.csv")

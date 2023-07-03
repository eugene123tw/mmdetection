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
from mmcv.ops import nms
from pycocotools import _mask as coco_mask

from mmdet.apis import inference_detector, init_detector

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


def ensemble_masks(boxes,
                   scores,
                   masks,
                   iou_thr=0.6,
                   score_thr=0.001,
                   max_num=100,
                   mask_score_thres=0.5):
    """Ensemble masks from boxes.

    Args:
        fused_boxes (_type_):
        masks (_type_): _description_
        scores (_type_): _description_
        iou_thr (float): iou_thr
        mask_score_thres (float, optional): _description_. Defaults to 0.5.
        max_num (int):

    Returns:
        _type_: _description_
    """
    encoded_string = []
    ensemble_scores = []

    dets, inds = nms(
        boxes,
        scores,
        iou_threshold=iou_thr,
        score_threshold=score_thr,
        max_num=1000)

    masks = masks[inds]

    for box, mask in zip(dets, masks):
        if mask.sum() > 0:
            score = box[-1]
            kernel = np.ones(shape=(3, 3), dtype=np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, 3)
            encoded_string.append(encode_binary_mask(mask.astype(bool)))
            ensemble_scores.append(score)
    return encoded_string, ensemble_scores


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

    detectors = [
        init_model(model_dict, iou_thr, score_thr, max_num)
        for model_dict in model_list
    ]

    for i, fname in enumerate(fnames):
        h, w, c = cv2.imread(str(fname)).shape
        pred_string = ''
        aggregate_boxes = []
        aggregate_scores = []
        aggregate_labels = []
        aggregate_masks = []
        for detector in detectors:
            result = inference_detector(detector, str(fname))
            blood_vessel_bboxes = result[BBOX_INDEX][LABLE_INDEX][:, :4]
            blood_vessel_scores = result[BBOX_INDEX][LABLE_INDEX][:, -1]
            blood_vessel_masks = result[MASK_INDEX][LABLE_INDEX]

            aggregate_boxes.append(blood_vessel_bboxes)
            aggregate_scores.append(blood_vessel_scores)
            aggregate_labels.append(len(blood_vessel_bboxes) * [0])
            aggregate_masks.extend(blood_vessel_masks)

        aggregate_boxes = np.vstack(aggregate_boxes)
        aggregate_scores = np.hstack(aggregate_scores)
        aggregate_masks = np.stack(aggregate_masks)

        encoded_strings, scores = ensemble_masks(aggregate_boxes,
                                                 aggregate_scores,
                                                 aggregate_masks, iou_thr,
                                                 score_thr, max_num)
        assert len(encoded_strings) == len(
            scores), 'Length of encoded strings and scores should be the same.'
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
            'name':
            'mask_rcnn_r50_fpn_giou_loss_strategy5',
            'config':
            'work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy5/mask_rcnn_r50_fpn_giou_loss_strategy5.py',
            'ckpt':
            'work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy5/best_segm_mAP_epoch_14.pth'
        },
        {
            'name':
            'point_rend_r50_caffe_fpn_hubmap_s5',
            'config':
            'work_dirs/point_rend_r50_caffe_fpn_hubmap_s5/point_rend_r50_caffe_fpn_hubmap_s5.py',
            'ckpt':
            'work_dirs/point_rend_r50_caffe_fpn_hubmap_s5/best_segm_mAP_epoch_19.pth'
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

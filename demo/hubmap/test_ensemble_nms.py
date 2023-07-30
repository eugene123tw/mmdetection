import base64
import os
import typing as t
import zlib
from pathlib import Path

import cv2
import mmcv
import numpy as np
import pandas as pd
import torch
from pycocotools import _mask as coco_mask

from mmdet.apis import inference_detector, init_detector
from mmdet.core import mask_matrix_nms

BBOX_INDEX = 0
MASK_INDEX = 1
LABLE_INDEX = 0  # ONLY PICK BLOOD VESSEL


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

    for i, fname in enumerate(fnames):
        h, w, c = cv2.imread(str(fname)).shape
        pred_string = ''
        aug_scores = []
        aug_labels = []
        aug_masks = []
        for detector in detectors:
            results = inference_detector(detector, str(fname))
            bboxes, masks = results
            masks = np.stack(masks[LABLE_INDEX])
            scores = bboxes[LABLE_INDEX][:, 4]

            aug_scores.append(scores)
            aug_labels.append(len(masks) * [0])
            aug_masks.extend(masks)

        aug_scores = np.hstack(aug_scores)
        aug_labels = np.hstack(aug_labels)
        aug_masks = np.stack(aug_masks)
        sum_masks = aug_masks.sum((1, 2))

        aug_scores, aug_labels, aug_masks, keep_inds = mask_matrix_nms(
            torch.tensor(aug_masks),
            torch.tensor(aug_labels).long(),
            torch.tensor(aug_scores),
            mask_area=torch.tensor(sum_masks),
            nms_pre=500,
            max_num=100,
            filter_thr=0.001)

        aug_scores = aug_scores.numpy()
        aug_labels = aug_labels.numpy()
        aug_masks = aug_masks.numpy()

        pred_string = format_results(
            aug_masks, aug_scores, score_thr=score_thr)
        ids.append(str(os.path.basename(fname)).split('.')[0].split('/')[-1])
        heights.append(h)
        widths.append(w)
        prediction_strings.append(pred_string)

    return ids, prediction_strings, heights, widths


if __name__ == '__main__':
    image_root = '/home/yuchunli/_DATASET/hubmap-hacking-the-human-vasculature/train'

    model_list = [
        {
            'config':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_0/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_0.py',
            'ckpt':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_0/best_segm_mAP_epoch_9.pth'
        },
        {
            'config':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_1/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_1.py',
            'ckpt':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_1/best_segm_mAP_epoch_17.pth'
        },
        {
            'config':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_2/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_2.py',
            'ckpt':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_2/best_segm_mAP_epoch_11.pth'
        },
        {
            'config':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_3/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_3.py',
            'ckpt':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_3/best_segm_mAP_epoch_17.pth'
        },
        {
            'config':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_4/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_4.py',
            'ckpt':
            'work_dirs/solov2_x101_dcn_fpn_hubmap_s0_cls1_fold_4/best_segm_mAP_epoch_11.pth'
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

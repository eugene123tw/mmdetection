import base64
import glob
import os
import typing as t
import zlib

import cv2
import mmcv
import numpy as np
import pandas as pd
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


def initialize_detector(config,
                        ckpt,
                        iou_thr=0.5,
                        score_thr=0.001,
                        max_num=100):
    if 'rcnn' in config.model.test_cfg:
        # Mask RCNN
        config.model.test_cfg.rcnn.nms.iou_threshold = iou_thr
        config.model.test_cfg.rcnn.score_thr = score_thr
        config.model.test_cfg.rcnn.max_per_img = max_num
    else:
        # SOLOv2
        config.model.test_cfg.score_thr = score_thr
        config.model.test_cfg.max_per_img = max_num

    model = init_detector(config, ckpt, device='cuda:0')
    return model


def hubmap_single_model(model_dict,
                        image_root,
                        iou_thr=0.5,
                        score_thr=0.001,
                        max_num=100):
    ids = []
    heights = []
    widths = []
    prediction_strings = []

    fnames = glob.glob(image_root + '/*.tif')
    config_file = model_dict['config']
    ckpt = model_dict['ckpt']
    config = mmcv.Config.fromfile(config_file)
    detector = initialize_detector(config, ckpt, iou_thr, score_thr, max_num)

    for fname in fnames:
        result = inference_detector(detector, fname)
        id = os.path.splitext(os.path.basename(fname))[0]
        bboxes, masks = result
        bboxes, masks = bboxes[LABLE_INDEX], masks[LABLE_INDEX]

        pred_string = ''
        num_predictions = bboxes.shape[0]
        n = 0
        for i in range(num_predictions):
            mask = masks[i]
            score = bboxes[i][-1]
            if score >= score_thr and mask.sum() > 0:
                # NOTE: add dilation to make the mask larger
                mask = mask.astype(np.uint8)
                kernel = np.ones(shape=(3, 3), dtype=np.uint8)
                bitmask = cv2.dilate(mask, kernel, 3)
                bitmask = bitmask.astype(bool)

                encoded = encode_binary_mask(bitmask)
                if n == 0:
                    pred_string += f"0 {score} {encoded.decode('utf-8')}"
                else:
                    pred_string += f" 0 {score} {encoded.decode('utf-8')}"
                n += 1

        height, width = cv2.imread(str(fname)).shape[:2]
        ids.append(id)
        prediction_strings.append(pred_string)
        heights.append(height)
        widths.append(width)

    return ids, prediction_strings, heights, widths


if __name__ == '__main__':
    image_root = '/home/yuchunli/git/mmdetection/demo/hubmap/samples'

    model_dict = {
        'name': 'solov2_x101_dcn_fpn_s1',
        'config': 'work_dirs/solov2_x101_dcn_fpn_s1/solov2_x101_dcn_fpn_s1.py',
        'ckpt': 'work_dirs/solov2_x101_dcn_fpn_s1/epoch_24.pth'
    }

    ids, prediction_strings, heights, widths = hubmap_single_model(
        model_dict, image_root, iou_thr=0.6, score_thr=0.001, max_num=1000)
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    submission.to_csv('submission.csv')

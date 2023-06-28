import base64
import os
import pickle
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


def single_model_inference(fnames, model_dict, output_path):
    model_name = model_dict['name']
    config_file = model_dict['config']
    ckpt = model_dict['ckpt']

    config = mmcv.Config.fromfile(config_file)
    model = init_detector(config, ckpt, device='cuda:0')
    results = inference_detector(model, fnames)
    for result in results:
        cls_encoded_masks = encode_mask_results(result[MASK_INDEX])
        for i, encoded_masks in enumerate(cls_encoded_masks):
            result[MASK_INDEX][i] = encoded_masks
    mmcv.dump(results, os.path.join(output_path, f'{model_name}.pkl'))


def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


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
    masks = mask_util.decode(masks)
    masks = masks.transpose(2, 0, 1)
    fused_masks = np.zeros((len(fused_boxes), img_h, img_w), dtype=bool)
    for i, fused_box in enumerate(fused_boxes):
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
        fused_masks[i, y1:y2, x1:x2] = vote_mask
    return fused_masks


def hubmap_ensemble(model_list, image_root, pkl_path):
    fnames = list(Path(image_root).glob('*.tif'))
    for i, model_dict in enumerate(model_list):
        single_model_inference(fnames, model_dict, pkl_path)

    model_level = []
    for i, model_dict in enumerate(model_list):
        results = read_pkl(os.path.join(pkl_path, f'{model_dict["name"]}.pkl'))
        model_level.append(results)

    num_models = len(model_level)
    fused_masks = []

    ids = []
    heights = []
    widths = []
    prediction_strings = []

    for i, fname in enumerate(fnames):
        h, w, c = cv2.imread(str(fname)).shape
        pred_string = ''
        img_dim = np.array([w, h, w, h])

        boxes = []
        scores = []
        labels = []
        masks = []
        for j in range(num_models):
            blood_vessel_bboxes = model_level[j][i][BBOX_INDEX][
                LABLE_INDEX][:, :4]
            blood_vessel_scores = model_level[j][i][BBOX_INDEX][LABLE_INDEX][:,
                                                                             4]
            blood_vessel_masks = model_level[j][i][MASK_INDEX][LABLE_INDEX]

            # NOTE: normalize bbox to [0, 1]
            boxes.append(blood_vessel_bboxes / img_dim)
            scores.append(blood_vessel_scores)
            labels.append(len(blood_vessel_bboxes) * [0])
            masks.extend(blood_vessel_masks)
        fused_box, fused_score, fused_label = weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=None,
            iou_thr=0.5,
            skip_box_thr=0.0001)
        fused = np.concatenate(
            [fused_box * img_dim,
             fused_score.reshape(-1, 1)], axis=1)
        # decode masks and stack masks and take mean
        # crop masks from bounding boxes
        fused_masks = ensemble_masks_from_boxes(fused, masks, h, w)
        assert len(fused) == len(fused_masks)
        for n, (pred_box, pred_mask) in enumerate(zip(fused, fused_masks)):
            score = pred_box[-1]
            encoded = encode_binary_mask(pred_mask)
            if n == 0:
                pred_string += f"0 {score} {encoded.decode('utf-8')}"
            else:
                pred_string += f" 0 {score} {encoded.decode('utf-8')}"
        ids.append(str(os.path.basename(fname)).split('.')[0].split('/')[-1])
        heights.append(h)
        widths.append(w)
        prediction_strings.append(pred_string)

    return ids, prediction_strings, heights, widths


if __name__ == '__main__':
    pkl_path = '/home/yuchunli/git/mmdetection/demo/hubmap/ensemble_pkls'
    image_root = '/home/yuchunli/git/mmdetection/demo/hubmap/samples'

    model_list = [{
        'name':
        'mask_rcnn_r50_fpn_2x_hubmap_giou_loss',
        'config':
        '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_2x_hubmap_giou_loss/mask_rcnn_r50_fpn_2x_hubmap_giou_loss.py',
        'ckpt':
        '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_2x_hubmap_giou_loss/best_segm_mAP_epoch_20.pth'
    }, {
        'name':
        'mask_rcnn_r50_fpn_giou_loss_strategy4',
        'config':
        '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy4/mask_rcnn_r50_fpn_giou_loss_strategy4.py',
        'ckpt':
        '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r50_fpn_giou_loss_strategy4/best_segm_mAP_epoch_12.pth'
    }]

    ids, prediction_strings, heights, widths = hubmap_ensemble(
        model_list, image_root, pkl_path)
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    # submission.to_csv("/kaggle/working/mmdetection/submission.csv")
    # submission.to_csv("submission.csv")

import os
import pickle
from pathlib import Path

import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from ensemble_boxes_wbf import weighted_boxes_fusion

from mmdet.apis import inference_detector, init_detector
from mmdet.core import encode_mask_results

BBOX_INDEX = 0
MASK_INDEX = 1
LABLE_INDEX = 1  # ONLY PICK BLOOD VESSEL


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


if __name__ == '__main__':
    pkl_path = '/home/yuchunli/git/mmdetection/demo/hubmap/ensemble_pkls'
    image_root = '/home/yuchunli/git/mmdetection/demo/hubmap/samples'
    fnames = list(Path(image_root).glob('*.tif'))

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

    for i, model_dict in enumerate(model_list):
        single_model_inference(fnames, model_dict, pkl_path)

    model_level = []
    for i, model_dict in enumerate(model_list):
        results = read_pkl(os.path.join(pkl_path, f'{model_dict["name"]}.pkl'))
        model_level.append(results)

    num_imgs = len(model_level[0])
    num_models = len(model_level)
    fused_boxes = []
    fused_scores = []
    fused_masks = []
    for i, fname in enumerate(fnames):
        h, w, c = cv2.imread(str(fname)).shape
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
        masks = mask_util.decode(masks)

        fused_boxes.append(fused)

import os
from pathlib import Path

import cv2
import mmcv
import numpy as np
import pandas as pd
from ensemble_boxes_wbf import weighted_boxes_masks_fusion

from mmdet.apis import inference_detector, init_detector

BBOX_INDEX = 0
MASK_INDEX = 1
LABLE_INDEX = 1  # ONLY PICK BLOOD VESSEL


def single_model_inference(fname,
                           config,
                           ckpt,
                           iou_thr=0.5,
                           score_thr=0.001,
                           max_num=100):
    assert isinstance(fname, str), f'fname must be str, but got {type(fname)}'
    config.model.test_cfg.rcnn.nms.iou_threshold = iou_thr
    config.model.test_cfg.rcnn.score_thr = score_thr
    config.model.test_cfg.rcnn.max_per_img = max_num
    config.model.roi_head.mask_head.type = 'CustomFCNMaskHead'

    model = init_detector(config, ckpt, device='cuda:0')
    results = inference_detector(model, fname)
    return results


def hubmap_ensemble(model_list,
                    image_root,
                    iou_thr=0.5,
                    score_thr=0.001,
                    max_num=100):
    ids = []
    heights = []
    widths = []
    prediction_strings = []

    fnames = list(Path(image_root).glob('*.tif'))
    for i, fname in enumerate(fnames):
        h, w, c = cv2.imread(str(fname)).shape
        pred_string = ''
        img_dim = np.array([w, h, w, h])

        boxes = []
        scores = []
        labels = []
        masks = []
        for j, model_dict in enumerate(model_list):
            config_file = model_dict['config']
            ckpt = model_dict['ckpt']
            config = mmcv.Config.fromfile(config_file)
            results = single_model_inference(
                str(fname), config, ckpt, iou_thr, score_thr, max_num)

            num_pred = len(results[BBOX_INDEX][LABLE_INDEX])
            # NOTE: normalize bbox to [0, 1]
            boxes.append(results[BBOX_INDEX][LABLE_INDEX][:, :4] / img_dim)
            scores.append(results[BBOX_INDEX][LABLE_INDEX][:, 4])
            labels.append(num_pred * [0])
            masks.append(np.stack(results[MASK_INDEX][LABLE_INDEX]))

        fused_box, encoded_strings, fused_score, fused_label = weighted_boxes_masks_fusion(
            boxes,
            masks,
            scores,
            labels,
            img_h=h,
            img_w=w,
            weights=None,
            iou_thr=iou_thr,
            skip_box_thr=score_thr)

        for n, (score,
                encoded) in enumerate(zip(fused_score, encoded_strings)):
            if n == 0:
                pred_string += f"0 {score} {encoded.decode('utf-8')}"
            else:
                if score > score_thr:
                    pred_string += f" 0 {score} {encoded.decode('utf-8')}"
        ids.append(str(os.path.basename(fname)).split('.')[0].split('/')[-1])
        heights.append(h)
        widths.append(w)
        prediction_strings.append(pred_string)

    return ids, prediction_strings, heights, widths


if __name__ == '__main__':
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
    }, {
        'name':
        'mask_rcnn_r101_fpn_strategy1',
        'config':
        '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r101_fpn_strategy1/mask_rcnn_r101_fpn_strategy1.py',
        'ckpt':
        '/home/yuchunli/git/mmdetection/work_dirs/mask_rcnn_r101_fpn_strategy1/best_segm_mAP_epoch_19.pth'
    }]

    ids, prediction_strings, heights, widths = hubmap_ensemble(
        model_list, image_root)
    submission = pd.DataFrame()
    submission['id'] = ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    # submission.to_csv("/kaggle/working/mmdetection/submission.csv")
    # submission.to_csv("submission.csv")

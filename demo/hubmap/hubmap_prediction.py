# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import base64
import typing as t
import zlib
from pathlib import Path

import copy

import cv2
import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv import Config
from pycocotools import _mask as coco_mask
from mmdet.apis import init_detector

from mmdet.core import encode_mask_results
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.utils import build_dp


@DATASETS.register_module()
class HubmapTTADataset(CustomDataset):
    CLASSES = ('blood_vessel')
    PALETTE = [(220, 20, 60)]
    TARGET_LABLE_INDEX = 0  # blood_vessel

    def load_annotations(self, ann_file):
        del ann_file  # not required
        img_infos = []

        # modify this according to your path
        paths = list(Path(self.img_prefix).glob('*'))

        paths.sort(key=lambda p: p.name)
        for img_path in paths:
            img_id = img_path.name.split('.')[0]
            filename = str(img_path)
            height, width = mmcv.imread(str(img_path)).shape[:2]
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        print(f'dataset size: {len(img_infos)}')
        return img_infos


class HubMAPTest:
    def __init__(self, cfg_path, ckpt_path, image_root, iou_threshold=0.6,
                 score_thr=0.001, max_num=1000):
        self.image_root = image_root
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.cfg = self.init_cfg(cfg_path)
        self.model = self.init_model(iou_threshold, score_thr, max_num)

    def encode_binary_mask(self, mask: np.ndarray) -> t.Text:
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

    def init_cfg(self, cfg_path):
        cfg = Config.fromfile(cfg_path)

        cfg.data.test.test_mode = True
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        cfg.data.test.type = 'HubmapTTADataset'
        cfg.data.test.ann_file = ''
        cfg.data.test.img_prefix = self.image_root
        return cfg

    def init_model(self, iou_threshold, score_thr, max_num):
        cfg = copy.deepcopy(self.cfg)
        test_cfg = cfg.model.test_cfg

        if test_cfg.get('max_per_img', None):
            test_cfg.max_per_img = max_num
        if test_cfg.get('iou_threshold', None):
            test_cfg.iou_threshold = iou_threshold
        if test_cfg.get('score_thr', None):
            test_cfg.score_thr = score_thr

        model = init_detector(cfg, self.ckpt_path, device='cuda:0')
        return model

    def format_results(self, results, score_thr=0.001, kernel_size=3):
        target_lable_index = HubmapTTADataset.TARGET_LABLE_INDEX
        bboxes, masks = results[0]
        bboxes, masks = bboxes[target_lable_index], masks[target_lable_index]
        pred_string = ''
        num_predictions = bboxes.shape[0]
        n = 0
        for i in range(num_predictions):
            mask = masks[i]
            score = bboxes[i][-1]
            if score >= score_thr and mask.sum() > 32:
                # NOTE: add dilation to make the mask larger
                mask = mask.astype(np.uint8)
                kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)
                bitmask = cv2.dilate(mask, kernel, 3)
                bitmask = bitmask.astype(bool)

                encoded = self.encode_binary_mask(bitmask)
                if n == 0:
                    pred_string += f"0 {score} {encoded.decode('utf-8')}"
                else:
                    pred_string += f" 0 {score} {encoded.decode('utf-8')}"
                n += 1
        return pred_string

    def predict(self, dump=False, score_thr=0.001, kernel_size=3):
        cfg = copy.deepcopy(self.cfg)

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, shuffle=False)
        test_loader_cfg = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        results = []
        image_ids = []
        heights = []
        widths = []
        prediction_strings = []

        model = build_dp(self.model, device_ids=[0])
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                image_id = str(
                    Path(data['img_metas'][0].data[0][0]['filename']).stem)
                result = model(return_loss=False, rescale=True, **data)
                if dump:
                    result = [(bbox_results, encode_mask_results(mask_results))
                              for bbox_results, mask_results in result]
                    results.extend(result)
                else:
                    pred_string = self.format_results(result, score_thr, kernel_size)
                    height, width = cv2.imread(
                        data['img_metas'][1].data[0][0]['filename']).shape[:2]
                    image_ids.append(image_id)
                    prediction_strings.append(pred_string)
                    heights.append(height)
                    widths.append(width)
        if dump:
            mmcv.dump(results, self.cfg.work_dir + '/val2.pkl')
        return image_ids, prediction_strings, heights, widths

    def predict_tta(self, dump=False, score_thr=0.001, kernel_size=3):
        cfg = copy.deepcopy(self.cfg)

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, shuffle=False)
        test_loader_cfg = {**test_dataloader_default_args, **cfg.data.get('test_dataloader', {})}
        if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
            tta_pipeline = cfg.data.test.pipeline[1]
            tta_pipeline.flip = True
            tta_pipeline.flip_direction = ['horizontal', 'vertical']
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        results = []
        image_ids = []
        heights = []
        widths = []
        prediction_strings = []

        model = build_dp(self.model, device_ids=[0])
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                image_id = str(
                    Path(data['img_metas'][0].data[0][0]['filename']).stem)
                result = model(return_loss=False, rescale=True, **data)
                bboxes, masks = result[0]
                bboxes = bboxes[0]
                masks = masks[0]

                if bboxes[0, :4].sum() == 0:
                    for i in range(len(masks)):
                        mask = masks[i]
                        y_s, x_s = np.where(mask == 1)
                        bboxes[i, 0] = x_s.min()
                        bboxes[i, 1] = y_s.min()
                        bboxes[i, 2] = x_s.max()
                        bboxes[i, 3] = y_s.max()

                result = [([bboxes], [masks])]
                if dump:
                    result = [(bbox_results, encode_mask_results(mask_results))
                              for bbox_results, mask_results in result]
                    results.extend(result)
                else:
                    pred_string = self.format_results(result, score_thr, kernel_size)
                    height, width = cv2.imread(
                        data['img_metas'][1].data[0][0]['filename']).shape[:2]
                    image_ids.append(image_id)
                    prediction_strings.append(pred_string)
                    heights.append(height)
                    widths.append(width)
        if dump:
            mmcv.dump(results, self.cfg.work_dir + '/tta_results2.pkl')
        return image_ids, prediction_strings, heights, widths


if __name__ == '__main__':
    parser = argparse.ArgumentParser("hubmap prediction")
    parser.add_argument('--config', type=str,
                        default='work_dirs/solov2_x101_dcn_fpn_hubmap_s5_cls1/solov2_x101_dcn_fpn_hubmap_s5_cls1.py')
    parser.add_argument('--ckpt', type=str,
                        default='work_dirs/solov2_x101_dcn_fpn_hubmap_s5_cls1/best_segm_mAP_epoch_8.pth')
    parser.add_argument('--image_root', type=str,
                        default='/home/yuchunli/_DATASET/HuBMAP-vasculature-coco-s5-cls_1/images/val')

    args = parser.parse_args()
    hubmap = HubMAPTest(args.config, args.ckpt, args.image_root)
    image_ids, prediction_strings, heights, widths = hubmap.predict_tta(dump=False)
    # hubmap.predict(dump=True)

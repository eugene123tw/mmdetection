# Copyright (c) OpenMMLab. All rights reserved.
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
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from pycocotools import _mask as coco_mask

from mmdet.core import encode_mask_results
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg, get_device, replace_cfg_vals,
                         rfnext_init_model, setup_multi_processes,
                         update_data_root)

LABLE_INDEX = 0  # blood_vessel


@DATASETS.register_module()
class HubmapTTADataset(CustomDataset):
    CLASSES = ('blood_vessel')
    PALETTE = [(220, 20, 60)]

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


def tta(model_dict, image_root, score_thr=0.001, max_num=1000, dump=False):

    config = model_dict['config']
    checkpoint = model_dict['ckpt']

    cfg = Config.fromfile(config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.device = get_device()

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # NOTE: hard coded TTA for now
    cfg.data.test.type = 'HubmapTTADataset'
    cfg.data.test.ann_file = ''
    cfg.data.test.img_prefix = image_root

    if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
        tta_pipeline = cfg.data.test.pipeline[1]
        tta_pipeline.flip = True
        tta_pipeline.flip_direction = ['horizontal', 'vertical']

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # init rfnext if 'RFSearchHook' is defined in cfg
    rfnext_init_model(model, cfg=cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    image_ids = []
    heights = []
    widths = []
    prediction_strings = []

    if dump:
        results = []

    model = build_dp(model, device_ids=[0])
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            image_id = str(
                Path(data['img_metas'][1].data[0][0]['filename']).stem)
            result = model(return_loss=False, rescale=True, **data)
            if dump:
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
                results.extend(result)
            else:
                bboxes, masks = result[0]
                bboxes, masks = bboxes[LABLE_INDEX], masks[LABLE_INDEX]
                pred_string = ''
                num_predictions = bboxes.shape[0]
                n = 0
                for i in range(num_predictions):
                    mask = masks[i]
                    score = bboxes[i][-1]
                    if score >= score_thr and mask.sum() > 32:
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
                height, width = cv2.imread(
                    data['img_metas'][1].data[0][0]['filename']).shape[:2]
                image_ids.append(image_id)
                prediction_strings.append(pred_string)
                heights.append(height)
                widths.append(width)

    if dump:
        mmcv.dump(results, os.path.dirname(checkpoint) + '/tta_results.pkl')

    return image_ids, prediction_strings, heights, widths


if __name__ == '__main__':
    image_root = '/home/yuchunli/_DATASET/HuBMAP-vasculature-coco-s5-cls_1/images/val'
    model_dict = {
        'name':
        'solov2_x101_dcn_fpn_hubmap_s5_cls1_1344x1344',
        'config':
        '/home/yuchunli/git/mmdetection/work_dirs/solov2_x101_dcn_fpn_hubmap_s5_cls1_1344x1344/solov2_x101_dcn_fpn_hubmap_s5_cls1_1344x1344.py',
        'ckpt':
        '/home/yuchunli/git/mmdetection/work_dirs/solov2_x101_dcn_fpn_hubmap_s5_cls1_1344x1344/best_segm_mAP_epoch_16.pth'
    }

    image_ids, prediction_strings, heights, widths = tta(
        model_dict, image_root, score_thr=0.001, max_num=1000, dump=True)
    submission = pd.DataFrame()
    submission['id'] = image_ids
    submission['height'] = heights
    submission['width'] = widths
    submission['prediction_string'] = prediction_strings
    submission = submission.set_index('id')
    submission.to_csv('submission.csv')
    print(submission)

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.ops.nms import nms

from mmdet.core import bbox2result
from mmdet.core.post_processing import multiclass_nms
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, idxs: np.ndarray,
                   iou_threshold: float, max_num: int, score_thr: float):
    """NMS for multi-class bboxes.

    Args:
        boxes (np.ndarray):  boxes in shape (N, 4).
        scores (np.ndarray): scores in shape (N, ).
        idxs (np.ndarray):  each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        iou_threshold (float): IoU threshold to be used to suppress boxes
            in tiles' overlap areas.
        max_num (int): if there are more than max_per_img bboxes after
            NMS, only top max_per_img will be kept.

    Returns:
        tuple: tuple: kept dets and indice.
    """
    if len(boxes) == 0:
        return None, []
    max_coordinate = boxes.max()
    offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    dets, keep = nms(
        boxes_for_nms, scores, iou_threshold, score_threshold=score_thr)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    return dets, keep


@DETECTORS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def aug_test(self, imgs, img_metas, rescale=False):
        num_classes = self.roi_head.bbox_head.num_classes

        aug_masks = []
        aug_bboxes = []
        aug_labels = []
        for img, img_meta in zip(imgs, img_metas):
            result = super(MaskRCNN, self).aug_test([img], [img_meta],
                                                    rescale=rescale)
            cls_bboxes, cls_masks = result[0]
            bboxes = np.vstack(cls_bboxes)
            labels = []
            for i, bbox in enumerate(cls_bboxes):
                labels.extend([i] * bbox.shape[0])
            masks = np.vstack(cls_masks)
            aug_masks.extend(masks)
            aug_labels.extend(labels)
            aug_bboxes.extend(bboxes)

        aug_masks = np.stack(aug_masks)
        aug_labels = np.stack(aug_labels)
        aug_bboxes = np.stack(aug_bboxes)

        _, keep = multiclass_nms(
            boxes=aug_bboxes[:, :4],
            scores=aug_bboxes[:, -1],
            idxs=aug_labels,
            iou_threshold=self.test_cfg.rcnn.nms.iou_threshold,
            score_thr=self.test_cfg.rcnn.score_thr,
            max_num=self.test_cfg.rcnn.max_per_img)

        aug_masks = aug_masks[keep]
        aug_labels = aug_labels[keep]
        aug_bboxes = aug_bboxes[keep]

        N = aug_bboxes.shape[0]

        bbox_results = bbox2result(aug_bboxes, aug_labels, num_classes)

        cls_segms = [[] for _ in range(num_classes)]
        for i in range(N):
            cls_segms[aug_labels[i]].append(aug_masks[i])
        return [(bbox_results, cls_segms)]

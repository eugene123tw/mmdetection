# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmdet.core import bbox2result, bbox_flip, mask_matrix_nms
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


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

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Augment testing of SOLOv2. The function implements the testing.

        Args:
            imgs (_type_): _description_
            img_metas (_type_): _description_
            rescale (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        aug_masks = []
        aug_boxes = []
        aug_scores = []
        aug_labels = []
        for img, img_meta in zip(imgs, img_metas):
            result = super().simple_test(img, img_meta, rescale=rescale)

            bboxes, masks = result[0]

            masks = masks[0]
            bboxes = bboxes[0]
            boxes = bboxes[:, :4]
            scores = bboxes[:, -1]

            labels = np.zeros(len(boxes))
            # flip mask back
            flip = img_meta[0]['flip']
            if flip:
                direction = img_meta[0]['flip_direction']
                boxes = bbox_flip(boxes, img_meta[0]['img_shape'], direction)

                for i in range(len(masks)):
                    if direction == 'horizontal':
                        masks[i] = np.fliplr(masks[i])
                    elif direction == 'vertical':
                        masks[i] = np.flipud(masks[i])

            aug_masks.extend(masks)
            aug_labels.extend(labels)
            aug_boxes.extend(boxes)
            aug_scores.extend(scores)

        aug_masks = np.stack(aug_masks)
        aug_labels = np.stack(aug_labels)
        aug_boxes = np.stack(aug_boxes)
        aug_scores = np.stack(aug_scores)
        sum_masks = aug_masks.sum((1, 2))

        keep = sum_masks > 0
        aug_masks = aug_masks[keep]
        aug_labels = aug_labels[keep]
        aug_boxes = aug_boxes[keep]
        aug_scores = aug_scores[keep]
        sum_masks = sum_masks[keep]

        scores, labels, masks, keep_inds = mask_matrix_nms(
            torch.tensor(aug_masks),
            torch.tensor(aug_labels).long(),
            torch.tensor(aug_scores),
            mask_area=torch.tensor(sum_masks),
            nms_pre=500,
            max_num=100,
            filter_thr=0.01)
        boxes = torch.tensor(aug_boxes[keep_inds])

        bboxes = torch.cat([boxes, scores[:, None]], dim=-1)
        num_classes = self.roi_head.bbox_head.num_classes
        bbox_results = bbox2result(bboxes, labels, num_classes)
        mask_results = [[] for _ in range(num_classes)]
        for j, label in enumerate(labels):
            mask = masks[j].detach().cpu().numpy()
            mask_results[label].append(mask)
        result = (bbox_results, mask_results)
        return [result]

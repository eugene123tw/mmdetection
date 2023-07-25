# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox_flip, mask_matrix_nms
from ..builder import DETECTORS
from .maskformer import MaskFormer


@DETECTORS.register_module()
class Mask2Former(MaskFormer):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
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
            feats = self.extract_feat(img)
            mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(
                feats, img_meta, rescale=rescale)
            results = self.panoptic_fusion_head.simple_test(
                mask_cls_results, mask_pred_results, img_meta, rescale=rescale)

            labels, bboxes, masks = results[0]['ins_results']

            labels = labels.detach().cpu()
            bboxes = bboxes.detach().cpu()
            masks = masks.detach().cpu()

            boxes = bboxes[:, :4]
            scores = bboxes[:, -1]

            # flip mask back
            flip = img_meta[0]['flip']
            if flip:
                direction = img_meta[0]['flip_direction']
                boxes = bbox_flip(boxes, img_meta[0]['img_shape'], direction)

                for i in range(len(masks)):
                    if direction == 'horizontal':
                        masks[i] = masks[i].fliplr()
                    elif direction == 'vertical':
                        masks[i] = masks[i].flipud()

            aug_masks.extend(masks)
            aug_labels.extend(labels)
            aug_boxes.extend(boxes)
            aug_scores.extend(scores)

        aug_masks = torch.stack(aug_masks)
        aug_labels = torch.stack(aug_labels)
        aug_boxes = torch.stack(aug_boxes)
        aug_scores = torch.stack(aug_scores)
        sum_masks = aug_masks.sum((1, 2)).float()

        keep = sum_masks > 0
        aug_masks = aug_masks[keep]
        aug_labels = aug_labels[keep]
        aug_boxes = aug_boxes[keep]
        aug_scores = aug_scores[keep]
        sum_masks = sum_masks[keep]

        scores, labels, masks, keep_inds = mask_matrix_nms(
            aug_masks,
            aug_labels,
            aug_scores,
            mask_area=sum_masks,
            nms_pre=500,
            max_num=100,
            filter_thr=0.01)
        boxes = aug_boxes[keep_inds]

        bboxes = torch.cat([boxes, scores[:, None]], dim=-1)
        bbox_results = bbox2result(bboxes, labels, self.num_things_classes)
        mask_results = [[] for _ in range(self.num_things_classes)]
        for j, label in enumerate(labels):
            mask = masks[j].detach().cpu().numpy()
            mask_results[label].append(mask)
        result = (bbox_results, mask_results)
        return [result]

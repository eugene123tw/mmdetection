# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import InstanceData, mask_matrix_nms
from ..builder import DETECTORS
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class SOLOv2(SingleStageInstanceSegmentor):
    """`SOLOv2: Dynamic and Fast Instance Segmentation
    <https://arxiv.org/abs/2003.10152>`_

    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

    def aug_test(self, imgs, img_metas, rescale=False):
        aug_masks = []
        aug_scores = []
        aug_labels = []
        for img, img_meta in zip(imgs, img_metas):
            # only one image in the batch
            feat = self.extract_feat(img)
            results_list = self.mask_head.simple_test(
                feat, img_meta, rescale=rescale, instances_list=None)
            result = results_list[0]
            # flip mask back
            flip = img_meta[0]['flip']
            if flip:
                direction = img_meta[0]['flip_direction']
                for i in range(len(result.masks)):
                    if direction == 'horizontal':
                        result.masks[i] = result.masks[i].fliplr()
                    elif direction == 'vertical':
                        result.masks[i] = result.masks[i].flipud()

            aug_masks.extend(result.masks)
            aug_labels.extend(result.labels)
            aug_scores.extend(result.scores)

        aug_masks = torch.stack(aug_masks)
        aug_labels = torch.stack(aug_labels)
        aug_scores = torch.stack(aug_scores)
        sum_masks = aug_masks.sum((1, 2)).float()

        scores, labels, masks, keep_inds = mask_matrix_nms(
            aug_masks,
            aug_labels,
            aug_scores,
            mask_area=sum_masks,
            nms_pre=self.test_cfg.nms_pre,
            max_num=self.test_cfg.max_per_img,
            kernel=self.test_cfg.kernel,
            sigma=self.test_cfg.sigma,
            filter_thr=self.test_cfg.filter_thr)

        results = InstanceData()
        results.masks = masks
        results.labels = labels
        results.scores = scores
        results = self.format_results(results)
        return [results]

    # def aug_test(self, imgs, img_metas, rescale=False):
    #     aug_masks = []
    #     aug_scores = []
    #     aug_labels = []
    #     for img, img_meta in zip(imgs, img_metas):
    #         # only one image in the batch
    #         feat = self.extract_feat(img)
    #         results_list = self.mask_head.simple_test(
    #             feat, img_meta, rescale=rescale, instances_list=None)
    #         result = results_list[0]
    #         # flip mask back
    #         masks = result.masks.cpu().numpy()
    #         labels = result.labels.cpu().numpy()
    #         scores = result.scores.cpu().numpy()

    #         flip = img_meta[0]['flip']
    #         if flip:
    #             direction = img_meta[0]['flip_direction']
    #             height, width = masks.shape[1:]
    #             masks = BitmapMasks(masks, height, width).flip(direction)
    #             masks = masks.to_ndarray()

    #         aug_masks.extend(masks)
    #         aug_labels.extend(labels)
    #         aug_scores.extend(scores)

    #     aug_masks = np.stack(aug_masks)
    #     aug_labels = np.stack(aug_labels)
    #     aug_scores = np.stack(aug_scores)

    #     mask_results = [[] for _ in range(self.mask_head.num_classes)]
    #     num_masks = len(aug_masks)
    #     if num_masks == 0:
    #         bbox_results = [
    #             np.zeros((0, 5), dtype=np.float32)
    #             for _ in range(self.mask_head.num_classes)
    #         ]
    #         results = (bbox_results, mask_results)
    #         return [results]

    #     aug_bboxes = np.zeros((num_masks, 4), dtype=np.float32)
    #     det_bboxes = np.concatenate([aug_bboxes, aug_scores[:, np.newaxis]], -1)
    #     bbox_results = [
    #         det_bboxes[aug_labels == i, :]
    #         for i in range(self.mask_head.num_classes)
    #     ]

    #     for idx in range(num_masks):
    #         aug_mask = aug_masks[idx]
    #         mask_results[aug_labels[idx]].append(aug_mask)

    #     results = (bbox_results, mask_results)
    #     return [results]

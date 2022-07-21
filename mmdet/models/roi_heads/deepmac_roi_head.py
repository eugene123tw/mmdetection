# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2roi
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead

@HEADS.register_module()
class DeepMACRoIHead(StandardRoIHead):
    def __init__(self, allowed_mask_classes, bbox_roi_extractor=None, bbox_head=None, mask_roi_extractor=None, mask_head=None, shared_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.allowed_mask_classes = allowed_mask_classes

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            if mask_results:
                losses.update(mask_results['loss_mask'])

        return losses

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        # TODO: keep allowed mask classes only


        # only get roi from ground-truth boxes
        gt_rois = bbox2roi([res.pos_bboxes[res.pos_assigned_gt_inds.unique()] for res in sampling_results])
        
        # fetch gt labels from sampling
        gt_labels = torch.cat([res.pos_gt_labels[res.pos_assigned_gt_inds.unique()]  for res in sampling_results])

        # empty keep handling
        if isinstance(self.allowed_mask_classes, list):
            self.allowed_mask_classes = torch.as_tensor(
              self.allowed_mask_classes,
              device=gt_labels.device)
        match_ids = gt_labels[:, None] == self.allowed_mask_classes[None, :]
        match_ids = torch.any(match_ids, 1)
        keep = torch.where(match_ids)[0]

        # TODO: tracing issue?
        # if len(keep) == 0:
        #     return None

        # forward Ground-Truth RoIs
        mask_results = self._mask_forward(x, gt_rois)
        
        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        loss_mask = self.mask_head.loss(mask_results['mask_pred'][keep],
                                        mask_targets[keep], gt_labels[keep])

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results
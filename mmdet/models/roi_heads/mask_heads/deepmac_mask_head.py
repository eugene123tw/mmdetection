# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import mask_target
from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
class DeepMacMaskHead(FCNMaskHead):

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        gt_inds = [
            res.pos_assigned_gt_inds.unique() for res in sampling_results
        ]
        gt_proposals = [
            res.pos_bboxes[res.pos_assigned_gt_inds.unique()]
            for res in sampling_results
        ]
        mask_targets = mask_target(gt_proposals, gt_inds, gt_masks,
                                   rcnn_train_cfg)
        return mask_targets

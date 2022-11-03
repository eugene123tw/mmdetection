from turtle import forward
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from torchvision.models import squeezenet1_0

from mmcv.cnn import Scale

import torch
from torch import nn
import numpy as np
from mmdet.models.losses import FocalLoss


class SubNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.loss_fun = nn.BCEWithLogitsLoss()

    def forward(self, img):
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y

    def loss(self, pred, target):
        loss = self.loss_fun(pred, target)
        return loss


# class SubNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = squeezenet1_0(num_classes=1)
#         # self.classifier = nn.Linear(1000, 1)
#         # self.loss_fun = nn.BCEWithLogitsLoss()
#         self.focal_loss = FocalLoss()

#     def forward(self, img):
#         x = self.features(img)
#         # y = self.classifier(x)
#         return x

#     def loss(self, pred, target):
#         loss = self.focal_loss(pred, target.long())
#         return loss


# class SubNet(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(20480, 1024),
#             nn.Linear(1024, 1024),
#             nn.Linear(1024, 1)
#         )
#         self.loss_fun = nn.BCEWithLogitsLoss()
    
#     def forward(self, x):
#         y = self.classifier(x.flatten(1))
#         return y

#     def loss(self, pred, target):
#         loss = self.loss_fun(pred, target)
#         return loss



@DETECTORS.register_module()
class TileRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(TileRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.objectness = SubNet()

    # def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
    #     x = self.extract_feat(img)

    #     targets = [len(gt_bbox) > 0 for gt_bbox in gt_bboxes]
    #     pred = self.objectness(img)
    #     target_labels = torch.tensor(targets, device=pred.device)
    #     loss_tile = self.objectness.loss(pred, target_labels.unsqueeze(1).float())

    #     losses = dict(loss_tile=loss_tile)

    #     # RPN forward and loss
    #     if self.with_rpn:
    #         proposal_cfg = self.train_cfg.get('rpn_proposal',
    #                                           self.test_cfg.rpn)
    #         rpn_losses, proposal_list = self.rpn_head.forward_train(
    #             x,
    #             img_metas,
    #             gt_bboxes,
    #             gt_labels=None,
    #             gt_bboxes_ignore=gt_bboxes_ignore,
    #             proposal_cfg=proposal_cfg,
    #             **kwargs)
    #         losses.update(rpn_losses)
    #     else:
    #         proposal_list = proposals

    #     roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
    #                                              gt_bboxes, gt_labels,
    #                                              gt_bboxes_ignore, gt_masks,
    #                                              **kwargs)
    #     losses.update(roi_losses)

    #     return losses

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        losses = dict()
        targets = [len(gt_bbox) > 0 for gt_bbox in gt_bboxes]
        pred = self.objectness(img)
        target_labels = torch.tensor(targets, device=pred.device)
        loss_tile = self.objectness.loss(pred, target_labels.unsqueeze(1).float())

        losses.update(dict(loss_tile=loss_tile))

        if not any(targets):
            return losses

        img = img[targets]
        img_metas = [item for keep, item in zip(targets, img_metas) if keep]
        gt_labels = [item for keep, item in zip(targets, gt_labels) if keep]
        gt_bboxes = [item for keep, item in zip(targets, gt_bboxes) if keep]
        gt_masks = [item for keep, item in zip(targets, gt_masks) if keep]
        if gt_bboxes_ignore:
            gt_bboxes_ignore = [
                item for keep, item in zip(targets, gt_bboxes_ignore) if keep]
        rcnn_loss = super().forward_train(
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore, gt_masks, proposals, **kwargs)
        losses.update(rcnn_loss)
        return losses

    # def simple_test(self, img, img_metas, **kwargs):
    #     @torch.jit.script_if_tracing
    #     def _inner_impl(img, keep: bool):
    #         if keep:
    #             return img
    #         img = torch.empty_like(img)
    #         return img

    #     # TODO: Assume batch size = 1
    #     pred = self.objectness(img)
    #     keep = (pred > 0.0)[0]

    #     img = _inner_impl(img, keep)
    #     return super().simple_test(img, img_metas, **kwargs)

    # def simple_test(self, img, img_metas, **kwargs):
    #     pred = self.objectness(img)
    #     keep = torch.sigmoid(pred)[0][0] > 0.5
    #     tmp_results = []
    #     num_classes = 1
    #     if keep:
    #         return super().simple_test(img, img_metas, **kwargs)

    #     bbox_results = []
    #     mask_results = []
    #     for _ in range(num_classes):
    #         bbox_results.append(np.empty((0, 5), dtype=np.float32))
    #         mask_results.append([])
    #     tmp_results.append((bbox_results, mask_results))
    #     return tmp_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        pred = self.objectness(img)
        keep = torch.sigmoid(pred)[0][0] > 0.45
        
        if not keep:
            tmp_results = []
            num_classes = 1
            bbox_results = []
            mask_results = []
            for _ in range(num_classes):
                bbox_results.append(np.empty((0, 5), dtype=np.float32))
                mask_results.append([])
            tmp_results.append((bbox_results, mask_results))
            return tmp_results


        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
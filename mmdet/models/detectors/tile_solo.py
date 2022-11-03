# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from ..builder import DETECTORS
from .solo import SOLO
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models.losses import CrossEntropyLoss
from mmdet.models.losses import accuracy

from torch import nn

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
            nn.Linear(256, 2),
        )
        self.loss_fun = CrossEntropyLoss()
    
    def forward(self, img, prob=False):
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        if prob:
            return torch.softmax(logits, -1)
        return logits
    
    def loss(self, pred, label):
        loss = self.loss_fun(pred, label.long())
        return loss


#TODO[EUGENE]: TRY MASKRCNN

@DETECTORS.register_module()
class TiledSOLO(SOLO):
    """`SOLO: Segmenting Objects by Locations
    <https://arxiv.org/abs/1912.04488>`_

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
        self.objectness = SubNet()


    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (B, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_masks (list[:obj:`BitmapMasks`] | None) : The segmentation
                masks for each box.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes (list[Tensor]): Each item is the truth boxes
                of each image in [tl_x, tl_y, br_x, br_y] format.
                Default: None.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss = dict()
        keep = [len(gt_mask) > 0 for gt_mask in gt_masks]
        logits = self.objectness(img)
        objectness_labels = torch.tensor(keep, device=logits.device)
        loss_objectness = self.objectness.loss(logits, objectness_labels)

        probs = torch.softmax(logits, -1)
        loss['acc'] = accuracy(logits, objectness_labels)

        loss.update(dict(loss_objectness=loss_objectness))
        if not any(keep):
            return loss

        img = img[keep]
        img_metas = [item for keep, item in zip(keep, img_metas) if keep]
        gt_labels = [item for keep, item in zip(keep, gt_labels) if keep]
        gt_bboxes = [item for keep, item in zip(keep, gt_bboxes) if keep]
        gt_masks = [item for keep, item in zip(keep, gt_masks) if keep]
        if gt_bboxes_ignore:
            gt_bboxes_ignore = [
                item for keep, item in zip(keep, gt_bboxes_ignore) if keep]

        solo_loss = super().forward_train(
            img,
            img_metas,
            gt_masks,
            gt_labels,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            **kwargs
        )
        loss.update(solo_loss)
        return loss
    
    def simple_test(self, img, img_metas, rescale=False):
        prob = self.objectness(img, prob=True)
        keep_list = prob[:, -1] >= 0.5
        img = img[keep_list]
        img_metas = [item for keep, item in zip(keep_list, img_metas) if keep]
        tmp_results = []
        num_classes = self.mask_head.num_classes
        if any(keep_list):
            results = super().simple_test(img, img_metas, rescale)
            for keep in keep_list:
                if keep:
                    result = results.pop(0)
                    tmp_results.append(result)
                else:
                    bbox_results = []
                    mask_results = []
                    for _ in range(num_classes):
                        bbox_results.append(np.empty((0, 5)))
                        mask_results.append([])
                    tmp_results.append((bbox_results, mask_results))
            return tmp_results
        for keep in keep_list:
            bbox_results = []
            mask_results = []
            for _ in range(num_classes):
                bbox_results.append(np.empty((0, 5)))
                mask_results.append([])
            tmp_results.append((bbox_results, mask_results))
        return tmp_results
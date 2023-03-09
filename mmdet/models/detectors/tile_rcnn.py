from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmcv.runner import auto_fp16

import torch
from torch import nn


class TileClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16_enabled = False
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
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 6 * 6, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1)
        )

        # TODO: FIND A WAY TO INJECT POS WEIGHT
        self.loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
        self.sigmoid = torch.nn.Sigmoid()

    @auto_fp16()
    def forward(self, img):
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y

    @auto_fp16()
    def loss(self, pred, target):
        loss = self.loss_fun(pred, target)
        return loss

    @auto_fp16()
    def simple_test(self, img):
        return self.sigmoid(self.forward(img))[0][0]


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
        self.tile_classifier = TileClassifier()

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
        pred = self.tile_classifier(img)
        target_labels = torch.tensor(targets, device=pred.device)
        loss_tile = self.tile_classifier.loss(pred, target_labels.unsqueeze(1).float())

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


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        # mean = img_metas[0]['img_norm_cfg']['mean']
        # std = img_metas[0]['img_norm_cfg']['std']
        # original_image = img * torch.tensor(std, device=img.device).view(3, 1, 1) + torch.tensor(mean, device=img.device).view(3, 1, 1)
        # original_image = original_image[0].permute(1, 2, 0).cpu().numpy().copy()
        # original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_RGB2BGR) 
        # cv2.putText(original_image, f"{torch.sigmoid(pred)[0][0]}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imwrite(f"/home/yuchunli/git/mmdetection/work_dirs/tile_rcnn_r50_fpn_1x_coliform_tile/eval_output/{os.path.basename(img_metas[0]['tile_path'])}", original_image)

        # @torch.jit.script_if_tracing
        # def _inner_impl(img, keep):
        #     if not keep:
        #         img = F.adaptive_avg_pool2d(img, (64, 64))
        #         return img
        #     return img
        # img = _inner_impl(img, keep)

        keep = self.tile_classifier.simple_test(img) > 0.45
        if not keep:
            tmp_results = []
            num_classes = 1
            bbox_results = []
            mask_results = []
            for _ in range(num_classes):
                bbox_results.append(torch.empty((0, 5), dtype=torch.float32))
                mask_results.append([])
            tmp_results.append((bbox_results, mask_results))
            return tmp_results

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

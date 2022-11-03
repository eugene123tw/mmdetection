# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import numpy as np

from mmdet.models.dense_heads import RPNHead

from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor


def _demo_mm_inputs(input_shape=(1, 3, 300, 300),
                    num_items=None, num_classes=10,
                    with_semantic=False):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': np.array([1.1, 1.2, 1.1, 1.2]),
        'flip': False,
        'flip_direction': None,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
        'gt_masks': gt_masks,
    }

    if with_semantic:
        # assume gt_semantic_seg using scale 1/8 of the img
        gt_semantic_seg = np.random.randint(
            0, num_classes, (1, 1, H // 8, W // 8), dtype=np.uint8)
        mm_inputs.update(
            {'gt_semantic_seg': torch.ByteTensor(gt_semantic_seg)})

    return mm_inputs


def test_single_roi_extractor():
    # test w.o. pre/post
    cfg = dict(
        roi_layer=dict(type='RoIAlign', output_size=416, sampling_ratio=2),
        out_channels=256,
        featmap_strides=[1])

    roie = SingleRoIExtractor(**cfg)

    imgs = torch.rand((2, 256, 200, 336)),
    
    rois = torch.tensor([
        [0.0000, 587.8285, 52.1405, 886.2484, 341.5644],
        [1.0000, 587.8285, 52.1405, 886.2484, 341.5644],
        ])

    res = roie(imgs, rois)
    assert res.shape == torch.Size([1, 256, 7, 7])


def test_rpn_head():
    s = 256
    channels = 32
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    train_cfg = mmcv.Config(
        {
            'assigner': {
                'type': 'MaxIoUAssigner', 
                'pos_iou_thr': 0.7, 
                'neg_iou_thr': 0.3, 
                'min_pos_iou': 0.3, 
                'match_low_quality': True, 
                'ignore_iof_thr': -1}, 
            'sampler': {
                'type': 'RandomSampler', 
                'num': 256, 
                'pos_fraction': 0.5, 
                'neg_pos_ub': -1, 
                'add_gt_as_proposals': False}, 
            'allowed_border': -1, 
            'pos_weight': -1, 
            'debug': False})
    
    test_cfg = mmcv.Config(
        {
            'nms_pre': 1000, 
            'max_per_img': 1000, 
            'nms': {'type': 'nms', 'iou_threshold': 0.7}, 
            'min_bbox_size': 0
        })
    
    rpn_head = RPNHead(in_channels=channels, train_cfg=train_cfg)

    # Anchor head expects a multiple levels of features per image
    feats = [
        torch.rand(1, channels, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(len(rpn_head.anchor_generator.strides))
    ]

    gt_bboxes = [
        torch.Tensor([[78.0, 78.0, 156.0, 156.0]]),
    ]
    
    rpn_losses, proposal_list = rpn_head.forward_train(
                                    feats,
                                    img_metas,
                                    gt_bboxes,
                                    gt_labels=None,
                                    proposal_cfg=test_cfg)
    
    print(len(proposal_list))
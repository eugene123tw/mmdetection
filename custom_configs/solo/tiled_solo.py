_base_ = [
    './decoupled_solo_r50_fpn_3x.py',
    '../__base__/datasets/vitens_coliform_tile_shorten.py',
    # '../__base__/datasets/vitens_coliform_tile.py',
]

# model settings
model = dict(
    type='TiledSOLO',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='DecoupledSOLOLightHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        loss_mask=dict(
            type='DiceLoss', use_sigmoid=True, loss_weight=3.0,
            activate=False),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',
        sigma=2.0,
        max_per_img=100))

load_from = '/home/yuchunli/_MODELS/mmdet/decoupled_solo_light_r50_fpn_3x_coco_20210906_142703-e70e226f.pth'

optimizer = dict(type='SGD', lr=0.01)

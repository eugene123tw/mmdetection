_base_ = [
    '../_base_/datasets/hubmap_strategy5_cls1.py',
    './mask_rcnn_r50_fpn.py',
]

model = dict(
    backbone=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    roi_head=dict(bbox_head=dict(num_classes=1)))

runner = dict(type='EpochBasedRunnerWithCancel', max_epochs=100)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='ReduceLROnPlateau',
    metric='segm_mAP',
    patience=5,
    iteration_patience=0,
    interval=1,
    min_lr=1e-06,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        patience=10,
        iteration_patience=0,
        metric='segm_mAP',
        interval=1,
        priority=75)
]

_base_ = [
    '../_base_/datasets/hubmap_strategy5_aug.py',
    './point_rend_r50_caffe_fpn_hubmap.py'
]

data = dict(samples_per_gpu=8)

runner = dict(type='EpochBasedRunnerWithCancel', max_epochs=100)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='ReduceLROnPlateau',
    metric='mAP',
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

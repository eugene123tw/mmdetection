_base_ = [
    '../_base_/datasets/hubmap_strategy5_cls1_ms.py',
    './solov2_x101_dcn_fpn.py'
]

model = dict(mask_head=dict(num_classes=1))

data = dict(samples_per_gpu=2)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333,
    step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)

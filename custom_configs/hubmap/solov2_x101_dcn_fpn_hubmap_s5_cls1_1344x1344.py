_base_ = [
    '../_base_/datasets/hubmap_1344x1344_s5_cls1.py',
    './solov2_x101_dcn_fpn.py'
]

model = dict(mask_head=dict(num_classes=1))

data = dict(samples_per_gpu=2)

runner = dict(type='EpochBasedRunner', max_epochs=24)

# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])

custom_hooks = []

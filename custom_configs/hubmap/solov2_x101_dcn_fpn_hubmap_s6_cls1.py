_base_ = [
    '../_base_/datasets/hubmap_strategy6_cls1.py',
    './solov2_x101_dcn_fpn.py'
]

model = dict(mask_head=dict(num_classes=1))

data = dict(samples_per_gpu=4)

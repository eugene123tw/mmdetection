_base_ = [
    '../_base_/datasets/hubmap_strategy5_cls1.py',
    './cascade_mask_rcnn_convnext-s_s5_cls1.py'
]

data = dict(samples_per_gpu=2)
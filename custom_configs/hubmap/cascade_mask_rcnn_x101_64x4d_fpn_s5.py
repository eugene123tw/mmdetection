_base_ = [
    '../_base_/datasets/hubmap_strategy5.py',
    './cascade_mask_rcnn_x101_64x4d_fpn.py'
]

data = dict(samples_per_gpu=4)

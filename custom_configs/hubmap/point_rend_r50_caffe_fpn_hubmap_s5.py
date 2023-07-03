_base_ = [
    '../_base_/datasets/hubmap_strategy5.py',
    './point_rend_r50_caffe_fpn_hubmap.py'
]

data = dict(samples_per_gpu=4)

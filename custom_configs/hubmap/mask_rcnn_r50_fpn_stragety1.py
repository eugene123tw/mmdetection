_base_ = ['../_base_/datasets/hubmap_strategy1.py', './mask_rcnn_r50_fpn.py']

data = dict(samples_per_gpu=8)

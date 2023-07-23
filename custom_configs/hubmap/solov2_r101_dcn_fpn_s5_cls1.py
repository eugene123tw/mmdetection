_base_ = [
    '../_base_/datasets/hubmap_strategy5_cls1.py', 
    './solov2_r101_dcn_fpn.py'
]

data = dict(samples_per_gpu=4)
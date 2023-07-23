_base_ = [
    '../_base_/datasets/hubmap_strategy5_cls1_mask2former.py',
    './mask2former_swin-s-p4-w7-224_lsj.py',
]

data = dict(samples_per_gpu=4)

_base_ = [
    '../_base_/datasets/hubmap_strategy4.py',
    './mask_rcnn_r50_fpn.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))), )

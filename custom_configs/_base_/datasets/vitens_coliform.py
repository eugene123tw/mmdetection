dataset_type = 'CocoDataset'
data_root = '/home/yuchunli/_DATASET/Vitens-Coliform-coco/'
samples_per_gpu = 4
img_size = (1024, 1024)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
classes = ['coliform']

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(
        type='RandomFlip',
        flip_ratio=0.5,
        direction=['horizontal', 'vertical']),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline,
        classes=classes))

# dataset settings
dataset_type = 'CocoDataset'
data_root_s5 = '/home/yuchunli/_DATASET/HuBMAP-vasculature-coco-strategy_5/'
data_root_pseudo = '/home/yuchunli/_DATASET/HuBMAP-vasculature-coco-pseudo-labeling/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
classes = ['glomerulus', 'blood_vessel']

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.25),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.25),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.25),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(
        type='RandomFlip',
        flip_ratio=0.5,
        direction=['horizontal', 'vertical']),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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

dataset_s5 = dict(
    type=dataset_type,
    ann_file=data_root_s5 + 'annotations/instances_train.json',
    img_prefix=data_root_s5 + 'images/train/',
    pipeline=train_pipeline,
    classes=classes)

dataset_pseudo = dict(
    type=dataset_type,
    ann_file=data_root_pseudo + 'annotations/instances_train.json',
    img_prefix=data_root_pseudo + 'images/train/',
    pipeline=train_pipeline,
    classes=classes)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[dataset_s5, dataset_pseudo]),
    val=dict(
        type=dataset_type,
        ann_file=data_root_s5 + 'annotations/instances_val.json',
        img_prefix=data_root_s5 + 'images/val/',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root_s5 + 'annotations/instances_val.json',
        img_prefix=data_root_s5 + 'images/val/',
        pipeline=test_pipeline,
        classes=classes))

dataset_type = 'CocoDataset'
data_root = '/home/yuchunli/_DATASET/ship-detection-coco-full/'
samples_per_gpu = 12
img_size = (1024, 1024)

tile_cfg = dict(
    tile_size=600,  # y = (1024**2 * 896) / 128**2
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.45,
    max_per_img=1700,
    filter_empty_gt=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.2,
        scale_limit=1,
        rotate_limit=45,
        interpolation=1,
        p=0.8),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
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
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.5),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=False),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=False),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
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
            dict(type='Collect', keys=['img']),
        ],
    )
]

train_dataset = dict(
    type='ImageTilingDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix='/home/yuchunli/_DATASET/ship-detection/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        classes=['ship'],
    ),
    pipeline=train_pipeline,
    **tile_cfg)

val_dataset = dict(
    type='ImageTilingDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix='/home/yuchunli/_DATASET/ship-detection/train',
        test_mode=True,
        pipeline=[dict(type='LoadImageFromFile')],
        classes=['ship'],
    ),
    pipeline=test_pipeline,
    **tile_cfg)

test_dataset = dict(
    type='ImageTilingDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix='/home/yuchunli/_DATASET/ship-detection/test',
        test_mode=True,
        pipeline=[dict(type='LoadImageFromFile')],
        classes=['ship'],
    ),
    pipeline=test_pipeline,
    **tile_cfg)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=train_dataset,
    val=val_dataset,
    test=test_dataset)

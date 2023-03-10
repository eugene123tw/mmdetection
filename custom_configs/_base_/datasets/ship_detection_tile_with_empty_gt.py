dataset_type = 'CocoDataset'
data_root = "/home/yuchunli/_DATASET/ship-detection-coco/"
samples_per_gpu = 12
img_size = (1024, 1024)

tile_cfg = dict(
    tile_size=400,  # y = (1024**2 * 896) / 128**2
    min_area_ratio=0.9,
    overlap_ratio=0.2,
    iou_threshold=0.45,
    max_per_img=1700,
    filter_empty_gt=False
)


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="Resize", img_scale=img_size, keep_ratio=False),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    )
]

train_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_train.json",
        img_prefix=data_root + "images/train",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        classes=['ship'],
    ),
    pipeline=train_pipeline,
    **tile_cfg
)

val_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val.json",
        img_prefix=data_root + "images/val",
        test_mode=True,
        pipeline=[dict(type="LoadImageFromFile")],
        classes=['ship'],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)

test_dataset = dict(
    type="ImageTilingDataset",
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_test.json",
        img_prefix=data_root + "images/test",
        test_mode=True,
        pipeline=[dict(type="LoadImageFromFile")],
        classes=['ship'],
    ),
    pipeline=test_pipeline,
    **tile_cfg
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    train=train_dataset,
    val=val_dataset,
    test=test_dataset
)

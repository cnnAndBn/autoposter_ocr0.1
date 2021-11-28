model = dict(
    type='PSENet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPNF',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_type='concat'),
    bbox_head=dict(
        type='PSEHead',
        text_repr_type='poly',
        in_channels=[256],
        out_channels=7,
        loss=dict(type='PSELoss')),
    train_cfg=None,
    test_cfg=None)
train_cfg = None
test_cfg = None
dataset_type = 'IcdarDataset'
data_root = '/media/yons/myspace1/dataset/OcrDataset/CTW1500/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=0.12549019607843137, saturation=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='ScaleAspectJitter',
        img_scale=[(3000, 736)],
        ratio_range=(0.5, 3),
        aspect_ratio_range=(1, 1),
        multiscale_mode='value',
        long_size_bound=1280,
        short_size_bound=640,
        resize_type='long_short_bound',
        keep_ratio=False),
    dict(type='PSENetTargets'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        instance_key='gt_kernels'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=True, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='IcdarDataset',
        ann_file=
        '/media/yons/myspace1/dataset/OcrDataset/CTW1500/json_style/instances_training.json',
        img_prefix='/media/yons/myspace1/dataset/OcrDataset/CTW1500//imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='ScaleAspectJitter',
                img_scale=[(3000, 736)],
                ratio_range=(0.5, 3),
                aspect_ratio_range=(1, 1),
                multiscale_mode='value',
                long_size_bound=1280,
                short_size_bound=640,
                resize_type='long_short_bound',
                keep_ratio=False),
            dict(type='PSENetTargets'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomRotateTextDet'),
            dict(
                type='RandomCropInstances',
                target_size=(640, 640),
                instance_key='gt_kernels'),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                keys=['gt_kernels', 'gt_mask'],
                visualize=dict(flag=True, boundary_key='gt_kernels')),
            dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
        ]),
    val=dict(
        type='IcdarDataset',
        ann_file=
        '/media/yons/myspace1/dataset/OcrDataset/CTW1500/json_style/instances_training.json',
        img_prefix='/media/yons/myspace1/dataset/OcrDataset/CTW1500//imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 1280),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1280, 1280),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='IcdarDataset',
        ann_file=
        '/media/yons/myspace1/dataset/OcrDataset/CTW1500/json_style/instances_training.json',
        img_prefix='/media/yons/myspace1/dataset/OcrDataset/CTW1500//imgs',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 1280),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1280, 1280),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=10, metric='hmean-iou')
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[200, 400])
total_epochs = 600
checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/psenet_r50_fpnf_600e_ctw1500'
gpu_ids = range(0, 1)

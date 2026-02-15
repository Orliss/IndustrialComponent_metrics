norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/segnext_tiny_512x512_ade_160k.pth')),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=3.0,
                class_weight=[0.7, 0.3],
                avg_non_ignore=True),
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                class_weight=[3, 7],
                loss_weight=1.0)
        ],
        ham_kwargs=dict(MD_R=16)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CustomDataset'
data_root = '/media/yaogan103/5C6410CF6410AE30/SegNext_0204/datasets/data'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(
        type='CUSTAM_AUG',
        crack_dir=
        '/media/yaogan103/5C6410CF6410AE30/SegNext_0204/datasets/crackset/images',
        max_crack_num=5,
        paste_crack=True,
        image_shape=(512, 512),
        prob=0.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AdjustGamma'),
    dict(
        type='Normalize',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255.0, 255.0, 255.0],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type='CustomDataset',
            data_root=
            '/media/yaogan103/5C6410CF6410AE30/SegNext_0204/datasets/data',
            img_dir='images/train',
            ann_dir='labels/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                dict(
                    type='CUSTAM_AUG',
                    crack_dir=
                    '/media/yaogan103/5C6410CF6410AE30/SegNext_0204/datasets/crackset/images',
                    max_crack_num=5,
                    paste_crack=True,
                    image_shape=(512, 512),
                    prob=0.5),
                dict(type='RandomFlip', prob=0.5),
                dict(type='AdjustGamma'),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255.0, 255.0, 255.0],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='CustomDataset',
        data_root=
        '/media/yaogan103/5C6410CF6410AE30/SegNext_0204/datasets/data',
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        data_root=
        '/media/yaogan103/5C6410CF6410AE30/SegNext_0204/datasets/data',
        img_dir='images/test',
        ann_dir='labels/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=1000, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = ''
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0001,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=322000)
checkpoint_config = dict(by_epoch=False, interval=15000)
evaluation = dict(interval=1000000, metric='mIoU')
find_unused_parameters = True
work_dir = './work_dirs/sanyi_mianzhen_gauss_sobelconv_512_20260203_finetune'
gpu_ids = [0]
auto_resume = False

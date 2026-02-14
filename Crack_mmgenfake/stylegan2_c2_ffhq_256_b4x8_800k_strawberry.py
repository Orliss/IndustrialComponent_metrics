dataset_type = 'UnconditionalImageDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
    dict(type='Resize', keys=['real_img'], scale=(256, 256)),
    dict(type='Flip', keys=['real_img'], direction='horizontal'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='UnconditionalImageDataset',
        imgs_root=
        r'Crack\data',
        pipeline=[
            dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
            dict(type='Resize', keys=['real_img'], scale=(256, 256)),
            dict(type='Flip', keys=['real_img'], direction='horizontal'),
            dict(
                type='Normalize',
                keys=['real_img'],
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['real_img']),
            dict(
                type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
        ]),
    val=dict(
        type='UnconditionalImageDataset',
        imgs_root=
        r'Crack\data',
        pipeline=[
            dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
            dict(type='Resize', keys=['real_img'], scale=(256, 256)),
            dict(type='Flip', keys=['real_img'], direction='horizontal'),
            dict(
                type='Normalize',
                keys=['real_img'],
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['real_img']),
            dict(
                type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
        ]))
d_reg_interval = 16
g_reg_interval = 4
g_reg_ratio = 0.8
d_reg_ratio = 0.9411764705882353
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        type='StyleGANv2Generator', out_size=256, style_channels=512),
    discriminator=dict(
        type='ADAStyleGAN2Discriminator', in_size=256, data_aug=None),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=dict(
        type='R1GradientPenalty',
        loss_weight=80.0,
        interval=16,
        norm_mode='HWC',
        data_info=dict(real_data='real_imgs', discriminator='disc')),
    gen_auxiliary_loss=dict(
        type='GeneratorPathRegularizer',
        loss_weight=8.0,
        pl_batch_shrink=2,
        interval=4,
        data_info=dict(generator='gen', num_batches='batch_size')))
train_cfg = dict(use_ema=True)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=0.0016, betas=(0, 0.9919919678228657)),
    discriminator=dict(
        type='Adam', lr=0.0018823529411764706, betas=(0, 0.9905854573074332)))
checkpoint_config = dict(interval=3000, by_epoch=False, max_keep_ckpts=30)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=500),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.9977843871238888),
        priority='VERY_HIGH')
]
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000)]
find_unused_parameters = True
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
aug_kwargs = dict(
    xflip=1,
    rotate90=1,
    xint=1,
    scale=1,
    rotate=1,
    aniso=1,
    xfrac=1,
    brightness=1,
    contrast=1,
    lumaflip=1,
    hue=1,
    saturation=1)
ema_half_life = 10.0
lr_config = None
total_iters = 50000
num_sample = 500
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=500,
        inception_pkl=
        r'Crack\crack.pkl',
        bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=500, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=500))
evaluation = dict(
    type='GenerativeEvalHook',
    interval=3000,
    metrics=dict(
        type='FID',
        num_images=500,
        inception_pkl=
        r'Crack\crack.pkl',
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))
work_dir = r'Crack\work_dir'
gpu_ids = [0]

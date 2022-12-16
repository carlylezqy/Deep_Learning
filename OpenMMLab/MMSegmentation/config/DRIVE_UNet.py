_base_ = [
    '/home/akiyo/nfs/zhang/github/mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py', 
    '/home/akiyo/nfs/zhang/github/mmsegmentation/configs/_base_/datasets/drive.py',
    '/home/akiyo/nfs/zhang/github/mmsegmentation/configs/_base_/default_runtime.py', 
    '/home/akiyo/nfs/zhang/github/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

model = dict(test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
norm_cfg = dict(_delete_=True, type='BN', requires_grad=True)

evaluation = dict(metric='mDice')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (584, 565)
crop_size = (64, 64)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            img_dir='training/images',
            ann_dir='training/1st_manual',
            pipeline=train_pipeline
        )),
    val=dict(
        img_dir='training/images',
        ann_dir='training/1st_manual',
        pipeline=test_pipeline),
    test=dict(
        img_dir='test/images',
        pipeline=test_pipeline)
)

work_dir = '/home/akiyo/sandbox/work_dirs/DRIVE'

runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=200, metric='mIoU', pre_eval=True)

device='cuda'
gpu_ids=[0]
seed=43
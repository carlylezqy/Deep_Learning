img_scale = (64, 64)
crop_size = (64, 64)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

test_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    dict(type='Resize', img_scale=img_scale),
    dict(type='RandomFlip', prob=0, direction='vertical'),
    #dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=0),
    #dict(type='Pad', size_divisor=200, pad_val=0, seg_pad_val=0),
    #dict(type='SegRescale', scale_factor=2),
    #dict(type='RandomCrop', crop_size=(512, 512)),
    
    #dict(type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
    
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Transpose', keys=['img'], order=(2, 0, 1)),
    #dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    #dict(type='ToDataContainer'),
]

pipeline = [
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

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='MSDDataset',
        pipeline=pipeline,
        data_root='/home/akiyo/cached_dataset/MSD_FOR_MMSEG/Task03_Liver',
        img_dir='images',
        ann_dir='annotations',
    ),
    val=dict(
        type='MSDDataset',
        pipeline=pipeline,
        data_root='/home/akiyo/cached_dataset/MSD_FOR_MMSEG/Task01_BrainTumour',
        img_dir='images',
        ann_dir='annotations',
    ),
)
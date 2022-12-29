"""
type='ImageClassifier' ==> 对应类名
mmclassification/mmcls/models/classifiers/image.py/ImageClassifier
"""

#Model setting
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,       # use all 4 conv layers
        out_indices=(3, ),  # output of the last conv layer
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=1000,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)

#Dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

dataset_type = 'ImageNet'
data = dict(
    samples_per_gpu=128, workers_per_gpu=4,
    train=dict(type=dataset_type, data_prefix='data/mnist', pipeline=train_pipeline),
    val=dict(type=dataset_type, data_prefix='data/mnist', pipeline=train_pipeline),
    test=dict(type=dataset_type, data_prefix='data/mnist', pipeline=test_pipeline)
)

"""
samples_per_gpu: Batch Size
workers_per_gpu: 读取数据时,单个CPU/GPU的线程数
data_prefix: 数据前缀
"""

#Optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=100)

work_dir = '/home/akiyo/sandbox/work_dirs/MINST'
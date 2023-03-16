_base_ = [
    'mmcls::_base_/models/resnest50.py',
    'mmcls::_base_/datasets/cifar10_bs16.py',
]

custom_imports = dict(
    imports=[
        'mmcls.utils.setup_env',
        'mmcls.datasets.cifar',
    ], 
    allow_failed_imports=False
)

model = _base_.model
num_classes = 10

model = dict(
    head = dict(
        num_classes=num_classes,
        loss = dict(num_classes=num_classes),
    ),
    data_preprocessor = dict(
        num_classes=num_classes,
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False
    )
)

train_pipeline = [
    dict(type='Resize', scale=224, keep_ratio=False),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='PackClsInputs'),
]

work_dir='/home/akiyo/sandbox/ResNetConfigTrain'
CIFAR10_PATH = "/home/akiyo/nfs/zhang/dataset"

train_dataloader=dict(
    dataset=dict(
        data_prefix=CIFAR10_PATH,
        pipeline=train_pipeline
    ),
    batch_size=10,
    num_workers=4
)

val_dataloader=dict(
    dataset=dict(
        data_prefix=CIFAR10_PATH,
        pipeline=test_pipeline
    ),
    batch_size=10,
    num_workers=4
)

# test_dataloader=dict(
#     dataset=dict(type='ToyDataset'),
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     batch_size=1,
#     num_workers=0
# )

auto_scale_lr=dict(base_batch_size=16, enable=False)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.001))
param_scheduler=dict(type='MultiStepLR', milestones=[1, 2])
val_evaluator=dict(type='Accuracy', topk=(1, ))
#test_evaluator=dict(type='ToyEvaluator')
train_cfg=dict(by_epoch=True, max_epochs=3, val_interval=1)
val_cfg=dict()
#test_cfg=dict()
custom_hooks=[]

default_hooks=dict(
    timer=dict(type='IterTimerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook'),
    optimizer=dict(type='OptimizerHook', grad_clip=False),
    param_scheduler=dict(type='ParamSchedulerHook')
)

launcher='none'
env_cfg=dict(dist_cfg=dict(backend='nccl'))
log_processor=dict(window_size=20)
visualizer=dict(
    type='Visualizer',
    vis_backends=[dict(type='LocalVisBackend', save_dir='temp_dir')]
)

gpu_ids=[7]
device='cuda'
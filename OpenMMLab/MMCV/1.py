import torchvision
from mmcv import Config, ConfigDict
from mmcv.runner import build_optimizer, build_runner

# Image Classification
from mmcls.models import build_classifier
from mmcls.datasets import build_dataloader

from mmcv.utils.logging import get_logger

img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False
)

train_pipeline = [
    dict(type='Resize', size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

cfg = ConfigDict()
cfg.optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
cfg.data = ConfigDict(samples_per_gpu=4, workers_per_gpu=4, train=dict(pipeline=train_pipeline))

cfg.model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4, 
        out_indices=(3, ),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

cfg.workflow = [('train', 1)]
#print(cfg)

model = build_classifier(cfg.model)

# initialize model
#model = torchvision.models.resnet18(pretrained=True)
# initialize optimizer
optimizer = build_optimizer(model, cfg.optimizer)

# initialize the dataloader corresponding to the workflow(train/val)
dataset = torchvision.datasets.CIFAR10(root='/home/akiyo/nfs/zhang/dataset', train=True, download=True)
dataloaders = [build_dataloader(dataset, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu)]
    # build_dataloader(
    #     ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu,
    # ) for ds in dataset]

print(len(dataloaders))
cfg.runner = dict(type='EpochBasedRunner', max_epochs=200)

logger = get_logger('mmcv')

runner = build_runner(
    cfg.runner,
    default_args=dict(
        model=model,
        batch_processor=None,
        optimizer=optimizer,
        logger=logger
    ))

print(type(dataloaders[0]))
runner.run(dataloaders, cfg.workflow)
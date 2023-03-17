_base_ = [
    '../_base_/models/swin_transformer/base_224.py',
    #'../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/cifar10_bs16.py',
    #'../_base_/schedules/cifar10_bs128.py'

]

work_dir = '/home/akiyo/sandbox/SwinTrans_Hook'

model = dict(
    backbone=dict(_delete_=True,
        type='SwinTransformer', 
        #type='SwinTransformerLoRA', 
        arch='base', img_size=224, drop_path_rate=0.5,
        #init_cfg=dict(type='Pretrained', checkpoint="/home/akiyo/nfs/zhang/pretrained_weights/swin_base_patch4_window7_224_22k.pth"),
    ),
    head=dict(num_classes=10,),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=10, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=10, prob=0.5)
    ]),
    init_cfg=dict(
        _delete_=True,
        type='Pretrained',
        checkpoint="/home/akiyo/nfs/zhang/pretrained_weights/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth",
    ),
)

train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

custom_hooks = [dict(
    type='LoRAHook2', 
    priority='VERY_HIGH',
    apply_lora=True, lora_r=8, lora_alpha=8
)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
])

find_unused_parameters=True
#load_from='/home/akiyo/nfs/zhang/pretrained_weights/swin_base_patch4_window7_224_22k.pth'

runner = dict(type='EpochBasedRunner', max_epochs=3)
_base_ = ['CUSTM_Dataset.py']
custom_imports = dict(
    imports=[
        'mmcv.runner.optimizer', 'mmcv.runner.hooks', 'mmcv.runner.dist_utils',
    ]
)



norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        out_channels=1,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(64, 64), stride=(42, 42)))

data = dict(
    train_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4, num_gpus=1, seed=43),
    val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4, num_gpus=1)
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

workflow = [('train', 1)]#, ('val', 1)]

cudnn_benchmark = True

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=200)
#runner = dict(type='EpochBasedRunner', max_epochs=2)
checkpoint_config = dict(by_epoch=False, interval=500)

evaluation = dict(interval=50, metric='mIoU', pre_eval=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
])

gpu_ids = [0]
seed = 43
work_dir = '/home/akiyo/temp'
device = 'cuda'
hook = dict(type='TestHook', name='ZHANG')

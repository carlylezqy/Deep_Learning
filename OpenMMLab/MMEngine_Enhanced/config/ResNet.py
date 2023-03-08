# Almost copied from the Official Example
# https://mmengine.readthedocs.io/en/latest/tutorials/runner.html#best-practice-of-the-runner-config-files

# _base_ = [
#     #'mmcls::_base_/datasets/cifar10_bs16.py', 
#     'mmcls::_base_/models/resnet50.py'
# ]

_base_ = [
    '/home/akiyo/nfs/zhang/library/mmclassification/configs/_base_/datasets/cifar10_bs16.py',
    '/home/akiyo/nfs/zhang/library/mmclassification/configs/_base_/models/resnet50.py'
]


model = {{_base_.model}}
# custom_imports = dict(
#     imports=[
#         'metric.accurary'
#     ], 
#     allow_failed_imports=False
# )




# work_dir = '/home/akiyo/sandbox/ResNetConfigTrain'

# data = dict(
#     samples_per_gpu=64,
#     workers_per_gpu=2,
#     train_dataloader = dict(
#         samples_per_gpu=64,
#         workers_per_gpu=2,
#         pin_memory=True),
#     val_dataloader = dict(
#         samples_per_gpu=64,
#         workers_per_gpu=2,
#         pin_memory=True)
# )

# device='cuda'

# train_cfg = dict(by_epoch=True, max_epochs=10, val_begin=2, val_interval=1)
# val_cfg = dict()

# optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.001))
# param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[4, 8], gamma=0.1)

# val_evaluator = dict(type='Accuracy')

# default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

# launcher = 'none'
# #env_cfg = dict(cudnn_benchmark=False, backend='nccl', mp_cfg=dict(mp_start_method='fork'))
# log_level = 'INFO'
# load_from = None
# resume = False

# dist_params = dict(backend='nccl')
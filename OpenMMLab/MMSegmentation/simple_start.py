from mmcv import Config

from mmcv.runner import build_runner
from mmseg.models import build_segmentor
from mmcv.runner import HOOKS
from mmseg.core import EvalHook
from mmcv.utils import build_from_cfg
from mmseg.utils import build_dp, get_root_logger
from mmseg.datasets import build_dataloader, build_dataset
from mmcv.cnn.utils import revert_sync_batchnorm

cfg = Config.fromfile('/home/akiyo/nfs/zhang/github/Deep_Learning/OpenMMLab/MMSegmentation/config/DRIVE_UNet.py')
model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
model = revert_sync_batchnorm(model)

logger = get_root_logger(cfg.log_level)

runner = build_runner(
    cfg.runner,
    default_args=dict(
        model=model,
        batch_processor=None,
        #optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        #meta=meta
))

loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        #dist=distributed,
        seed=cfg.seed,
        drop_last=True)
loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
val_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('val_dataloader', {}),
        }
val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)

eval_cfg = cfg.get('evaluation', {})
eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'

hook_cfg = dict(type='TestHook', name='ZHANG')
priority = hook_cfg.pop('priority', 'NORMAL')
test_hook = build_from_cfg(hook_cfg, HOOKS)

runner.register_hook(EvalHook(val_dataloader, **eval_cfg), priority='LOW')
runner.register_hook(test_hook, priority=priority)

dataset = [build_dataset(cfg.data.train)]
train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

runner.run(data_loaders, cfg.workflow)

print(runner)
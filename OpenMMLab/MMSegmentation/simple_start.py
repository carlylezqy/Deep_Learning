import mmcv
from mmcv import Config

from mmcv.runner import build_runner
from mmseg.models import build_segmentor
from mmcv.runner import HOOKS
from mmseg.core import EvalHook
from mmcv.utils import build_from_cfg
from mmseg.utils import build_dp, get_root_logger
from mmseg.datasets import build_dataloader, build_dataset
from mmcv.cnn.utils import revert_sync_batchnorm

from mmcv.runner.optimizer import build_optimizer

import os; os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    cfg = Config.fromfile('config/DRIVE_UNet.py')
    
    model = build_segmentor(cfg.model)
    model = build_dp(model, device=cfg.device, device_ids=cfg.gpu_ids)
    model = revert_sync_batchnorm(model)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            work_dir=cfg.work_dir,
            logger=get_root_logger(cfg.log_level),
            optimizer=build_optimizer(model, cfg.optimizer),
    ))

    train_dataset = build_dataset(cfg.data.train)
    train_dataloader = build_dataloader(train_dataset, **cfg.data.train_dataloader)

    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(val_dataset, **cfg.data.get('val_dataloader', {}))

    eval_cfg = cfg.get('evaluation', {})
    eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'

    eval_hook = EvalHook(val_dataloader, **eval_cfg)
    runner.register_hook(eval_hook, priority='LOW')

    test_hook = build_from_cfg(cfg.hook, HOOKS)
    runner.register_hook(test_hook, priority='LOW')
    runner.register_logger_hooks(cfg.log_config)
    
    dataloader = [train_dataloader]
    runner.run(dataloader, cfg.workflow)
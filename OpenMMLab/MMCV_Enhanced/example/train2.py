import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from mmcv import Config, ConfigDict
from mmcv.parallel import MMDataParallel
from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger

from mmcls.core import EvalHook
from mmcls.datasets import build_dataset, build_dataloader

from test_hook import TestHook
import types

class Model(nn.Module):
    def __init__(self, pretrained=True):
        super(Model, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def train_step(self, data, optimizer):
        images, labels = data['img'], data['gt_label']
        predicts = self(images)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        return {'predicts': predicts, 'loss': loss}

    def val_step(self, data, optimizer):
        images, labels = data['img'], data['gt_label']
        predicts = self(images)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        return {'predicts': predicts, 'loss': loss}

    def test_step(self, data, optimizer):
        images, labels = data['img'], data['gt_label']
        predicts = self(images)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        return {'predicts': predicts, 'loss': loss}

if __name__ == '__main__': 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model = MMDataParallel(model, device_ids=[0])

    cfg = Config.fromfile('/home/akiyo/nfs/zhang/github/mmclassification/configs/_base_/datasets/cifar10_bs16.py')

    dataset = build_dataset(cfg.data.train)
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    loader_cfg = dict(num_gpus=1, dist=False, round_up=True, sampler_cfg=None,)
    loader_cfg.update({
        k: v for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 
            'train_dataloader', 'val_dataloader', 'test_dataloader'
        ]
    })

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    logger = get_logger('mmcv')
    # runner is a scheduler to manage the training
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir='/home/akiyo/sandbox/MMCV_EXAMPLE',
        logger=logger, max_epochs=4
    )

    # learning rate scheduler config
    lr_config = dict(policy='step', step=[2, 3])
    # configuration of optimizer
    optimizer_config = dict(grad_clip=None)
    # configuration of saving checkpoints periodically
    checkpoint_config = dict(interval=1)
    # save log periodically and multiple hooks can be used simultaneously
    log_config = dict(
        interval=100, 
        hooks=[
            dict(type='TextLoggerHook')
        ]
    )

    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        log_config=log_config
    )
    
    eval_config = dict(metrics=['accuracy'], greater_keys=['accuracy'])
    eval_hook = EvalHook(data_loaders[0], interval=10, **eval_config)
    test_hook = TestHook()
    
    runner.register_hook(eval_hook, priority='NORMAL')
    runner.register_hook(test_hook, priority='NORMAL')
    
    runner.run(data_loaders, [('train', 1)])

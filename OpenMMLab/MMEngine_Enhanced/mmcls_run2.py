import os; os.chdir(os.path.dirname(__file__))
from model import MMResNet50
from mmengine import Config
from mmengine.runner import Runner
from mmcls.models import build_classifier
        
if __name__ == '__main__':
    cfg_path = "config/resnet50_cifar10.py"
    config = Config.fromfile(cfg_path)
    work_dir='/home/akiyo/sandbox/ResNetConfigTrain'
    
    train_dataloader = Runner.build_dataloader(config.train_dataloader)
    model = MMResNet50()

    runner = Runner(
        model, work_dir, 
        train_dataloader, train_cfg=config.train_cfg, optim_wrapper=config.optim_wrapper,
        cfg=config
    )

    runner.train()
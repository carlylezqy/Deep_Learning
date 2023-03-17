import os; os.chdir(os.path.dirname(__file__))
from mmengine import Config
from mmengine.runner import Runner
from mmengine.hub import get_model
from mmcls.models import build_classifier
        
if __name__ == '__main__':
    cfg_path = "config/resnet50_cifar10.py"
    config = Config.fromfile(cfg_path)
    work_dir='/home/akiyo/sandbox/ResNetConfigTrain'
    
    train_dataloader = Runner.build_dataloader(config.train_dataloader)
    
    # Method 1
    model = build_classifier(config.model)

    # Method 2
    #model = get_model('mmcls::configs/resnet/resnet50_8xb16_cifar10.py')

    runner = Runner(
        model, work_dir, 
        train_dataloader, train_cfg=config.train_cfg, optim_wrapper=config.optim_wrapper,
        cfg=config
    )

    runner.train()
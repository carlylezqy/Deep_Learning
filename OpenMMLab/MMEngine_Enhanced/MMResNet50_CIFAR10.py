import os; os.chdir(os.path.dirname(__file__))

from mmengine import Config
from mmengine.runner import Runner
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset#, build_dataloader

#from mmengine.runner import build_classifier
#from mmengine.runner import build_dataset, build_dataloader

def main():

    config_path = "config/ResNet.py"
    config = Config.fromfile(config_path)

    
    model = build_classifier(config.model)
    # train_dataset = build_dataset(config.data.train)
    # val_dataset   = build_dataset(config.data.val)
    #train_dataloader = build_dataloader(train_dataset, **config.data.train_dataloader)
    #val_dataloader   = build_dataloader(val_dataset, **config.data.val_dataloader)

    # runner = Runner(
    #     model=model,
    #     work_dir=config.work_dir,
    #     train_dataloader=train_dataloader,
    #     train_cfg=config.train_cfg,
    #     val_dataloader=val_dataloader,
    #     val_cfg=config.val_cfg,
    #     val_evaluator=config.val_evaluator,
    #     optim_wrapper=config.optim_wrapper,
    # )

    # runner.train()

if __name__ == "__main__":
    main()
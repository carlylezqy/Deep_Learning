import os; os.chdir(os.path.dirname(__file__))

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import MODELS

def runner_from_cfg(cfg):
    runner = Runner(cfg)
    runner.train()

if __name__ == '__main__':
    #config_path = "config/ResNet.py"
    config_path = "config/Model_Exp.py"
    config = Config.fromfile(config_path)

    model = MODELS.build(config.model)
    #model = build_model_from_cfg(config, MODELS)
    print(model)
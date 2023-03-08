#import os; os.chdir(os.path.dirname(__file__))
#import sys; sys.path.append("/home/akiyo/nfs/zhang/library/mmclassification")

from mmengine import Config
from mmcls.models import build_classifier
from mmcls.utils import register_all_modules


if __name__ == '__main__':
    register_all_modules()

    config_path = "config/ResNet.py"
    config = Config.fromfile(config_path)
    model = build_classifier(config.model)
    
    

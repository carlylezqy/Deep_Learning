from mmcv import Config
from mmseg.datasets import build_dataset

cfg_path = "/home/akiyo/nfs/zhang/github/Deep_Learning/OpenMMLab/MMSegmentation/config/CUSTM_Dataset.py"
cfg = Config.fromfile(cfg_path)
cfg_dt = cfg.data.train
dataset = build_dataset(cfg_dt)

print(dataset[0])
for i in dataset:
    print(i)
    break
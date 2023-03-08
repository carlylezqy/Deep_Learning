from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset

if __name__ == '__main__':
    cfg = Config.fromfile('/home/akiyo/nfs/zhang/github/Deep_Learning/OpenMMLab/MMSegmentation/config/DRIVE_UNet.py')
    cfg.dump("/home/akiyo/temp/1.py")

    train_dataset = [build_dataset(cfg.data.train)]
    train_dataloader = [build_dataloader(ds, **cfg.data.train_dataloader) for ds in train_dataset][0]

    for i in train_dataloader:
        print('.')
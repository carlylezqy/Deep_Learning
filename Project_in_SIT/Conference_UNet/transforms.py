import torch
import torchvision
from torchvision.transforms import transforms
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A

class Transforms(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, size):
        self.transform = A.Compose([
            A.Resize(size, size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.MotionBlur(blur_limit=7, always_apply=False, p=0.2),
            A.GaussianBlur(blur_limit=7, always_apply=False, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomContrast(p=0.2),
            A.RandomGamma(p=0.2),
            ToTensorV2(p=1.0),
        ])

    def __call__(self, x):
        if self.n_views:
            return [self.transform(x) for i in range(self.n_views)]
        else:
            return self.transform(x)

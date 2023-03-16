import torch
import torchvision
from mmengine.model import BaseModel
from mmengine.registry import MODELS

@MODELS.register_module()
class MMResNet50(BaseModel):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': torch.nn.functional.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
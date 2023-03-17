import torch
import torchvision
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from torchvision.models import resnet50, ResNet50_Weights

@MODELS.register_module()
class MMResNet50(BaseModel):
    def __init__(self, out_features=10, pretrained=True):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc.out_features = out_features
        exit()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': torch.nn.functional.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
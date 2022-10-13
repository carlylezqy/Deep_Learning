from turtle import forward
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from torchvision.models._utils import IntermediateLayerGetter
import segmentation_models_pytorch as smp
#from model.aspp import aspp
from torchinfo import summary

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Connect2d(nn.Module):
    def __init__(self):
        super(Connect2d, self).__init__()

    def forward(self, layers, target_layers):
        diffY = layers.size()[2] - target_layers.size()[2]
        diffX = layers.size()[3] - target_layers.size()[3]
        target_layers = F.pad(target_layers, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        return torch.cat([layers, target_layers], dim=1)

class Decoder(nn.Module):
    def __init__(self, ipt_ch=512, n_classes=2):
        super(Decoder, self).__init__()
        self.connect = Connect2d()

        self.upsample = lambda ipt: F.interpolate(ipt, scale_factor=2, mode="bilinear")

        self.dconv_up1 = double_conv(ipt_ch    + ipt_ch//2,  ipt_ch//2,  ipt_ch)
        self.dconv_up2 = double_conv(ipt_ch//2 + ipt_ch//4,  ipt_ch//4,  ipt_ch//2)
        self.dconv_up3 = double_conv(ipt_ch//4 + ipt_ch//8,  ipt_ch//8,  ipt_ch//4)
        self.dconv_up4 = double_conv((ipt_ch//8) * 2      ,  ipt_ch//8,  ipt_ch//8)
        self.dconv_up5 = double_conv((ipt_ch//8) * 2      ,  ipt_ch//8,  ipt_ch//8)

        self.conv_last = nn.Conv2d(ipt_ch//8, n_classes, kernel_size=1)

        self.aspp = aspp.ASPP(input_canl=ipt_ch, num_classes=ipt_ch)
        self.center = nn.Identity()

    def forward(self, kwargs):
        # conv1  torch.Size([3, 64, 256, 256])
        # layer1 torch.Size([3, 64, 128, 128])
        # layer2 torch.Size([3, 128, 64, 64])
        # layer3 torch.Size([3, 256, 32, 32])
        # layer4 torch.Size([3, 512, 16, 16])
        
        #intermediate_layer = self.aspp(kwargs["output"])
        intermediate_layer = self.center(kwargs["output"])         # 512        (16, 16)
        ep_layer1 = self.upsample(intermediate_layer)              # 512        (32, 32)
        ep_layer1 = self.connect(ep_layer1, kwargs['layer5'])      # 512+256    (32, 32)
        ep_layer1 = self.dconv_up1(ep_layer1)                      # 768 -> 256 (32, 32)     
        
        ep_layer2 = self.upsample(ep_layer1)                       # 256        (64, 64)
        ep_layer2 = self.connect(ep_layer2, kwargs['layer4'])      # 256+128    (64, 64)
        ep_layer2 = self.dconv_up2(ep_layer2)                      # 384 -> 128 (64, 64)
        
        ep_layer3 = self.upsample(ep_layer2)                       # 128        (128, 128)
        ep_layer3 = self.connect(ep_layer3, kwargs['layer3'])      # 128+64     (128, 128)
        ep_layer3 = self.dconv_up3(ep_layer3)                      # 192 -> 64  (128, 128)
        
        ep_layer4 = self.upsample(ep_layer3)                       # 64         (256, 256)
        ep_layer4 = self.connect(ep_layer4, kwargs['layer2'])      # 64+64      (256, 256)
        ep_layer4 = self.dconv_up4(ep_layer4)                      # 128 -> 64  (256, 256)

        ep_layer5 = self.upsample(ep_layer4)                       # 64         (512, 512)
        ep_layer5 = self.connect(ep_layer5, kwargs['layer1'])      # 64+64      (512, 512)
        ep_layer5 = self.dconv_up5(ep_layer5)                      # 192 -> 64  (512, 512)

        output = self.conv_last(ep_layer5)
        return output

class UNet(nn.Module):
    def __init__(self, ipt_ch=3, n_classes=2, pretrained=True):
        super(UNet, self).__init__()
        encoder = torchvision.models.resnet34(pretrained=False)
        if pretrained:
            weights_path="/home/akiyo/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth"
            encoder.load_state_dict(torch.load(weights_path))
        self.backbone = IntermediateLayerGetter(
            encoder, {
                'relu'   : 'layer1',
                'maxpool': 'layer2',
                'layer1' : 'layer3', 
                'layer2' : 'layer4', 
                'layer3' : 'layer5', 
                'layer4' : 'output' 
            }
        )
        self.decoder = Decoder(ipt_ch=encoder.fc.in_features, n_classes=n_classes)
        self.encoder = None

    def forward(self, x):
        intermediat_value = self.backbone(x)
        output = self.decoder(intermediat_value)
        return output


def main():
    model = UNet()
    ipt = torch.randn(1, 3, 256, 256)
    output = model(ipt)
    print(output.shape)


if __name__ == "__main__":
   main()

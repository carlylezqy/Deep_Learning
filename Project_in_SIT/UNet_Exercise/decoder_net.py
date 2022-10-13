from turtle import forward
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from torchvision.models._utils import IntermediateLayerGetter

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
    def __init__(self, ipt_ch=1024, n_classes=2):
        super(Decoder, self).__init__()
        self.connect = Connect2d()

        self.convtrans1 = nn.ConvTranspose2d(ipt_ch, ipt_ch//2, kernel_size=2, stride=2)
        self.convtrans2 = nn.ConvTranspose2d(ipt_ch//2, ipt_ch//4, kernel_size=2, stride=2)
        self.convtrans3 = nn.ConvTranspose2d(ipt_ch//4, ipt_ch//8, kernel_size=2, stride=2)
        self.convtrans4 = nn.ConvTranspose2d(ipt_ch//8, ipt_ch//16, kernel_size=2, stride=2)

        # self.layer1_num = 64
        # self.layer2_num = 128
        # self.layer3_num = 256
        # self.layer4_num = 512

        self.dconv_donw = double_conv(512, 1024, 1024)

        self.dconv_up1 = double_conv(1024, 512, 512)
        self.dconv_up2 = double_conv(512, 256, 256)
        self.dconv_up3 = double_conv(256, 128, 128)
        self.dconv_up4 = double_conv(128, 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)
        self.maxpool_layer = torch.nn.MaxPool2d((2, 2))
    
    def forward(self, x, kwargs):
        # conv1 torch.Size([3, 64, 256, 256])
        # layer1 torch.Size([3, 64, 128, 128])
        # layer2 torch.Size([3, 128, 64, 64])
        # layer3 torch.Size([3, 256, 32, 32])
        # layer4 torch.Size([3, 512, 16, 16])

        layer4 = self.dconv_donw(kwargs["layer4"])
        intermediate_layer = self.maxpool_layer(layer4)
        ep_layer1 = self.convtrans1(intermediate_layer)

        ep_layer1 = self.connect(kwargs["layer4"], ep_layer1)
        ep_layer1 = self.dconv_up1(ep_layer1)

        ep_layer2 = self.convtrans2(ep_layer1)
        ep_layer2 = self.connect(kwargs['layer3'], ep_layer2)
        ep_layer2 = self.dconv_up2(ep_layer2) 

        ep_layer3 = self.convtrans3(ep_layer2)
        ep_layer3 = self.connect(kwargs['layer2'], ep_layer3)
        ep_layer3 = self.dconv_up3(ep_layer3)

        ep_layer4 = self.convtrans4(ep_layer3)
        ep_layer4 = self.connect(kwargs['layer1'], ep_layer4)
        ep_layer4 = self.dconv_up4(ep_layer4)
        
        output = self.conv_last(ep_layer4)
        
        return output

class UNet(nn.Module):
    def __init__(self, ipt_ch=3, n_classes=2):
        super(UNet, self).__init__()
        encoder = torchvision.models.resnet18(pretrained=True)
        self.backbone = IntermediateLayerGetter(
            encoder, {'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}
        )
        self.decoder = Decoder()

    def forward(self, x):
        intermediat_value = self.backbone(x)
        output = self.decoder(x, intermediat_value)
        return output

if __name__ == "__main__":
    unet = UNet()
    input_data = torch.randn(3, 3, 512, 512)
    result = unet(input_data)

    


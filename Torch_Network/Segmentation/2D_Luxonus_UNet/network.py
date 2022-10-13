import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf

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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.dconv_down1 = double_conv(n_channels, 64, 64)
        self.dconv_down2 = double_conv(64, 128, 128)
        self.dconv_down3 = double_conv(128, 256, 256)
        self.dconv_down4 = double_conv(256, 512, 512)
        self.dconv_downup = double_conv(512, 1024, 1024)        

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.connect = Connect2d()

        self.convtrans4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.convtrans3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.convtrans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dconv_up4 = double_conv(1024, 512, 512)
        self.dconv_up3 = double_conv(512, 256, 256)
        self.dconv_up2 = double_conv(256, 128, 128)
        self.dconv_up1 = double_conv(128, 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, input_data):

        cp_layer1 = self.dconv_down1(input_data)  #torch.Size([4, 64, 565, 565])
        x = self.maxpool(cp_layer1)      #torch.Size([4, 64, 282, 282])

        cp_layer2 = self.dconv_down2(x)  #torch.Size([2, 64, 280, 280])
        x = self.maxpool(cp_layer2)      #torch.Size([4, 128, 141, 141])

        cp_layer3 = self.dconv_down3(x)  #torch.Size([4, 256, 141, 141])
        x = self.maxpool(cp_layer3)      #torch.Size([4, 256, 70, 70])
        
        cp_layer4 = self.dconv_down4(x)  #torch.Size([4, 512, 70, 70])
        x = self.maxpool(cp_layer4)      #torch.Size([4, 512, 35, 35])


        c_layer5 = self.dconv_downup(x)  #torch.Size([4, 1024, 35, 35])

        #ep_layer4 = self.upsample5(c_layer5)                   #torch.Size([4, 512, 70, 70])
        ep_layer4 = self.convtrans4(c_layer5)
        ep_layer4 = self.connect(cp_layer4, ep_layer4)       #torch.Size([4, 1024, 70, 70])
        ep_layer4 = self.dconv_up4(ep_layer4)                      #torch.Size([4, 512, 70, 70])

        #ep_layer3 = self.upsample4(ep_layer4)                  #torch.Size([4, 256, 140, 140])
        ep_layer3 = self.convtrans3(ep_layer4)
        ep_layer3 = self.connect(cp_layer3, ep_layer3)
        ep_layer3 = self.dconv_up3(ep_layer3)                      #torch.Size([4, 256, 140, 140])

        #ep_layer2 = self.upsample3(ep_layer3)                  #torch.Size([4, 128, 280, 280])
        ep_layer2 = self.convtrans2(ep_layer3)
        ep_layer2 = self.connect(cp_layer2, ep_layer2)
        ep_layer2 = self.dconv_up2(ep_layer2)

        #ep_layer1 = self.upsample2(ep_layer2)   #torch.Size([4, 64, 560, 560])
        ep_layer1 = self.convtrans1(ep_layer2)
        ep_layer1 = self.connect(cp_layer1, ep_layer1)
        ep_layer1 = self.dconv_up1(ep_layer1)
        
        output = self.conv_last(ep_layer1)
        
        return output

'''
device = torch.device('cuda')
model = UNet(n_channels=1, n_classes=1).to(device)
input_data = torch.rand(10, 1, 565, 565).to(device)
model(input_data)'''
import torch
from unet import xception_unet
import segmentation_models_pytorch as smp

if __name__ == "__main__":
    model = smp.Unet(
        encoder_name="xception",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )
    ipt = torch.randn(1, 3, 512, 512)
    out = model(ipt)
    print(model)
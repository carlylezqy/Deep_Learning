import torch
import segmentation_models_pytorch as smp

if __name__ == "__main__":
    input_data = torch.randn(3, 3, 512, 512)
    model = smp.Unet(
        encoder_name="vgg16_bn",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,        # model output channels (number of classes in your dataset)
        #decoder_attention_type="scse",
    )

    output = model(input_data)
    
    print(model)
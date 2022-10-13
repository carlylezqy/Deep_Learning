import timm
import torch
import torchvision
import decoder_net

if __name__ == '__main__':
    model = timm.create_model("resnet101", pretrained=True)
    print(model)
    encoder = torchvision.models._utils.IntermediateLayerGetter(
        model, 
        return_layers={"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}
    )
    #print(encoder["layer1"], encoder["layer2"], encoder["layer3"], encoder["layer4"])
    x = torch.randn(1, 3, 512, 512).cuda()
    encoder = encoder.cuda()
    decoder = decoder_net.Decoder(n_classes=1).cuda()

    opt = encoder(x)
    print(opt.keys())

    result = decoder(x, opt)
    print(result.shape)
    
    #print(x.shape, opt["layer1"].shape, opt["layer2"].shape, opt["layer3"].shape, opt["layer4"].shape)
    # torch.Size([1, 3, 512, 512]) 
    # torch.Size([1, 256, 128, 128]) 
    # torch.Size([1, 512, 64, 64]) 
    # torch.Size([1, 1024, 32, 32]) 
    # torch.Size([1, 2048, 16, 16])










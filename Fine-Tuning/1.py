import torch
import numpy as np
import loralib as lora
from timm_new.models.vision_transformer import VisionTransformer

model = VisionTransformer(
    img_size=224, patch_size=16, in_chans=3, 
    embed_dim=768, depth=12, num_heads=12,
    apply_lora=False, lora_r=8, lora_alpha=8)

x = torch.randn(1, 3, 224, 224)
result = model(x)
lora.mark_only_lora_as_trainable(model)
print(result.shape)

from torchsummary import summary
summary(model, (3, 224, 224))

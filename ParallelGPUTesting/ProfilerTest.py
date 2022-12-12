import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(inputs)
    print(prof.table())

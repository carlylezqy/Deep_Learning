import torch

class DUpsampling(torch.nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding = pad,bias=False)
        self.scale = scale
    
    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        
        return x

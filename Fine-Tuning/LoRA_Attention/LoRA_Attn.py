import torch
import loralib as lora
from timm.models.layers import Mlp, DropPath

class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., apply_lora=False, lora_r=0, lora_alpha=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if apply_lora:
            self.qkv = lora.Linear(dim, dim * 3, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
        else:
            self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)

        if apply_lora:
            self.proj = lora.Linear(dim, dim, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
        else:
            self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale(torch.nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = torch.nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(torch.nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm, apply_lora=False, lora_r=0, lora_alpha=0):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, apply_lora=apply_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=torch.nn.GELU, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., apply_lora=False, lora_r=0, lora_alpha=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if apply_lora:
            self.qkv = lora.Linear(dim, dim * 3, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
        else:
            self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)

        if apply_lora:
            self.proj = lora.Linear(dim, dim, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
        else:
            self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    depth = 12

    dict = dict(
        dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
        drop=0., attn_drop=0., apply_lora=True, lora_r=8, lora_alpha=8
    )

    model = torch.nn.Sequential(*[Block(**dict) for i in range(depth)])
    
    x = torch.randn(1, 197, 768)
    result = model(x)
    print(result.shape)
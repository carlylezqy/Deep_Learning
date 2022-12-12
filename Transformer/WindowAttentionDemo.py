import torch
from typing import Optional
from timm.models.layers import trunc_normal_

def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

class WindowAttention(torch.nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        win_h, win_w = window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = torch.nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))
        #print(self.relative_position_bias_table.shape) #torch.Size([169, 3])
    
        # get pair-wise relative position index for each token inside the window
        relative_position_index = get_relative_position_index(win_h, win_w)
        #print(relative_position_index.shape) #torch.Size([49, 49])
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv       = torch.nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj      = torch.nn.Linear(attn_dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = torch.nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        #self.relative_position_index.view(-1).shape #[49, 49] --> [2401]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        
        #[169, 3] --> [2401, 3] --> [49, 49, 3]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv: torch.Size([3, 8, 3, 49, 32])
        #   q: torch.Size([8, 3, 49, 32])
        #   k: torch.Size([8, 3, 49, 32])
        #   v: torch.Size([8, 3, 49, 32])
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # attn:    torch.Size([8, 3, 49, 49])
        # rp_bias: torch.Size([1, 3, 49, 49])
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

x = torch.randn(8, 49, 96)
window_attention = WindowAttention(dim=96, num_heads=3, head_dim=None, window_size=(7, 7), qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
output = window_attention(x)
print(output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
from inspect import isfunction
import math
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    """Apply multi-head cross-attention between image and condition features."""
    def __init__(self, query_dim, heads=4, dim_head=16, context_dim=None, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        context_dim = context_dim if context_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        b, c, H, W = x.shape
        n = H * W

        x_flat = rearrange(x, "b c h w -> b (h w) c")  # (b, n, c)

        if context is None:
            context = x_flat  # self-attn fallback

        # ensure context has token dim: (b, m, ctx_dim)
        if context.dim() == 2:
            context = context.unsqueeze(1)

        q = self.to_q(x_flat)        # (b, n, inner)
        k = self.to_k(context)       # (b, m, inner)
        v = self.to_v(context)       # (b, m, inner)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b,h,n,m)
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(attn, v)                              # (b,h,n,d)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)                                    # (b,n,c)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

        return x + out


class SelfAttention(nn.Module):
    """Apply memory-efficient self-attention to spatial features."""
    def __init__(self, in_channels, heads=4, head_dim=16, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim

        self.norm = Normalize(in_channels)

        # Note: qkv outputs inner_dim instead of in_channels
        self.qkv = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1, bias=False)

        # project the output back to in_channels for the residual add
        self.proj_out = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1, bias=False)

        self.dropout = dropout

    def forward(self, x):
        b, c, H, W = x.shape
        h_ = self.norm(x)

        qkv = self.qkv(h_)                      # (b, 3*inner_dim, H, W)
        q, k, v = qkv.chunk(3, dim=1)           # each: (b, inner_dim, H, W)

        n = H * W
        # (b, heads, n, head_dim)
        q = q.reshape(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = k.reshape(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)
        v = v.reshape(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )  # (b, heads, n, head_dim)

        out = out.permute(0, 1, 3, 2).reshape(b, self.inner_dim, H, W)  # (b, inner_dim, H, W)
        out = self.proj_out(out)                                        # (b, in_channels, H, W)
        return x + out


class LinearAttention(nn.Module):
    """
    Linear Attention for feature maps (B, C, H, W)
    Complexity: ~O(B * heads * (H*W) * head_dim^2)  with small head_dim (e.g. 32)
    Memory: ~O(B * heads * (H*W) * head_dim)  (no N x N attention matrix)
    """
    def __init__(self, in_channels, heads=4, head_dim=32, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim

        self.norm = Normalize(in_channels)

        # Note: inner_dim is decoupled from in_channels so head_dim can stay fixed
        self.to_qkv = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(self.inner_dim, in_channels, kernel_size=1, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, H, W = x.shape
        n = H * W

        x_in = x
        x = self.norm(x)

        qkv = self.to_qkv(x)  # (b, 3*inner_dim, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # each: (b, inner_dim, H, W)

        # reshape to (b, heads, head_dim, n)
        q = q.reshape(b, self.heads, self.head_dim, n)
        k = k.reshape(b, self.heads, self.head_dim, n)
        v = v.reshape(b, self.heads, self.head_dim, n)

        # key: apply softmax on different dims to linearize attention
        # common stable form: softmax q over head_dim; softmax k over tokens (n)
        q = F.softmax(q, dim=2)          # over channels per head (head_dim)
        k = F.softmax(k, dim=3)          # over spatial positions (n)

        # optional: scaling v helps stability (not required)
        v = v / (n ** 0.5)

        # context: (b, heads, head_dim, head_dim)
        # einsum: sum over n
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # out: (b, heads, head_dim, n)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        out = out.reshape(b, self.inner_dim, H, W)
        out = self.to_out(out)
        return x_in + out


class Upsample(nn.Module):
    """Upsample a feature map by a factor of two."""
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x



class Downsample(nn.Module):
    """Downsample a feature map by a factor of two."""
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """Apply a residual convolutional block with timestep conditioning."""
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def get_timestep_embedding(timesteps, embedding_dim):

    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb



def window_partition(x, window_size: int):
    """
    x: (B, C, H, W)
    return: windows (B * num_windows, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0
    x = x.view(B, C,
               H // window_size, window_size,
               W // window_size, window_size)
    # (B, num_h, num_w, ws, ws, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(B * (H // window_size) * (W // window_size),
                     window_size * window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int, B: int):
    """
    windows: (B * num_windows, window_size*window_size, C)
    return: (B, C, H, W)
    """
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x


def build_shift_attn_mask(Hp, Wp, ws, shift, device):
    # (1, 1, Hp, Wp) as a region-id map
    img_mask = torch.zeros((1, 1, Hp, Wp), device=device)
    cnt = 0
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt
            cnt += 1

    # after window partition: (num_win, T, 1) -> (num_win, T)
    mask_windows = window_partition(img_mask, ws).squeeze(-1)

    # diff: 0 within the same region, non-zero across regions
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_win, T, T)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, 0.0)
    # for SDPA: broadcasts to (B*nwin, heads, T, T)
    return attn_mask.unsqueeze(1)  # (num_win, 1, T, T)


class ShiftedWindowSelfAttention2D(nn.Module):
    """Apply shifted-window self-attention to two-dimensional features."""
    def __init__(self, in_channels, window_size=8, shift_size=0, heads=4, head_dim=16, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.shift_size = shift_size
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.dropout = dropout

        self.norm = Normalize(in_channels)
        self.qkv = nn.Conv2d(in_channels, 3 * self.inner_dim, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(self.inner_dim, in_channels, 1, bias=False),
            nn.Dropout(dropout)
        )

        # simple cache (rebuilt when the resolution changes)
        self._mask_cache = {}

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        shift = self.shift_size % ws

        # pad to a multiple of ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        x_in = x
        x = self.norm(x)

        # cyclic shift
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q_w = window_partition(q, ws)
        k_w = window_partition(k, ws)
        v_w = window_partition(v, ws)

        tokens = ws * ws
        q_w = q_w.view(-1, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k_w = k_w.view(-1, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v_w = v_w.view(-1, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn_mask = None
        if shift > 0:
            key = (Hp, Wp, ws, shift, x.device.type)
            if key not in self._mask_cache:
                self._mask_cache[key] = build_shift_attn_mask(Hp, Wp, ws, shift, x.device)
            base_mask = self._mask_cache[key]  # (num_win, 1, T, T)

            num_win = (Hp // ws) * (Wp // ws)
            # repeat across batch: (B*num_win, 1, T, T)
            attn_mask = base_mask.repeat(B, 1, 1, 1)

        out_w = F.scaled_dot_product_attention(
            q_w, k_w, v_w,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        out_w = out_w.permute(0, 2, 1, 3).contiguous().view(-1, tokens, self.inner_dim)
        out = window_reverse(out_w, ws, Hp, Wp, B)

        # reverse shift
        if shift > 0:
            out = torch.roll(out, shifts=(shift, shift), dims=(2, 3))

        out = self.proj(out)
        out = x_in + out

        # unpad
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out

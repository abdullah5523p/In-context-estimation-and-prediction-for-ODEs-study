import torch
from torch import nn
from torch.nn import functional as F
from .families import SPECS


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True)+1e-6) * self.gain


class FullWidthAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        # Every consecutive width-sized output slice is a separate square head matrix.
        self.q = nn.Linear(width, heads*width, bias=False)
        self.k = nn.Linear(width, heads*width, bias=False)
        self.v = nn.Linear(width, heads*width, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = [layer(x).view(b, n, self.heads, self.width).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]
        # Row-token convention internally: softmax(Q K^T / sqrt(width)) V.
        # Equivalent to column-token V X softmax_col((K X)^T Q X).
        heads = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return heads.sum(dim=1)  # No concatenation or trainable output projection.


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ff_norm = RMSNorm(cfg.width)
        self.attn_norm = RMSNorm(cfg.width)
        self.fc1 = nn.Linear(cfg.width, cfg.ff_width, bias=False)
        self.fc2 = nn.Linear(cfg.ff_width, cfg.width, bias=False)
        self.attn = FullWidthAttention(cfg.width, cfg.heads)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.dropout(self.fc2(F.relu(self.fc1(self.ff_norm(x))).square()))
        return x + self.dropout(self.attn(self.attn_norm(x)))


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        spec = SPECS[cfg.family]
        m, p = len(spec["scale"]), len(spec["low"])
        din, dout = 1+m+(cfg.task == "prediction"), p if cfg.task == "parameter" else m
        self.input = nn.Linear(din, cfg.width, bias=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = RMSNorm(cfg.width)
        self.output = nn.Linear(cfg.width, dout, bias=False)

    def forward(self, points):
        x = self.input(points)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


def all_token_loss(outputs, targets):
    return (outputs-targets[:, None, :]).square().mean()

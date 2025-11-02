import torch
import math
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _split(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def _merge(self, x):
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.h * self.d_k)

    def forward(self, x, mask=None):
        Q, K, V = self._split(self.Wq(x)), self._split(self.Wk(x)), self._split(self.Wv(x))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return self.Wo(self._merge(out))

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2).unsqueeze(0)
        div = torch.exp(-math.log(10000.0) * i / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)
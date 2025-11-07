import torch
import math
import torch.nn as nn
from typing import Optional, Tuple

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    return out, attn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (self- and cross-attention).
    用法：
      自注意力：y = mha(x, mask=mask)
      交叉注意：y = mha(q, k=enc_out, v=enc_out, mask=mask)

    约定：
      - 输入 Q/K/V: [B, T, d_model]
      - mask: 期望为 [B, 1, T_q, T_k] 或可广播到该形状；1=可见，0=遮挡
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        # 线性投影（对 Q/K/V 均使用 d_model -> d_model）
        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wk = nn.Linear(d_model, d_model, bias=bias)
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.Wo = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    # ---- 形状转换工具 ----
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, d_model] -> [B, h, T, d_k]
        B, T, _ = x.shape
        x = x.view(B, T, self.h, self.d_k).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, h, T, d_k] -> [B, T, d_model]
        x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.h * self.d_k)
        return x

    @staticmethod
    def _prepare_mask(mask: Optional[torch.Tensor],
                      B: int, Tq: int, Tk: int, device, dtype) -> Optional[torch.Tensor]:
        """
        统一将 mask 变为 [B, 1, T_q, T_k] 的布尔张量（True=可见/1；False=遮挡/0）。
        允许传入：
          - None
          - [B, 1, T_q, T_k]
          - [B, 1, 1, T_k] （仅 padding）
          - [1, 1, T_q, T_k]（仅 causal）
        """
        if mask is None:
            return None
        # 转 bool（True 表示“保留/可见”）
        mask = mask.to(device=device)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # 广播到目标形状
        if mask.dim() == 2:  # 兼容极简 [B, Tk]
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,Tk]
        mask = mask.expand(B, 1, Tq, Tk)  # 若本就可广播，这里不会复制内存
        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
          q: [B, T_q, d_model]
          k: [B, T_k, d_model]，默认与 q 相同（自注意力）
          v: [B, T_k, d_model]，默认与 k 相同
          mask: 可广播到 [B, 1, T_q, T_k] 的布尔张量（True=可见，False=遮挡）
          return_attn: 是否返回注意力权重 [B, h, T_q, T_k]
        返回：
          y: [B, T_q, d_model]（以及可选的 attn）
        """
        B, Tq, _ = q.shape
        if k is None:  # self-attention
            k = q
        if v is None:
            v = k
        Tk = k.size(1)

        # 1) 线性映射
        Q = self._split_heads(self.Wq(q))     # [B,h,T_q,d_k]
        K = self._split_heads(self.Wk(k))     # [B,h,T_k,d_k]
        V = self._split_heads(self.Wv(v))     # [B,h,T_k,d_k]

        # 2) 注意力分数
        # 用更高精度计算 softmax（避免 fp16 下数值不稳），计算后再回到原 dtype
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,h,T_q,T_k]

        # 3) 掩码
        m = self._prepare_mask(mask, B, Tq, Tk, device=scores.device, dtype=scores.dtype)
        if m is not None:
            scores = scores.masked_fill(~m, float("-inf"))

        # 4) softmax + dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # 5) 加权求和并合并头
        y = attn @ V                               # [B,h,T_q,d_k]
        y = self._merge_heads(y)                   # [B,T_q,d_model]
        y = self.proj_drop(self.Wo(y))             # [B,T_q,d_model]

        if return_attn:
            return y, attn
        return y

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
import torch
import torch.nn as nn
from .layers import MultiHeadAttention, PositionwiseFFN, PositionalEncoding
#encoder 代码
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        h = self.pos(self.embed(x))
        for blk in self.blocks:
            h = blk(h, mask)
        return self.norm(h)

class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, h):
        return self.proj(h)

#decoder代码
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_cross_attn: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_cross_attn = use_cross_attn

        # 说明：当 use_cross_attn=False 时，前向传播会跳过交叉注意力残差分支，实现“去条件化”的解码器。

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1️⃣ Decoder Self-Attention (masked)
        x = x + self.dropout(self.self_attn(self.ln1(x), mask=tgt_mask))
        # 2️⃣ Cross-Attention（可选）: query 来自 decoder，key/value 来自 encoder
        if self.use_cross_attn and enc_out is not None:
            x = x + self.dropout(self.cross_attn(self.ln2(x), k=enc_out, v=enc_out, mask=src_mask))
        # 3️⃣ Feed Forward
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x
        
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_len=512, use_cross_attn: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, use_cross_attn=use_cross_attn) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        h = self.pos(self.embed(tgt))
        for blk in self.blocks:
            h = blk(h, enc_out, src_mask, tgt_mask)
        return self.norm(h)

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_len=512,
        enc_num_layers: int | None = None,
        dec_num_layers: int | None = None,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        enc_layers = enc_num_layers if enc_num_layers is not None else num_layers
        dec_layers = dec_num_layers if dec_num_layers is not None else num_layers
        self.use_cross_attn = use_cross_attn
        self.encoder = TransformerEncoder(src_vocab, d_model, enc_layers, num_heads, d_ff, max_len)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, dec_layers, num_heads, d_ff, max_len, use_cross_attn=use_cross_attn)
        self.proj = nn.Linear(d_model, tgt_vocab)

    def make_masks(self, src, tgt, pad_id=0):
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,src_T]
        tgt_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,tgt_T]
        T = tgt.size(1)
        causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
        tgt_mask = tgt_mask & causal.unsqueeze(0).unsqueeze(1)
        return src_mask, tgt_mask

    def forward(self, src, tgt, pad_id=0):
        src_mask, tgt_mask = self.make_masks(src, tgt, pad_id)
        enc_out = self.encoder(src, src_mask) if self.use_cross_attn else None
        dec_out = self.decoder(tgt, enc_out, src_mask if self.use_cross_attn else None, tgt_mask)
        return self.proj(dec_out)


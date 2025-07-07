import torch
from torch import nn
from .attention import TASA_attention
from .modules import ConvolutionFrontEnd, ConvolutionBlock, FeedForwardBlock
from .model import R_TASA_Transformer


vocab_size = 100
n_layers = 2
d_model = 32
d_ff = 64
h = 4
p_dropout = 0.1
in_features = 40
batch_size = 2
seq_len_enc = 15
seq_len_dec = 27

# ==== Khởi tạo model ====
model = R_TASA_Transformer(
    in_features=in_features,
    n_layers=n_layers,
    d_model=d_model,
    d_ff=d_ff,
    h=h,
    p_dropout=p_dropout,
    vocab_size=vocab_size
)

# ==== Tạo dữ liệu đầu vào ====
src = torch.randn(batch_size, seq_len_enc, in_features)                   # encoder input
tgt = torch.randint(0, vocab_size, (batch_size, seq_len_dec))            # decoder input

# ==== Tạo mask ====
src_mask = torch.ones(batch_size, seq_len_enc)           # [B, 1, M, T]
tgt_mask = torch.ones(batch_size, 1, seq_len_dec, seq_len_dec)           # [B, 1, M, M]

# ==== Chạy forward ====
with torch.no_grad():
    out = model(src, tgt, src_mask, tgt_mask)

# ==== In kết quả ====
print("✅ Decoder input shape:", tgt.shape)
print("✅ Encoder input shape:", src.shape)
print("✅ Output shape:", out.shape)  # Expect: [B, M, vocab_size]
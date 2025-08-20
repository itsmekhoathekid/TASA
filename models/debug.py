import torch
from torch import nn
from attention import TASA_attention
from modules import ConvolutionFrontEnd , FeedForwardBlock, ConvDecBlock, ConvDec


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

# ==== Tạo dữ liệu đầu vào ====
src = torch.randn(batch_size, seq_len_enc, in_features)                   # encoder input
tgt = torch.randint(0, vocab_size, (batch_size, seq_len_dec))            # decoder input

# ==== Tạo mask ====
src_mask = torch.ones(batch_size, seq_len_enc)           # [B, 1, M, T]
tgt_mask = torch.ones(batch_size, 1, seq_len_dec, seq_len_dec)           # [B, 1, M, M]

conv_dec = ConvDec(
    num_blocks=3,
    in_channels=1,
    out_channels=[8, 16, 32],
    kernel_sizes=[3, 3, 3],
)

tgt = tgt.unsqueeze(1)  # Thêm chiều kênh cho ConvDecBlock, từ [B, M] thành [B, 1, M]
print("Input shape to ConvDecBlock:", tgt.shape)
x = conv_dec(tgt) 
print("Output shape after ConvDecBlock:", x.shape)






from .encoder import TASA_encoder
from .decoder import TransformerDecoder
from torch import nn
import torch

class R_TASA_Transformer(nn.Module):
    def __init__(self, in_features, n_layers, d_model, d_ff, h, p_dropout, vocab_size):
        super().__init__()
        self.encoder = TASA_encoder(
            in_features=in_features,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            p_dropout=p_dropout
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            ff_size=d_ff,
            h=h,
            p_dropout=p_dropout
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out, _ = self.encoder(src.float(), src_mask)  # [B, T, d_model]
        
        
        # print("Encoder output shape:", enc_out.shape)  # [B, T, d_model]
        # print("Encoder mask shape:", src_mask.shape)  # [B, T]

        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return dec_out # [B, M, vocab_size]

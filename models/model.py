from .encoder import TASA_encoder
from .decoder import TransformerDecoder
from torch import nn
import torch

class R_TASA_Transformer(nn.Module):
    def __init__(self, in_features, n_enc_layers, n_dec_layers, d_model, ff_size, h, p_dropout, vocab_size):
        super().__init__()
        self.encoder = TASA_encoder(
            in_features=in_features,
            n_layers=n_enc_layers,
            d_model=d_model,
            d_ff=ff_size,
            h=h,
            p_dropout=p_dropout
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            n_layers=n_dec_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            p_dropout=p_dropout
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out, mask = self.encoder(src.float(), src_mask)  # [B, T, d_model]
        enc_input_lengths = torch.sum(mask, dim=1) # [B]
        
        # print("Encoder output shape:", enc_out.shape)  # [B, T, d_model]
        # print("Encoder mask shape:", src_mask.shape)  # [B, T]

        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return enc_out, dec_out, enc_input_lengths 

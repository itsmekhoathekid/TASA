from .encoder import TASA_encoder, TransformerEncoder
from .decoder import TransformerDecoder
from torch import nn
import torch

class Transformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        if config['attention_type'] == 'r_tasa':
            self.encoder = TASA_encoder(
                in_features=config['in_features'],
                n_layers=config['n_enc_layers'],
                d_model=config['d_model'],
                d_ff=config['ff_size'],
                h=config['h'],
                p_dropout=config['p_dropout']
            )
        else:
            self.encoder = TransformerEncoder(
                in_features=config['in_features'],
                n_layers=config['n_enc_layers'],
                d_model=config['d_model'],
                ff_size=config['ff_size'],
                h=config['h'],
                p_dropout=config['p_dropout']
            )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            n_layers=config['n_dec_layers'],
            d_model=config['d_model'],
            ff_size=config['ff_size'],
            h=config['h'],
            p_dropout=config['p_dropout'],
            k = config['k']
        )
        self.ctc_lin = nn.Linear(config['d_model'], vocab_size)
        self.model_name = config['model_name']

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out, mask = self.encoder(src.float(), src_mask)  # [B, T, d_model]
        enc_input_lengths = torch.sum(mask, dim=1) # [B]

        dec_out = self.decoder(tgt, enc_out, mask, tgt_mask)
        enc_out = self.ctc_lin(enc_out)  # [B, T, vocab_size]
        enc_out = enc_out.log_softmax(dim=-1) 
        return enc_out, dec_out, enc_input_lengths 
    
    def encode(self, src, src_mask):
        """
        Encode the input sequence.
        Args:
            src (Tensor): Input sequence tensor of shape (B, T, in_features).
            src_mask (Tensor): Mask for the input sequence of shape (B, T).
        Returns:
            Tensor: Encoded output of shape (B, T, d_model).
        """
        enc_out, mask = self.encoder(src.float(), src_mask)
        return enc_out, mask

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        """
        Decode the target sequence.
        Args:
            tgt (Tensor): Target sequence tensor of shape (B, U).
            enc_out (Tensor): Encoded output from the encoder of shape (B, T, d_model).
            src_mask (Tensor): Mask for the input sequence of shape (B, T).
            tgt_mask (Tensor): Mask for the target sequence of shape (B, U).
        Returns:
            Tensor: Decoded output of shape (B, U, vocab_size).
        """
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return dec_out




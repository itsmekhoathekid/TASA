import torch
from torch import nn
from .attention import MultiHeadAttentionBlock
from .modules import FeedForwardBlock, ResidualConnection, ProjectionLayer, PositionalEncoding

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ff_size: int, dropout: float) -> None:
        super().__init__()
        self.ffn = FeedForwardBlock(d_model=d_model, d_ff=ff_size, dropout=dropout)
        self.self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        self.cross_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        self.residual_connections =  nn.ModuleList([
            ResidualConnection(features=d_model, dropout=dropout),
            ResidualConnection(features=d_model, dropout=dropout),
            ResidualConnection(features=d_model, dropout=dropout)
        ])

        

    
    def forward(self, x, encoder_out, enc_mask, dec_mask):
        
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, dec_mask))

        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_out, encoder_out, enc_mask))
        
        x = self.residual_connections[2](x, lambda x: self.ffn(x))
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, ff_size: int, h: int, p_dropout: float) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model=d_model) 
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, h=h, ff_size=ff_size, dropout=p_dropout) for _ in range(n_layers)]
        )
        self.projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        """Passes the input `x` through the decoder layers.

        Args:
            x (Tensor): The input tensor of shape [B, M]
            encoder_out (Tensor): The output from the encoder of shape [B, T, d_model]
            enc_mask (Tensor): The mask for the encoder output of shape [B, T]
            dec_mask (Tensor): The mask for the decoder input of shape [B, M]

        Returns:
            Tensor: The decoded output of shape [B, M, d_model].
        """
        out = self.emb(x)
        out = self.pe(out)
        for layer in self.layers:
            out = layer(out, encoder_out, enc_mask, dec_mask)
        out = self.projection(out)
        return out




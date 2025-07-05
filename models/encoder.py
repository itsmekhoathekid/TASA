import torch
from torch import nn
from modules import ConvolutionFrontEnd, FeedForwardBlock
from attention import TASA_attention

class TASA_layers(nn.Module):
    def __init__(self, in_features, n_layers, d_model, d_ff, h, p_dropout):
        super().__init__()
        self.in_features = in_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.p_dropout = p_dropout

        self.attention = TASA_attention(
            d_model=d_model,
            h=h,
            dropout=p_dropout
        )

        self.ffn = FeedForwardBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=p_dropout
        )
    
    def forward(self, x, mask=None, previous_attention_scores=None):
        """
        x: [batch, time, features]
        mask: [batch, time]
        previous_attention_scores: [batch, h, time, time]
        """

        x, atten_score = self.attention(x, x, x, mask, previous_attention_scores)
        x = self.ffn(x)

        return x, atten_score
    

class TASA_encoder(nn.Module):
    def __init__(self, in_features, n_layers, d_model, d_ff, h, p_dropout):
        super().__init__()
        self.in_features = in_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.p_dropout = p_dropout

        self.frontend = ConvolutionFrontEnd(
            in_channels=1,
            num_blocks=2,
            num_layers_per_block=1,
            out_channels=[64, 32],
            kernel_size=3,
            stride=1,
            dilation=1,
            residuals=[False, False],
            activation=nn.ReLU,
            norm=nn.BatchNorm2d,
            dropout=p_dropout
        )

        self.layers = nn.ModuleList(
            [
                TASA_layers(
                    in_features=in_features,
                    n_layers=n_layers,
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    p_dropout=p_dropout
                ) for _ in range(n_layers)
            ]
        )
    
    def forward(self, x, mask=None, previous_attention_scores=None):
        
        x = x.unsqueeze(1)  # [batch, channels, time, features]
        x = self.frontend(x)  # [batch, channels, time, features]

        x = x.transpose(1, 2).contiguous()  
        x = x.reshape(x.shape[0], x.shape[1], -1)

        for layer in self.layers:
            x, previous_attention_scores = layer(x, mask, previous_attention_scores)
        
        return x 
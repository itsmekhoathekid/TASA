import torch
from torch import nn
from .modules import ConvolutionFrontEnd, FeedForwardBlock, ResidualConnection
from .attention import TASA_attention

def calc_data_len(
    result_len: int,
    pad_len,
    data_len,
    kernel_size: int,
    stride: int,
):
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len

def get_mask_from_lens(lengths, max_len: int):
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)

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

        self.residual = ResidualConnection(
            features=d_model,
            dropout=p_dropout
        )
    def forward(self, x, mask=None, previous_attention_scores=None):
        """
        x: [batch, time, features]
        mask: [batch, time]
        previous_attention_scores: [batch, h, time, time]
        """
        

        x, atten_score = self.attention(x, x, x, mask, previous_attention_scores)
        x = self.residual(x, lambda x: self.ffn(x))  # Residual connection
        # x = self.ffn(x)

        return x, atten_score
    

class TASA_encoder(nn.Module):
    def __init__(self, in_features, n_layers, d_model, d_ff, h, p_dropout):
        super().__init__()
        self.input_embed = nn.Embedding(num_embeddings=in_features, embedding_dim=d_model)
        self.in_features = in_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.p_dropout = p_dropout

        self.num_blocks = 2
        self.num_layers_per_block = 1

        self.frontend = ConvolutionFrontEnd(
            in_channels=1,
            num_blocks=3,
            num_layers_per_block=2,
            out_channels=[8, 16, 32],
            kernel_sizes=[3, 3, 3],
            strides=[1, 2, 2],
            residuals=[True, True, True],
            activation=nn.ReLU,        
            norm=nn.BatchNorm2d,            
            dropout=0.1,
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

        self.projection = nn.Linear(in_features, d_model)

    
    def forward(self, x, mask=None, previous_attention_scores=None):
        x = x.unsqueeze(1)  # [batch, channels, time, features]
        # print("x shape before frontend:", x.shape)  # [batch, 1, time, features]
        x, mask = self.frontend(x, mask)  # [batch, channels, time, features]
        # print("x shape after frontend:", x.shape)
        x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, C * features]
        # print("x shape after reshape:", x.shape)
        
        
        

        # print("x shape after frontend:", x.shape)  # [batch, time, C * features]
        # print("mask shape after frontend:", mask.shape)  # [batch, time]
        x = self.projection(x)  # [batch, time, d_model]

        for layer in self.layers:
            
            x, previous_attention_scores = layer(x, mask, previous_attention_scores)
            # print("previous_attention_scores shape:", previous_attention_scores.shape)  # [batch, h, time, time]
        
        # x shape : # [batch, time', d_model]
        return x , mask



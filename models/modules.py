import torch
import torch.nn as nn
from typing import Optional, Callable, Type, List

class FeedForwardBlock(nn.Module):
    """
    A feed-forward block with two linear layers and a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        residual = x
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x + residual  # Add the residual connection

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        residual: bool = False,
        conv_module: Type[nn.Module] = nn.Conv2d,
        activation: Callable = nn.ReLU,
        norm: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            conv_stride = stride if i == num_layers - 1 else 1
            conv = conv_module(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=conv_stride,
                dilation=dilation,
                padding=(kernel_size // 2)  # mimic "same" padding
            )
            layers.append(conv)
            if norm:
                layers.append(norm(out_channels))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)
        self.residual = residual

        if residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm(out_channels) if norm else nn.Identity(),
                nn.Dropout(dropout),
            )
        elif residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.main(x)
        if self.residual:
            out = out + self.shortcut(x)
        return out

class ConvolutionFrontEnd(nn.Module):
    def __init__(
        self,
        in_channels : int, 
        num_blocks: int,
        num_layers_per_block: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        residuals: List[bool],
        activation: Callable = nn.ReLU,
        norm: Optional[Callable] = nn.BatchNorm2d,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []

        for i in range(num_blocks):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels[i],
                num_layers=num_layers_per_block,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                residual=residuals[i],
                activation=activation,
                norm=norm,
                dropout=dropout
            )
            blocks.append(block)
            in_channels = out_channels[i]  # cập nhật cho block sau

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x # batch, channels, time, features

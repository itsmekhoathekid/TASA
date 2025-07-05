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
            kernel_sizes=[3, 3],
            strides=[2, 2],
            residuals=[False, False],
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

        self.linear_proj = nn.Linear(640, d_model)
    
    def forward(self, x, mask=None, previous_attention_scores=None):
        
        x = x.unsqueeze(1)  # [batch, channels, time, features]
        x = self.frontend(x)  # [batch, channels, time, features]

        x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, features]

        print("Input shape:", x.shape)

        x = self.linear_proj(x)  # [batch, time, d_model]

        for layer in self.layers:
            
            x, previous_attention_scores = layer(x, mask, previous_attention_scores)
        
        return x 

# Các tham số mô hình
in_features = 80     # số feature đầu vào
n_layers = 4         # số tầng attention
d_model = 256        # chiều không gian attention
d_ff = 512           # hidden size FFN
h = 4                # số head
p_dropout = 0.1

# Khởi tạo mô hình
model = TASA_encoder(
    in_features=in_features,
    n_layers=n_layers,
    d_model=d_model,
    d_ff=d_ff,
    h=h,
    p_dropout=p_dropout
)

# Đưa mô hình lên GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dữ liệu giả
batch_size = 8
time_steps = 100
x = torch.randn(batch_size, time_steps, in_features).to(device)  # [B, T, F]

# Mặt nạ (mask) và attention scores giả (tuỳ chọn)
mask = torch.ones(batch_size, time_steps // 4, dtype=torch.bool).to(device)
previous_attention_scores = None  # hoặc: torch.zeros(batch_size, h, time_steps, time_steps).to(device)

# Forward pass
with torch.no_grad():
    output = model(x, mask=mask, previous_attention_scores=previous_attention_scores)

print("✅ Test passed. Output shape:", output.shape)
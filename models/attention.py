import torch
import torch.nn as nn

class TASA_attention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h  
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.transmit_module = nn.Conv2d(
            in_channels = h, 
            out_channels = h,
            kernel_size= (3,3),
            padding = 1
        )

        self.aggregate_module = nn.Conv2d(
            in_channels= h * 2,
            out_channels= h,
            kernel_size= (3,3),
            padding = 1
        )

    
    def attention(self, query, key, value, mask, dropout, previous_attention_scores):
        M = torch.matmul(query, key.transpose(-2, -1)) # (batch, h, seq_len, seq_len)
        Mt = self.transmit_module(previous_attention_scores)

        Ma = self.aggregate_module(torch.cat((M, Mt), dim=1)) # (batch, h, seq_len, seq_len)

        A = torch.softmax(Ma / torch.sqrt(torch.tensor(self.d_model)), dim = -1) # (batch, h, seq_len, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            A.masked_fill_(mask == 0, -1e9)

        
        return A


    def forward(self, q, k, v, mask, previous_attention_scores):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        A = self.attention(query, key, value, mask, self.dropout, previous_attention_scores)

        

        return self.w_o((A @ value).transpose(1, 2).contiguous().view(A.shape[0], -1, self.h * self.d_k)), A



# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     B, T, D, H = 2, 5, 64, 4

#     dummy_q = torch.randn(B, T, D).to(device)
#     dummy_k = torch.randn(B, T, D).to(device)
#     dummy_v = torch.randn(B, T, D).to(device)
#     dummy_mask = torch.ones(B, T).to(device)
#     dummy_prev_scores = torch.randn(B, H, T, T).to(device)

#     model = TASA_attention(d_model=D, h=H, dropout=0.1).to(device)
#     out = model(dummy_q, dummy_k, dummy_v, dummy_mask, dummy_prev_scores)

#     print("✅ Output shape:", out.shape)  # [B, T, D]
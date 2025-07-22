import torch
import torch.nn as nn
import math

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
            in_channels=h, 
            out_channels=h,
            kernel_size=(3, 3),
            padding=1
        )

        self.aggregate_module = nn.Conv2d(
            in_channels=h * 2,
            out_channels=h,
            kernel_size=(3, 3),
            padding=1
        )
        self._init_conv_weights()

    def _init_conv_weights(self):
        # Use small initial weights for stability
        nn.init.normal_(self.transmit_module.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.aggregate_module.weight, mean=0.0, std=0.01)

    def attention(self, query, key, value, mask, dropout, previous_attention_scores):
        M = torch.matmul(query, key.transpose(-2, -1))  # [B, H, T , d] @ [B, H, d, T] --> [B, H, T, T]

        if previous_attention_scores is not None:
            Mt = self.transmit_module(previous_attention_scores)  # CNNᵗ
            Ma_input = torch.cat((M, Mt), dim=1)  # [B, 2H, T, T]
            Ma = self.aggregate_module(Ma_input)  # CNNᵃ
        else:
            Ma = M  # No aggregation in the first layer

        # Normalize then apply mask
        A = Ma / math.sqrt(self.d_k)  
        

        # print("Attention shape:", A.shape)  # [B, H, T, T]

        if mask is not None:
            # print("Mask shape:", mask.shape)  # [B, T]
            # print("Attention shape before mask:", A.shape)  # [B, H, T
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            A = A.masked_fill(mask == 0, -1e9)

        A = A.softmax(dim=-1)  # [B, H, T, T]
        if dropout is not None:
            A = dropout(A)
        return A

    def forward(self, q, k, v, mask=None, previous_attention_scores=None):
        B, T, _ = q.size()

        query = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        A = self.attention(query, key, value, mask, self.dropout, previous_attention_scores)

        out = (A @ value).transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.w_o(out), A


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # print("Mask shape:", mask.shape)  # [B, T]
            # print("attention_scores shape:", attention_scores.shape)  # [B, h, T    , T]
            if mask.dim() != 4:
                mask = mask.unsqueeze(1)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)



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
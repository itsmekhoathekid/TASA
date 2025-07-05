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

    def attention(self, query, key, value, mask, dropout, previous_attention_scores):
        M = torch.matmul(query, key.transpose(-2, -1))  # [B, H, T, T]

        if previous_attention_scores is not None:
            Mt = self.transmit_module(previous_attention_scores)  # CNNᵗ
            Ma_input = torch.cat((M, Mt), dim=1)  # [B, 2H, T, T]
            Ma = self.aggregate_module(Ma_input)  # CNNᵃ
        else:
            Ma = M  # No aggregation in the first layer

        # Normalize then apply mask
        A = torch.softmax(Ma / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=Ma.device)), dim=-1)

        print("Attention shape:", A.shape)  # [B, H, T, T]

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            print("Mask shape:", mask.shape)  # [B, 1, 1, T]
            A = A.masked_fill(mask == 0, -1e9)

        return A

    def forward(self, q, k, v, mask=None, previous_attention_scores=None):
        B, T, _ = q.size()

        query = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        A = self.attention(query, key, value, mask, self.dropout, previous_attention_scores)

        out = (A @ value).transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.w_o(out), A




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
import torch.nn as nn 
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        #here we are first going for dropout,that matrix is passed through the down_projection
        return self.down_proj(self.dropout(activated_x))

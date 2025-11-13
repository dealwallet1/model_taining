
import torch.nn as nn

from model_config import ModelConfig
from repeat import repeat_kv
from rotary import Rotary 
import torch.nn.functional as F 


class Qwen3Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k

        # Separate linear layers for Q, K, V
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # QK-Normalization layers
        # Practice RMSNorm 1 on 1 with ChatGPT - https://chatgpt.com/share/68945c86-2dd4-8002-b017-725caab0c107
        self.q_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)

        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # 1. Project Q, K, V separately
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # 3. Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Apply RoPE
        # Transpose to (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k) for rotary
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Transpose for attention: (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = q.transpose(1, 2)
        K = k.transpose(1, 2)
        V = v.transpose(1, 2)

        # 5. Repeat K and V heads for GQA
        K = repeat_kv(K, self.n_kv_groups)
        V = repeat_kv(V, self.n_kv_groups)

        # 6. Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        # 7. Reshape and final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)
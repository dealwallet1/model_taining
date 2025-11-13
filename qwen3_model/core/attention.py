from configuration.model_config import ModelConfig
import torch.nn as nn 
from .roPE import Rotary
import torch.nn.functional as F
import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
  
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model 
        self.n_heads = config.n_heads 
        self.n_kv_heads = config.n_kv_heads 
        self.n_kv_groups = config.n_kv_groups 
        self.d_k = config.d_k 
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)

        
        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout # dropout value = 0.1

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # 3. Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Apply RoPE
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

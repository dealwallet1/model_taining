from configuration.model_config import ModelConfig
import torch.nn as nn 
from core.attention import Qwen3Attention
from core.dense_layer import SwiGLUFeedForward
from moe_layer.moe import MoE


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):  
        super().__init__()
        self.attention = Qwen3Attention(config)
        #this is for dense model
        #self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        #this is for moe layer(need add this in config file)
        self.feed_forward = MoE(config.d_model, config.n_expert, config.k,config.mult,config.swiglu,config.dropout)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss

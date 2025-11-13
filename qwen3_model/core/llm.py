import torch.nn as nn 
from configuration.model_config import ModelConfig
from .transformer import TransformerBlock
import torch
import math

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        # This ties the output layer (`lm_head`) weights to the input token embedding weights so the model shares parameters between input and output, reducing memory and improving generalization.
        # https://chatgpt.com/share/6894683e-ba44-8002-ae82-e42b4afc9d98
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        total_aux_loss = 0
        for block in self.transformer_blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss


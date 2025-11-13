import torch
import torch.nn as nn
from moe import MoE  # adjust if your file is named differently (e.g. part_5.moe)

class MoEWrapper(nn.Module):
    """
    A wrapper around the MoE layer that handles embeddings and output logits.
    Converts token IDs → embeddings → MoE → vocabulary logits.
    """
    def __init__(self, vocab_size: int, dim: int, n_expert: int = 4, k: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.moe = MoE(dim=dim, n_expert=n_expert, k=k)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x):
        """
        x: (B, T) — token IDs
        returns:
          logits: (B, T, vocab_size)
          aux_loss: scalar (MoE load-balancing loss)
        """
        x = self.embedding(x)        # (B, T, C)
        y, aux_loss = self.moe(x)    # (B, T, C)
        logits = self.output(y)      # (B, T, vocab)
        return logits, aux_loss

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    # Extract dimensions from input tensor
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    # Early return if no repetition is needed
    if n_rep == 1:
        return hidden_states

    # Add a new dimension at index 2 (after num_key_value_heads) and expand
    # Shape transformation:
    # (batch, num_key_value_heads, slen, head_dim)
    # -> (batch, num_key_value_heads, 1, slen, head_dim) [via None indexing]
    # -> (batch, num_key_value_heads, n_rep, slen, head_dim) [via expand]
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)

    # Flatten the num_key_value_heads and n_rep dimensions together
    # Final shape: (batch, num_key_value_heads * n_rep, slen, head_dim)
    # This effectively repeats each key/value head n_rep times
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


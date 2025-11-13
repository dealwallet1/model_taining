from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384 
    n_heads: int = 8 
    n_layers: int = 6 
    d_ff: int = 1536 
    batch_size: int = 24 
    max_steps: int = 2000

    # Qwen3-like parameters
    n_kv_heads: int = 4  
    sliding_window: int = 4096  
    attention_bias: bool = False  
    rms_norm_eps: float = 1e-6  

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01 

    # Data parameters
    max_seq_len: int = 512 
    num_documents: int = 2000
    max_tokens: int = 500000 

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    #moe
    n_expert :int= 8 
    k : int = 2  
    mult: int = 4
    swiglu: bool = True

    #sft training
    bpe_dir:str="finetunning/tokenizer_bpe"
    lr:float = 0.5
    out = "models"

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_groups = self.n_heads // self.n_kv_heads

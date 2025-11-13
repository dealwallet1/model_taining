import math
import os
import pickle
import random
from model_config import ModelConfig
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler 

from datasets import load_dataset

from muon_optimizer import Muon 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")
    
# Data Loading and Caching
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

# Evaluation Function
def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():  # Disable gradient computation for evaluation (saves memory and computation)
        for i, (x, y) in enumerate(val_loader):
            # Stop evaluation after specified number of steps to limit eval time
            if i >= config.eval_steps:
                break

            # Move input sequences (x) and target sequences (y) to GPU/device
            x, y = x.to(device), y.to(device)

            # Use automatic mixed precision if enabled (faster training with minimal accuracy loss)
            with autocast(enabled=config.use_amp):
                # Forward pass: get model predictions (logits) for input sequence
                logits = model(x)

                # Calculate cross-entropy loss between predictions and targets
                # Reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
                # for proper cross-entropy computation across all token positions
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            # Accumulate total loss weighted by number of tokens in this batch
            total_loss += loss.item() * y.numel()
            # Keep track of total number of tokens processed
            total_tokens += y.numel()

            # Get predicted token IDs by taking argmax over vocabulary dimension
            predictions = logits.argmax(dim=-1)
            # Count correct predictions for accuracy calculation
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

# Optimizer Setup
def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and
            'token_embedding' not in name and
            'norm' not in name and
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

from model_config import ModelConfig
import torch
import torch.nn as nn  # Neural network modules like Linear, Embedding, etc.
import torch.nn.functional as F  # Functional interface for operations like cross_entropy, silu, etc.
from torch.utils.data import Dataset, DataLoader  # Base class and utilities for loading datasets
from torch.cuda.amp import autocast, GradScaler  # 🔄 Automatic Mixed Precision (AMP) tools for faster/lower-memory training

import numpy as np  # Numerical computing library, used for random seeding and general array ops

from datasets import load_dataset  # 🧁 Hugging Face Datasets library for streaming large datasets
from tqdm import tqdm  # ⏳ Progress bar visualization library, great for loops

import time  # ⌛ Timing utilities, measuring time
from transformers import AutoTokenizer  # 🤗 Load pretrained tokenizers from HuggingFace with one line

from dataclasses import dataclass  # 🧱 Define simple classes for configs with less boilerplate
from typing import List, Optional  # ✍️ Type hints for better readability and tooling

import warnings  # ⚠️ Suppress or handle warnings
import os  # 🗂️ File system operations (creating folders, path checking, etc.)
import pickle

from dataset import TextTokenDataset
from trainin_model import train_model
from utilities import load_and_cache_data, set_seed  # 💾 Python object serialization (used to save/load preprocessed datasets)

warnings.filterwarnings('ignore') 

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
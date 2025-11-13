import torch
import torch.nn as nn 
from configuration.model_config import ModelConfig
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F
import math


def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    #put model in evaluation mode
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    #it will predict the model in gpu or cpu
    device = next(model.parameters()).device

    with torch.no_grad(): 
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
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            predictions = logits.argmax(dim=-1)
            
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    #again we will reset the model to training phase
    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

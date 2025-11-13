import math
import time
from model_config import ModelConfig
import torch
from tqdm import tqdm
from moe import MoE
from moe_wrapper import MoEWrapper
from utilities import evaluate_model, set_seed, setup_muon_optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the MoE model with Muon optimizer"""
    print(f"\n🚀 Training Mixture-of-Experts (MoE) model with Muon optimizer")

    # Initialize model
    set_seed(42)
    model = MoEWrapper(
    vocab_size=config.vocab_size,
    dim=config.d_model,
    n_expert=getattr(config, "num_experts", 4),
    k=getattr(config, "top_k", 1)
    ).to(config.device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float("inf")

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # ---- Forward pass ----
            if config.use_amp:
                with autocast():
                    logits, aux_loss = model(x)
                    loss_main = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss_main + config.moe_aux_loss_weight * aux_loss
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x)
                loss_main = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss_main + config.moe_aux_loss_weight * aux_loss
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # ---- Optimizer step ----
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # ---- Logging ----
            if step % 10 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{accuracy:.3f}",
                    "ppl": f"{perplexity:.1f}",
                    "lr": f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # ---- Evaluation ----
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics["val_loss"] < best_val_loss:
                    best_val_loss = eval_metrics["val_loss"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "final_metrics": eval_metrics
                    }, "best_model.pt")
                    print(f"💾 Saved best model with val_loss: {best_val_loss:.4f}")

            step += 1
            if step % 10 == 0:
                pbar.update(10)

    pbar.close()
    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # ---- Final evaluation ----
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, "
          f"PPL: {final_eval['val_perplexity']:.2f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "final_metrics": final_eval
    }, "final_model.pt")
    print("💾 Saved final model to final_model.pt")

    return model, final_eval

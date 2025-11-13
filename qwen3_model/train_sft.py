from __future__ import annotations
import argparse, torch
import torch.nn as nn
from pathlib import Path
torch.manual_seed(0)
from inference.inference import load_trained_model  
from fine_tunning.dataset_sft import load_tiny_hf
from fine_tunning.collator_sft import SFTCollator
from fine_tunning.curriculum import LengthCurriculum
from configuration.model_config import ModelConfig



def main():
 
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a tiny HF slice or fallback examples
    items = load_tiny_hf(split='train[:24]', sample_dataset=False)

    # Print few samples
    print(f"Loaded {len(items)} SFT items. Few samples:")
    for it in items[:3]:
        print(f"PROMPT: {it.prompt}\nRESPONSE: {it.response}\n{'-'*40}")

    # Curriculum over (prompt,response)
    tuples = [(it.prompt, it.response) for it in items]
    cur = list(LengthCurriculum(tuples))
    print(cur)

    # Collator + model
    col = SFTCollator(block_size=config.max_seq_len ,bpe_dir=config.bpe_dir)
    #here we are utilising the already trained model for finetunning
    model,model_config = load_trained_model("/models/final_model.pt")
    model.to(device)

    # if args.ckpt:
    #     print(f"Using model config from checkpoint {args.ckpt}")
    #     ckpt = torch.load(args.ckpt, map_location=device)
    #     cfg = ckpt.get('config', {})
    #     model.load_state_dict(ckpt['model'])

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
  


    # Simple loop (single machine). We just cycle curriculum to fill batches, for a few steps.
    step = 0
    i = 0
    while step < config.max_steps:
        batch = cur[i:i+config.batch_size]
        if not batch:
            # restart curriculum
            # cur = list(LengthCurriculum(tuples)); 
            i = 0
            continue
        xb, yb = col.collate(batch)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1; i += config.batch_size
        if step % 20 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    Path(config.out).mkdir(parents=True, exist_ok=True)
    cfg = {
        "vocab_size": col.vocab_size,
        "block_size": config.max_seq_len,
        "n_layer": config.n_layers,
        "n_head": config.n_heads,
        "n_embd": config.d_model,
        "dropout": 0.0,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "rope": True,
        # tokenizer info (best-effort)
        "tokenizer_type": "byte" if col.vocab_size == 256 else "bpe",
        "tokenizer_dir": None,   # set a real path if you have a trained BPE dir
    }
    torch.save({'model': model.state_dict(), 'config': cfg},
               str(Path(config.out)/'model_last.pt'))
    print(f"Saved SFT checkpoint to {config.out}/model_last.pt")

if __name__ == '__main__':
    main()
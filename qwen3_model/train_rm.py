from __future__ import annotations
import argparse, torch
from pathlib import Path

from rewardModel.data_prefs import load_preferences
from rewardModel.collator_rm import PairCollator
from rewardModel.model_reward import RewardModel
from rewardModel.loss_reward import bradley_terry_loss, margin_ranking_loss
from configuration.model_config import ModelConfig


def main():
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    items = load_preferences(split='train[:80]')
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]

    # collator + model
    col = PairCollator(block_size=config.max_seq_len, bpe_dir=config.bpe_dir)
    model = RewardModel(vocab_size=col.vocab_size, block_size=config.max_seq_len,
                        n_layer=config.n_layers, n_head=config.n_heads, n_embd=config.d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999))

    # train (tiny)
    step = 0; i = 0
    while step < config.max_steps:
        batch = triples[i:i+config.batch_size]
        if not batch:
            i = 0; continue
        pos, neg = col.collate(batch)
        pos, neg = pos.to(device), neg.to(device)
        r_pos = model(pos)
        r_neg = model(neg)
        if  config.loss == 'bt':
            loss = bradley_terry_loss(r_pos, r_neg)
        else:
            loss = margin_ranking_loss(r_pos, r_neg, margin=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1; i += config.batch_size
        if step % 25 == 0:
            acc = (r_pos > r_neg).float().mean().item()
            print(f"step {step}: loss={loss.item():.4f} acc={acc:.2f}")

    Path(config.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': {
        'vocab_size': col.vocab_size,
        'block_size': config.max_seq_len,
        'n_layer': config.n_layers,
        'n_head': config.n_heads,
        'n_embd': config.d_model,
    }}, str(Path(config.out)/'model_last.pt'))
    print(f"Saved reward model to {config.out}/model_last.pt")

if __name__ == '__main__':
    main()